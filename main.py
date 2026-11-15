"""FINN - push/pull investor agent. Run: uvicorn main:app --host 0.0.0.0 --port 8000"""
import os,re,json,sqlite3,hashlib,math,random,datetime as dt,io,logging
from pathlib import Path
from collections import Counter
import requests
from dotenv import load_dotenv
from fastapi import Body,FastAPI,HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel,Field
from pypdf.errors import PdfReadError
from typing import Literal

APP_DIR=Path(__file__).resolve().parent
load_dotenv(APP_DIR/".env")
PORTFOLIO="RELIANCE TCS HDFCBANK INFY ICICIBANK ITC SUNPHARMA MARUTI TATAMOTORS LT ADANIPORTS ASIANPAINT TITAN ZOMATO HINDUNILVR BHARTIARTL".split()
BASE=os.getenv("LLM_BASE_URL","https://api.groq.com/openai/v1").rstrip("/")
KEY=os.getenv("LLM_KEY","");CHEAP=os.getenv("LLM_CHEAP","llama-3.1-8b-instant");STRONG=os.getenv("LLM_STRONG","llama-3.3-70b-versatile")
db=sqlite3.connect(os.getenv("FINN_DB_PATH",str(APP_DIR/"finn.db")),check_same_thread=False);db.row_factory=sqlite3.Row
db.executescript("""create table if not exists prices(sym,d,close,vol,primary key(sym,d));
create table if not exists ann(h primary key,sym,ts,title,url,cat,score,summary,extra,demo);
create table if not exists chunks(sym,doc,page,txt);create table if not exists kv(k primary key,v);
create table if not exists cache(h primary key,v)""")
def kv(k,d=None):
    r=db.execute("select v from kv where k=?",(k,)).fetchone();return json.loads(r[0]) if r else d
def setkv(k,v): db.execute("replace into kv values(?,?)",(k,json.dumps(v)));db.commit()

# ---------- LLM (cheap model for bulk, strong for reasoning, cached by prompt hash) ----------
def llm(prompt,strong=False):
    if not (BASE and KEY): return None
    m=STRONG if strong else CHEAP;h=hashlib.md5((m+prompt).encode()).hexdigest()
    r=db.execute("select v from cache where h=?",(h,)).fetchone()
    if r: return r[0]
    try:
        response=requests.post(BASE+"/chat/completions",headers={"Authorization":"Bearer "+KEY},timeout=60,
          json={"model":m,"temperature":0,"messages":[{"role":"user","content":prompt}]})
        response.raise_for_status()
        x=response.json()["choices"][0]["message"]["content"]
    except (requests.RequestException,KeyError,IndexError,TypeError,ValueError) as e:
        logging.warning("LLM request failed: %s",e)
        return None
    db.execute("insert or replace into cache values(?,?)",(h,x));db.commit();return x
def jparse(s):
    try: return json.loads(re.search(r"\{.*\}",s,re.S).group(0))
    except Exception: return None

# ---------- classification: rules first (free), LLM only adds one-line summary for important items ----------
RULES=[("results",r"financial result|quarter|audited|unaudited|results"),("M&A",r"acqui|merger|amalgam|demerger|joint venture|subsidiary"),
("fund_raise",r"rights issue|qip|preferential|buy-?back|commercial paper|debenture|fund ?raising"),("management_change",r"resign|appoint|cessation|retire|chief financial|ceo|managing director"),
("litigation",r"court|penalty|fine|show cause|insolvency|nclt|sebi order|tribunal"),("insider_trading",r"insider|sast|regulation 7|trading window"),
("earnings_call",r"earnings call|concall|conference call|transcript|analyst"),("investor_meetings",r"investor meet|analyst meet|schedule of"),
("credit_rating",r"credit rating|crisil|icra|care ratings|india ratings"),("dividend",r"dividend|record date"),("bonus_split",r"bonus|split|sub-?division"),
("capex",r"capex|capacity expansion|new plant|facility"),("pledging",r"pledge|encumbrance"),("postal_ballot",r"postal ballot"),("agm_egm",r"\bagm\b|\begm\b|general meeting"),
("press_release",r"press release|order|award|launch|partnership|clarification")]
BASE_SCORE=dict(results=90,M_A=0,fund_raise=70,management_change=80,litigation=75,insider_trading=35,earnings_call=45,investor_meetings=15,credit_rating=60,
 dividend=55,bonus_split=55,capex=60,pledging=65,postal_ballot=20,agm_egm=20,press_release=40,unclassified=5);BASE_SCORE["M&A"]=85
def classify(t):
    for c,p in RULES:
        if re.search(p,t,re.I): return c
    return "unclassified"
def score(cat,t):
    s=BASE_SCORE[cat]
    if re.search(r"material|definitive|downgrade|default|fraud|resign",t,re.I): s+=10   # urgency boosters
    if re.search(r"intimation of (trading window|schedule)|newspaper|certificate",t,re.I): s-=25
    return max(0,min(100,s))
def pdf_text(url,maxp=None):
    from pypdf import PdfReader
    r=requests.get(url,timeout=40,headers={"User-Agent":"Mozilla/5.0"});r.raise_for_status();out=[]
    for i,p in enumerate(PdfReader(io.BytesIO(r.content)).pages[:maxp]): out.append((i+1,p.extract_text() or ""))
    return out
def pct(a,b): return None if a is None or not b else round((a-b)/abs(b)*100,1)   # Python does the maths, never the LLM

def nse_fetch(sym):
    s=requests.Session();s.headers["User-Agent"]="Mozilla/5.0";s.get("https://www.nseindia.com",timeout=15)
    j=s.get("https://www.nseindia.com/api/corporate-announcements",params={"index":"equities","symbol":sym},timeout=20).json()
    inserted=0
    for a in j[:40]:
        t=(a.get("desc","")+" - "+(a.get("attchmntText") or ""));ts=a.get("an_dt","")[:20]
        try: ts=dt.datetime.strptime(ts,"%d-%b-%Y %H:%M:%S").isoformat()
        except Exception: ts=dt.datetime.now().isoformat()
        h=hashlib.md5((sym+a.get("attchmntFile","")+t).encode()).hexdigest();c=classify(t);sc=score(c,t)
        summ=None
        if sc>=60: summ=llm(f"One plain sentence (max 25 words) on why an investor cares about this NSE filing for {sym}: {t[:600]}")
        cur=db.execute("insert or ignore into ann values(?,?,?,?,?,?,?,?,?,0)",(h,sym,ts,t[:300],a.get("attchmntFile"),c,sc,summ,None))
        inserted+=max(cur.rowcount,0)
    db.commit()
    return inserted

def seed_demo():
    """Offline fallback so the app always runs. Everything is flagged DEMO in the UI."""
    random.seed(7);today=dt.date.today()
    for s in PORTFOLIO:
        p=random.uniform(300,3000)
        for i in range(60,-1,-1):
            p*=1+random.gauss(0,.013);v=random.uniform(.8,1.2)*1e6*(3 if i==0 and random.random()<.3 else 1)
            db.execute("insert or replace into prices values(?,?,?,?)",(s,str(today-dt.timedelta(i)),p,v))
    ex=[("results","Outcome of Board Meeting - Unaudited Financial Results for the quarter"),("management_change","Resignation of Chief Financial Officer"),
        ("M&A","Definitive agreement to acquire 60% stake in subsidiary"),("dividend","Interim dividend declared, record date fixed"),("unclassified","Compliance certificate under Reg 74(5)"),("credit_rating","CRISIL reaffirms AAA rating"),("press_release","Receipt of order worth Rs 1,200 crore")]
    for i,(c,t) in enumerate(ex*3):
        s=PORTFOLIO[i*2%len(PORTFOLIO)];ts=(dt.datetime.now()-dt.timedelta(hours=random.randint(1,70))).isoformat()
        db.execute("insert or ignore into ann values(?,?,?,?,?,?,?,?,?,1)",(f"d{i}",s,ts,t,None,c,score(c,t),"Illustrative demo item - not a real disclosure.",None))
    db.commit()

def demo_story():
    """A connected, entirely fictional case for demonstrating FINN's workflows."""
    today=dt.date.today()
    closes=[91.0,90.5,92.0,91.3,93.0,94.0,92.6,95.0,96.0,95.4,97.0,96.2,98.0,97.5,99.2,98.8,100.0,99.4,100.0,104.2]
    prices=[]
    for i,close in enumerate(closes):
        day=today-dt.timedelta(days=len(closes)-1-i)
        prices.append({"d":str(day),"close":close,"vol":3200000 if i==len(closes)-1 else 1000000})
    metrics={
        "revenue":{"cur":1210,"prev_q":1100,"yago":1000},
        "ebitda":{"cur":131,"prev_q":126,"yago":108},
        "pbt":{"cur":82,"prev_q":78,"yago":70},
        "pat":{"cur":61,"prev_q":58,"yago":50},
    }
    for values in metrics.values():
        values["qoq"]=pct(values["cur"],values["prev_q"])
        values["yoy"]=pct(values["cur"],values["yago"])
    return {
        "sym":"DEMOCO",
        "company":"Aster Mobility",
        "disclaimer":"Fictional company and illustrative figures; not real market or company data.",
        "move_pct":pct(prices[-1]["close"],prices[-2]["close"]),
        "volume_multiple":round(prices[-1]["vol"]/(sum(p["vol"] for p in prices[-21:-1])/len(prices[-21:-1])),1),
        "prices":prices,
        "metrics":{"unit":"₹ crore","basis":"Illustrative consolidated","values":metrics},
        "events":[
            {"kind":"Guidance","when":"Prior earnings call","text":"Management guided to FY27 revenue growth of 20–25% and a Q2 EBITDA margin of at least 12%."},
            {"kind":"Quarterly results","when":"Latest quarter","text":"Revenue grew 21% YoY, within the guided range. EBITDA margin was 10.8%, 1.2 percentage points below the 12% target."},
            {"kind":"Price & volume","when":"Latest session","text":"The share closed up 4.2% on volume 3.2× its illustrative 20-session average."},
            {"kind":"Routine","when":"Same period","text":"Routine compliance certificate — shown as low priority and collapsed."},
        ],
        "sources":[
            {"n":1,"doc":"DEMO · Aster Mobility FY27 Q1 earnings-call transcript","page":18,"excerpt":"Illustrative management guidance: FY27 revenue growth of 20%–25%; target EBITDA margin of at least 12% in Q2 FY27."},
            {"n":2,"doc":"DEMO · Aster Mobility Q2 FY27 results filing","page":6,"excerpt":"Illustrative Q2 FY27 revenue ₹1,210 crore, up 21% year-on-year. EBITDA ₹131 crore; EBITDA margin 10.8%."},
        ],
    }

app=FastAPI(title="FINN")
@app.get("/api/demo/story")
def get_demo_story(): return demo_story()

@app.get("/api/portfolio")
def portfolio(): return kv("portfolio",PORTFOLIO)
class Sync(BaseModel): demo:bool=False
@app.post("/api/sync")
def sync(b:Sync):
    if b.demo: seed_demo();return {"mode":"demo"}
    import yfinance as yf
    err=[];price_rows=0;announcement_rows=0
    for s in portfolio():
        try:
            h=yf.Ticker(s+".NS").history(period="3mo")
            if h.empty:
                err.append(f"{s} (Yahoo: no price history)")
            else:
                for d,r in h.iterrows():
                    db.execute("insert or replace into prices values(?,?,?,?)",(s,str(d.date()),float(r.Close),float(r.Volume)))
                    price_rows+=1
        except Exception as e:
            logging.warning("Yahoo Finance sync failed for %s: %s",s,e)
            err.append(f"{s} (Yahoo: {type(e).__name__})")
        try:
            announcement_rows+=nse_fetch(s)
        except Exception as e:
            logging.warning("NSE sync failed for %s: %s",s,e)
            err.append(f"{s} (NSE: {type(e).__name__})")
    db.commit();return {"mode":"live","price_rows":price_rows,"announcement_rows":announcement_rows,"errors":err}

@app.get("/api/settings")
def get_settings(): return kv("settings",{"vol_x":2.0,"move_pct":3.0,"cats":["results","M&A","management_change","litigation","fund_raise"]})

class Settings(BaseModel):
    vol_x:float=Field(default=2.0,gt=0,le=100)
    move_pct:float=Field(default=3.0,gt=0,le=100)
    cats:list[str]=Field(default_factory=lambda:["results","M&A","management_change","litigation","fund_raise"])

@app.post("/api/settings")
def put_settings(s:Settings): setkv("settings",s.dict());return s.dict()

@app.get("/api/digest")
def digest(hours:int=24):
    cut=(dt.datetime.now()-dt.timedelta(hours=hours)).isoformat()
    return [dict(r) for r in db.execute("select h,sym,ts,title,url,cat,score,summary,demo from ann where ts>=? order by score desc,ts desc",(cut,))]

@app.get("/api/alerts")
def alerts():
    st=get_settings();out=[]
    for s in portfolio():
        r=db.execute("select d,close,vol from prices where sym=? order by d desc limit 21",(s,)).fetchall()
        if len(r)<3: continue
        chg=pct(r[0]["close"],r[1]["close"]);avg=sum(x["vol"] for x in r[1:])/len(r[1:]);vr=r[0]["vol"]/avg if avg else 0
        if vr>=st["vol_x"]: out.append(dict(sym=s,type="volume",d=r[0]["d"],msg=f"Volume {vr:.1f}x its {len(r)-1}-day average",val=round(vr,1)))
        if abs(chg)>=st["move_pct"]: out.append(dict(sym=s,type="price",d=r[0]["d"],msg=f"Closed {chg:+.1f}% on the day",val=chg))
    for a in db.execute("select * from ann order by ts desc limit 200"):
        if a["cat"] in st["cats"]: out.append(dict(sym=a["sym"],type="filing",d=a["ts"][:10],msg=f"{a['cat']}: {a['title'][:90]}",val=a["score"],demo=a["demo"]))
    return out

@app.get("/api/prices/{sym}")
def prices(sym:str): return [dict(r) for r in db.execute("select d,close,vol from prices where sym=? order by d",(sym.upper(),))]

@app.get("/api/results/{h}")
def results(h:str):
    """Headline numbers: LLM extracts raw columns from the filing PDF, Python computes QoQ/YoY."""
    a=db.execute("select * from ann where h=?",(h,)).fetchone()
    if not a or not a["url"]: raise HTTPException(404,"No filing PDF for this item")
    txt="\n".join(t for _,t in pdf_text(a["url"],6))[:12000]
    j=jparse(llm('From this results filing extract numbers for the standalone/consolidated column set (prefer consolidated) as JSON only: {"unit":"","basis":"","metrics":{"revenue":{"cur":0,"prev_q":0,"yago":0},"ebitda":{...},"pbt":{...},"pat":{...}}} use null if absent. Text:\n'+txt,strong=False) or "")
    if not j: raise HTTPException(502,"Could not extract (LLM unavailable or table unreadable). Open the filing instead.")
    for m in j["metrics"].values(): m["qoq"]=pct(m.get("cur"),m.get("prev_q"));m["yoy"]=pct(m.get("cur"),m.get("yago"))
    return j

# ---------- PULL: primary-source RAG (annual reports + concalls from screener.in), BM25 retrieval ----------
tok=lambda s:re.findall(r"[a-z0-9]+",s.lower())
@app.post("/api/ingest/{sym}")
def ingest(sym:str,urls:list[str]=Body(default=[])):
    sym=sym.upper()
    if not urls:
        from bs4 import BeautifulSoup
        try:
            response=requests.get(f"https://www.screener.in/company/{sym}/consolidated/",headers={"User-Agent":"Mozilla/5.0"},timeout=20)
            response.raise_for_status()
            soup=BeautifulSoup(response.text,"html.parser")
        except requests.RequestException as e: raise HTTPException(502,"screener.in unreachable - paste document URLs instead") from e
        urls=[a["href"] for a in soup.select("a[href]") if re.search(r"(annual|transcript|concall|\.pdf)",a["href"]+a.text,re.I)][:5]
    n=0;errors=[]
    for u in urls:
        if db.execute("select 1 from chunks where sym=? and doc=? limit 1",(sym,u)).fetchone(): continue   # cache
        try:
            for pg,t in pdf_text(u,150):
                for i in range(0,len(t),1200): db.execute("insert into chunks values(?,?,?,?)",(sym,u,pg,t[i:i+1400]));n+=1
        except (requests.RequestException,PdfReadError,ValueError,OSError) as e:
            errors.append({"doc":u,"error":str(e)})
    db.commit();return {"chunks_added":n,"docs":urls,"errors":errors}
def retrieve(sym,q,k=6):
    rows=db.execute("select doc,page,txt from chunks where sym=?",(sym,)).fetchall();qt=tok(q);N=len(rows) or 1
    df=Counter(w for r in rows for w in set(tok(r["txt"])));sc=[]
    for r in rows:
        tf=Counter(tok(r["txt"]));L=sum(tf.values()) or 1
        sc.append((sum(math.log(1+N/(1+df[w]))*tf[w]*2.2/(tf[w]+1.2*(.25+.75*L/180)) for w in qt if w in tf),r))
    return [r for s,r in sorted(sc,key=lambda x:-x[0])[:k] if s>0]
PROMPTS={"ask":"Answer the question","guidance":"List what management explicitly guided (growth, margins, capex) by period, then what was actually delivered, and mark each met/missed/unclear with the gap","bullbear":"Give a tight bull case and bear case (3 bullets each)"}
class Ask(BaseModel):
    q:str=""
    mode:Literal["ask","guidance","bullbear"]="ask"
@app.post("/api/research/{sym}")
def research(sym:str,b:Ask):
    sym=sym.upper()
    if sym=="DEMOCO":
        story=demo_story()
        if b.mode=="guidance":
            answer=("Promised vs delivered (fictional demo): FY27 revenue growth guidance was 20–25%; Q2 revenue grew 21% YoY, within range [1][2]. "
                    "The Q2 EBITDA-margin target was at least 12%; the reported illustrative margin was 10.8%, missing by 1.2 percentage points [1][2].")
        elif b.mode=="bullbear":
            answer=("Illustrative bull case: reported revenue growth is within management's stated range [1][2].\n"
                    "Illustrative bear case: the quarter's EBITDA margin missed management's stated target by 1.2 percentage points [1][2].")
        else:
            answer="This fictional case includes only the guidance and reported figures shown in the cited excerpts. Ask about revenue growth or the EBITDA-margin target."
        return {"answer":answer,"sources":story["sources"]}
    q=b.q or {"guidance":"management guidance outlook growth margin capex target","bullbear":"risks growth strengths outlook"}.get(b.mode,"business")
    ch=retrieve(sym,q)
    if not ch: return {"answer":"No primary documents found for this company yet. Tap 'Load documents' first. FINN will not answer from general knowledge.","sources":[]}
    ctx="\n\n".join(f"[{i+1}] ({c['doc'].split('/')[-1]} p.{c['page']}) {c['txt']}" for i,c in enumerate(ch))
    a=llm(f"{PROMPTS[b.mode]}. Use ONLY the numbered excerpts; cite like [1]; say 'insufficient evidence' if not supported; never use outside knowledge; be concise.\nQuestion: {q}\n\n{ctx}",strong=True)
    return {"answer":a or "LLM key not set - showing the most relevant excerpts.","sources":[{"n":i+1,"doc":c["doc"],"page":c["page"],"excerpt":c["txt"][:300]} for i,c in enumerate(ch)]}

@app.get("/")
def home(): return FileResponse(APP_DIR/"index.html")
