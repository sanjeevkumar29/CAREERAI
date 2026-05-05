import os
import re
import operator
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List, Optional
 
from langsmith import Client as LangSmithClient
from langsmith import traceable
 

from langgraph.graph import StateGraph, END
 

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
 

from langchain_groq import ChatGroq
 

from langchain_openai import ChatOpenAI
 
load_dotenv()
 

 
def init_langsmith() -> Optional[LangSmithClient]:
    """
    Sets LANGCHAIN_* env vars so every subsequent LangChain/LangGraph call
    is automatically traced in the LangSmith dashboard.
    """
    api_key = os.getenv("LANGSMITH_API_KEY", "")
    project = os.getenv("LANGSMITH_PROJECT", "CareerAI-Oracle")
    if not api_key:
        return None
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"]     = project
    os.environ["LANGCHAIN_API_KEY"]     = api_key
    os.environ["LANGCHAIN_ENDPOINT"]    = "https://api.smith.langchain.com"
    try:
        return LangSmithClient(api_key=api_key)
    except Exception:
        return None
 
_ls_client = init_langsmith()   
 

 
st.set_page_config(
    page_title="CareerAI Oracle",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)
 

 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');
 
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html,body,.stApp{background:#020817!important;font-family:'Inter',sans-serif!important;color:#e2e8f0!important}
 
.stApp::before{
    content:'';position:fixed;inset:0;z-index:-1;
    background:
      radial-gradient(ellipse 80% 60% at 10% 10%,rgba(99,102,241,0.18) 0%,transparent 60%),
      radial-gradient(ellipse 60% 50% at 90% 85%,rgba(16,185,129,0.14) 0%,transparent 55%),
      radial-gradient(ellipse 70% 50% at 50% 50%,rgba(6,182,212,0.07) 0%,transparent 70%),
      #020817;
    animation:bgP 12s ease-in-out infinite alternate;
}
@keyframes bgP{0%{opacity:1}100%{opacity:0.85}}
 
::-webkit-scrollbar{width:6px}
::-webkit-scrollbar-track{background:#0f172a}
::-webkit-scrollbar-thumb{background:#334155;border-radius:3px}
 
header[data-testid="stHeader"]{background:transparent!important}
.stDeployButton,#MainMenu,footer{display:none!important}
section[data-testid="stSidebar"]{background:rgba(15,23,42,0.96)!important;border-right:1px solid rgba(99,102,241,0.2)!important;backdrop-filter:blur(20px)}
 
/* Pipeline banner */
.pipe-banner{display:flex;align-items:center;justify-content:center;gap:0.4rem;flex-wrap:wrap;
  background:rgba(15,23,42,0.7);border:1px solid rgba(99,102,241,0.2);border-radius:12px;
  padding:0.6rem 1.2rem;margin:0.6rem auto 1.4rem;max-width:700px;font-size:0.78rem}
.pipe-step{display:flex;align-items:center;gap:5px;padding:4px 13px;border-radius:99px;font-family:'JetBrains Mono',monospace;font-weight:600}
.ps-ls {background:rgba(99,102,241,0.18);border:1px solid rgba(99,102,241,0.45);color:#a5b4fc}
.ps-gr {background:rgba(251,146,60,0.15);border:1px solid rgba(251,146,60,0.45);color:#fb923c}
.ps-oai{background:rgba(52,211,153,0.13);border:1px solid rgba(52,211,153,0.45);color:#34d399}
.pipe-arrow{color:#475569;font-size:1rem}
 
/* Hero */
.hero-title{text-align:center;padding:2.2rem 1rem 0.6rem;animation:fadeInDown 0.7s ease-out}
.hero-title h1{font-size:clamp(2rem,5vw,3.3rem);font-weight:900;
  background:linear-gradient(135deg,#818cf8 0%,#34d399 50%,#22d3ee 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
  letter-spacing:-1px;line-height:1.1}
.hero-sub{font-size:0.93rem;color:#475569;margin-top:0.45rem}
 
/* Glass cards */
.glass-card{background:rgba(15,23,42,0.75);border:1px solid rgba(99,102,241,0.2);border-radius:16px;
  padding:1.6rem;backdrop-filter:blur(16px);box-shadow:0 4px 40px rgba(0,0,0,0.4);margin-bottom:1.2rem;animation:fadeInUp 0.55s ease-out}
.gc-green {border-color:rgba(52,211,153,0.3);box-shadow:0 0 30px rgba(52,211,153,0.07),0 4px 40px rgba(0,0,0,0.4)}
.gc-purple{border-color:rgba(139,92,246,0.3);box-shadow:0 0 30px rgba(139,92,246,0.07),0 4px 40px rgba(0,0,0,0.4)}
 
/* Stage insight boxes */
.stage-box{border-radius:12px;padding:1.05rem 1.35rem;margin-top:0.7rem;position:relative;overflow:hidden}
.sb-groq {background:linear-gradient(135deg,rgba(251,146,60,0.08),rgba(245,158,11,0.05));border:1px solid rgba(251,146,60,0.25)}
.sb-oai  {background:linear-gradient(135deg,rgba(52,211,153,0.08),rgba(34,211,238,0.05));border:1px solid rgba(52,211,153,0.25)}
.stage-label{font-size:0.67rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.45rem;display:flex;align-items:center;gap:6px}
.sb-groq .stage-label{color:#fb923c}
.sb-oai  .stage-label{color:#34d399}
.stage-content{font-size:0.86rem;color:#cbd5e1;line-height:1.72}
 
/* Career info */
.career-name{font-size:1.65rem;font-weight:800;color:#e2e8f0;line-height:1.2;margin-bottom:0.3rem}
.sec-label{font-size:0.67rem;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;color:#6366f1;margin-bottom:0.3rem}
.career-summary{font-size:0.87rem;color:#94a3b8;line-height:1.65;margin-top:0.45rem}
.risk-badge{display:inline-flex;align-items:center;gap:6px;border-radius:99px;padding:4px 13px;font-size:0.74rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em}
.risk-low   {background:rgba(16,185,129,0.12);border:1px solid rgba(16,185,129,0.4);color:#34d399}
.risk-medium{background:rgba(251,191,36,0.12);border:1px solid rgba(251,191,36,0.4);color:#fbbf24}
.risk-high  {background:rgba(239,68,68,0.12); border:1px solid rgba(239,68,68,0.4); color:#f87171}
 
/* Score ring */
.score-wrapper{display:flex;justify-content:center;padding:0.6rem 0}
.score-ring{width:112px;height:112px;border-radius:50%;display:flex;flex-direction:column;align-items:center;justify-content:center;font-weight:900;position:relative}
.score-ring::before{content:'';position:absolute;inset:5px;border-radius:50%;background:#0f172a}
.snum{font-size:1.85rem;position:relative;z-index:1}
.slbl{font-size:0.6rem;letter-spacing:0.1em;text-transform:uppercase;position:relative;z-index:1;color:#64748b}
 
/* Metric chips */
.metric-grid{display:flex;flex-wrap:wrap;gap:0.5rem;margin:0.75rem 0}
.metric-chip{background:rgba(30,41,59,0.8);border:1px solid rgba(99,102,241,0.18);border-radius:10px;padding:0.5rem 0.85rem;flex:1;min-width:85px;text-align:center}
.mc-val{font-size:1.25rem;font-weight:800;line-height:1}
.mc-lbl{font-size:0.6rem;color:#64748b;text-transform:uppercase;letter-spacing:0.08em;margin-top:3px}
 
/* Progress bars */
.prog-wrap{margin:0.4rem 0}
.prog-hdr{display:flex;justify-content:space-between;font-size:0.74rem;margin-bottom:4px;color:#94a3b8}
.prog-bg{height:8px;border-radius:99px;background:rgba(30,41,59,0.8);overflow:hidden}
.prog-fill{height:100%;border-radius:99px}
 
/* Skills */
.skills-wrap{display:flex;flex-wrap:wrap;gap:0.38rem;margin-top:0.6rem}
.skill-pill{background:rgba(99,102,241,0.12);border:1px solid rgba(99,102,241,0.25);border-radius:99px;padding:3px 11px;font-size:0.71rem;color:#a5b4fc;font-family:'JetBrains Mono',monospace}
 
/* Compare */
.cmp-row{display:grid;grid-template-columns:1fr 2fr 2fr;gap:0.45rem;align-items:center;padding:0.5rem 0;border-bottom:1px solid rgba(30,41,59,0.6)}
.cmp-mname{font-size:0.73rem;color:#64748b;text-transform:uppercase;letter-spacing:0.06em}
.cmp-bwrap{display:flex;align-items:center;gap:7px;font-size:0.78rem}
.cmp-val{font-weight:700;min-width:38px}
.winner-badge{display:inline-flex;align-items:center;gap:6px;background:linear-gradient(135deg,rgba(251,191,36,0.2),rgba(245,158,11,0.1));border:1px solid rgba(251,191,36,0.5);border-radius:99px;padding:6px 18px;color:#fbbf24;font-weight:700;font-size:0.83rem;animation:glow 2s ease-in-out infinite alternate}
@keyframes glow{0%{box-shadow:0 0 8px rgba(251,191,36,0.3)}100%{box-shadow:0 0 20px rgba(251,191,36,0.5)}}
 
/* Sidebar trending */
.trend-item{background:rgba(30,41,59,0.5);border:1px solid rgba(99,102,241,0.13);border-radius:10px;padding:0.55rem 0.8rem;margin-bottom:0.42rem;display:flex;justify-content:space-between;align-items:center}
.trend-name{font-size:0.79rem;color:#e2e8f0;font-weight:600}
.trend-badge{border-radius:99px;padding:2px 9px;font-size:0.68rem;font-weight:700}
 
/* Inputs */
.stTextInput>div>div>input,.stTextArea>div>div>textarea{background:rgba(15,23,42,0.8)!important;border:1.5px solid rgba(99,102,241,0.3)!important;border-radius:10px!important;color:#e2e8f0!important;font-family:'Inter',sans-serif!important;font-size:0.92rem!important}
.stTextInput>div>div>input:focus{border-color:rgba(99,102,241,0.7)!important;box-shadow:0 0 0 3px rgba(99,102,241,0.1)!important}
.stSelectbox>div>div{background:rgba(15,23,42,0.8)!important;border:1.5px solid rgba(99,102,241,0.3)!important;border-radius:10px!important}
.stButton>button{background:linear-gradient(135deg,#6366f1,#4f46e5)!important;color:white!important;border:none!important;border-radius:10px!important;padding:0.58rem 1.7rem!important;font-weight:700!important;font-size:0.87rem!important;width:100%!important;box-shadow:0 4px 20px rgba(99,102,241,0.35)!important;transition:all 0.2s!important}
.stButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 8px 30px rgba(99,102,241,0.5)!important}
 
/* Tabs */
.stTabs [data-baseweb="tab-list"]{background:rgba(15,23,42,0.6)!important;border-radius:12px!important;padding:4px!important;gap:4px!important;border:1px solid rgba(99,102,241,0.15)!important}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:#64748b!important;border-radius:8px!important;font-weight:600!important;font-size:0.83rem!important;padding:0.46rem 1.05rem!important}
.stTabs [aria-selected="true"]{background:rgba(99,102,241,0.25)!important;color:#a5b4fc!important}
.stTabs [data-baseweb="tab-panel"]{padding-top:1.1rem!important}
 
hr{border-color:rgba(30,41,59,0.8)!important}
@keyframes fadeInDown{from{opacity:0;transform:translateY(-18px)}to{opacity:1;transform:translateY(0)}}
@keyframes fadeInUp  {from{opacity:0;transform:translateY(18px)} to{opacity:1;transform:translateY(0)}}
</style>
""", unsafe_allow_html=True)
 

 
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    df.columns = df.columns.str.strip()
    return df
 
DF = load_data()
 

def get_groq_llm():
    """Stage 2 – Groq LLaMA3-70B: blazing-fast raw career analysis."""
    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        return None
    try:
        return ChatGroq(model="llama3-70b-8192", groq_api_key=key, temperature=0.5, max_tokens=500)
    except Exception:
        return None
 
def get_openai_llm():
    """Stage 3 – OpenAI GPT-4o-mini: refined, polished final insights."""
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        return None
    try:
        return ChatOpenAI(model="gpt-4o-mini", openai_api_key=key, temperature=0.3, max_tokens=500)
    except Exception:
        return None
 

 
def match_career(query: str) -> Optional[pd.Series]:
    q = query.lower()
    for stop in ["how effective will","future scope of","what about","tell me about",
                 "career in","job in","will","be in","years","the"]:
        q = q.replace(stop, " ")
    q = re.sub(r"[^\w\s]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
 
    best_row, best_score = None, 0
    for _, row in DF.iterrows():
        kws   = str(row["keywords"]).lower().split(",")
        score = sum(len(k.strip()) for k in kws if k.strip() and k.strip() in q)
        if score > best_score:
            best_score, best_row = score, row
 
    if best_score == 0:
        words = [w for w in q.split() if len(w) > 2]
        for _, row in DF.iterrows():
            domain = str(row["career_domain"]).lower()
            hits   = sum(1 for w in words if w in domain)
            if hits > best_score:
                best_score, best_row = hits, row
 
    return best_row if best_score > 0 else None
 

 
class CareerState(TypedDict):
    query:          str
    matched_career: Optional[dict]
    groq_output:    str                            
    openai_output:  str                            
    error:          str
    pipeline_log:   Annotated[List[str], operator.add]
 

 

def node_match_career(state: CareerState) -> CareerState:
    """Node 1 — Keyword/fuzzy match against CSV dataset."""
    row = match_career(state["query"])
    if row is not None:
        return {
            **state,
            "matched_career": row.to_dict(),
            "error":          "",
            "pipeline_log":   [f"✅ Node 1 | Match → {row['career_domain']}"]
        }
    return {
        **state,
        "matched_career": None,
        "error":          "no_match",
        "pipeline_log":   ["❌ Node 1 | Match → No domain found"]
    }
 

@traceable(name="groq_analyze_node")     
def node_groq_analyze(state: CareerState) -> CareerState:
    """
    Node 2 — Groq LLaMA3-70B generates fast raw career intelligence.
    @traceable sends this execution to the LangSmith dashboard as a span.
    """
    if state.get("error"):
        return state
 
    row = state["matched_career"]
    llm = get_groq_llm()
 
    if llm is None:
        fallback = row.get("summary", "Add GROQ_API_KEY to enable Groq intelligence.")
        return {
            **state,
            "groq_output":  fallback,
            "pipeline_log": ["⚠️  Node 2 | Groq → No API key (CSV fallback used)"]
        }
 
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "You are a rapid career intelligence engine. "
            "Using the structured data provided, generate 2-3 sentences of raw, direct, "
            "data-backed career intelligence covering: market trajectory, salary outlook, "
            "and the single biggest risk factor. No fluff. Be specific and numbers-focused."
        )),
        HumanMessage(content=(
            f"Career Domain: {row['career_domain']}\n"
            f"Future Score: {row['future_score']}/100 | Demand: {row['demand_score']}/100\n"
            f"Annual Growth Rate: {row['growth_rate']}% | Risk Level: {row['risk_level']}\n"
            f"Avg Salary: ${row['avg_salary_usd']:,} | Automation Risk: {row['automation_risk']}%\n"
            f"Market Size: ${row['market_size_2024_b']}B (2024) → ${row['projected_market_2030_b']}B (2030)\n"
            f"Job Openings: {row['job_openings_k']}K+ | Top Skills: {row['required_skills']}\n\n"
            "Provide rapid raw career intelligence on this domain."
        ))
    ])
 
    try:
        resp = (prompt | llm).invoke({})
        return {
            **state,
            "groq_output":  resp.content,
            "pipeline_log": [f"🟠 Node 2 | Groq LLaMA3-70B → {len(resp.content)} chars generated"]
        }
    except Exception as e:
        fallback = row.get("summary", str(e))
        return {
            **state,
            "groq_output":  fallback,
            "pipeline_log": [f"⚠️  Node 2 | Groq error: {str(e)[:70]}"]
        }
 

@traceable(name="openai_refine_node")    
def node_openai_refine(state: CareerState) -> CareerState:
    """
    Node 3 — OpenAI GPT-4o-mini takes Groq's raw output and refines it
    into polished, empathetic, actionable career advice for the user.
    @traceable sends this execution to LangSmith as a separate span.
    """
    if state.get("error"):
        return state
 
    row         = state["matched_career"]
    groq_output = state.get("groq_output", "")
    llm         = get_openai_llm()
 
    if llm is None:
        return {
            **state,
            "openai_output": groq_output,
            "pipeline_log":  ["⚠️  Node 3 | OpenAI → No API key (Groq output used as final)"]
        }
 
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "You are the final refinement layer of CareerAI Oracle. "
            "You receive raw career intelligence and transform it into 3 polished, "
            "inspiring sentences tailored to the user's original question. "
            "Structure: (1) Core opportunity statement, (2) Key challenge to navigate, "
            "(3) One actionable next step. Be empathetic, realistic, and motivating. "
            "No bullet points. Flowing prose only."
        )),
        HumanMessage(content=(
            f"User's original query: \"{state['query']}\"\n"
            f"Career domain: {row['career_domain']}\n"
            f"Future Score: {row['future_score']}/100 | Avg Salary: ${row['avg_salary_usd']:,}\n\n"
            f"Raw intelligence from Groq LLaMA3-70B:\n{groq_output}\n\n"
            "Refine this into polished, user-ready career advice."
        ))
    ])
 
    try:
        resp = (prompt | llm).invoke({})
        return {
            **state,
            "openai_output": resp.content,
            "pipeline_log":  [f"🟢 Node 3 | OpenAI GPT-4o-mini → {len(resp.content)} chars (final output)"]
        }
    except Exception as e:
        return {
            **state,
            "openai_output": groq_output,
            "pipeline_log":  [f"⚠️  Node 3 | OpenAI error: {str(e)[:70]}"]
        }
 

 
def build_graph():
    g = StateGraph(CareerState)
    g.add_node("match",  node_match_career)
    g.add_node("groq",   node_groq_analyze)
    g.add_node("openai", node_openai_refine)
    g.set_entry_point("match")
    g.add_conditional_edges(
        "match",
        lambda s: "end" if s.get("error") == "no_match" else "groq",
        {"groq": "groq", "end": END}
    )
    g.add_edge("groq",   "openai")
    g.add_edge("openai", END)
    return g.compile()
 
@st.cache_resource
def get_graph():
    return build_graph()
 
def run_pipeline(query: str) -> CareerState:
    """Runs the full LangSmith→Groq→OpenAI pipeline via LangGraph."""
    return get_graph().invoke({
        "query":          query,
        "matched_career": None,
        "groq_output":    "",
        "openai_output":  "",
        "error":          "",
        "pipeline_log":   [],
    })
 

 
def score_color(s):
    if s >= 85: return "#34d399"
    if s >= 70: return "#fbbf24"
    return "#f87171"
 
def risk_html(risk):
    cls  = {"low":"risk-low","medium":"risk-medium","high":"risk-high"}.get(risk.lower(),"risk-medium")
    icon = {"low":"🟢","medium":"🟡","high":"🔴"}.get(risk.lower(),"⚪")
    return f'<span class="risk-badge {cls}">{icon} {risk}</span>'
 
def bar_html(pct, color, h=8):
    return (f'<div class="prog-bg" style="height:{h}px">'
            f'<div class="prog-fill" style="width:{pct}%;background:{color};height:{h}px"></div></div>')
 
def skills_html(s):
    pills = "".join(f'<span class="skill-pill">{x.strip()}</span>' for x in str(s).split(";"))
    return f'<div class="skills-wrap">{pills}</div>'
 
def score_ring_html(score):
    c   = score_color(score)
    deg = int(score * 3.6)
    return (f'<div class="score-wrapper">'
            f'<div class="score-ring" style="background:conic-gradient({c} 0deg {deg}deg,rgba(30,41,59,0.6) {deg}deg 360deg)">'
            f'<span class="snum" style="color:{c}">{score}</span>'
            f'<span class="slbl">/ 100</span></div></div>')
 

 
def render_career_card(row: dict, groq_out: str, openai_out: str, pipe_log: list):
    sc   = int(row["future_score"])
    dem  = int(row["demand_score"])
    grw  = int(row["growth_rate"])
    risk = str(row["risk_level"])
    sal  = int(row["avg_salary_usd"])
    auto = int(row["automation_risk"])
    m24  = float(row["market_size_2024_b"])
    m30  = float(row["projected_market_2030_b"])
    jobs = float(row["job_openings_k"])
    sc_c = score_color(sc);  dem_c = score_color(dem)
 

    st.markdown(f"""
    <div class="glass-card gc-green">
      <div class="sec-label">Career Domain Identified</div>
      <div class="career-name">{row['career_domain']}</div>
      {risk_html(risk)}
      <p class="career-summary">{row.get('summary','')}</p>
    </div>""", unsafe_allow_html=True)
 
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(score_ring_html(sc), unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-grid">
          <div class="metric-chip"><div class="mc-val" style="color:{dem_c}">{dem}</div><div class="mc-lbl">Demand</div></div>
          <div class="metric-chip"><div class="mc-val" style="color:#22d3ee">{grw}%</div><div class="mc-lbl">YoY Growth</div></div>
        </div>
        <div class="metric-grid">
          <div class="metric-chip"><div class="mc-val" style="color:#fbbf24">${sal//1000}K</div><div class="mc-lbl">Avg Salary</div></div>
          <div class="metric-chip"><div class="mc-val" style="color:#f87171">{auto}%</div><div class="mc-lbl">Auto-Risk</div></div>
        </div>""", unsafe_allow_html=True)
 
    with col2:
        st.markdown('<div class="glass-card" style="padding:1.1rem;margin-bottom:0">', unsafe_allow_html=True)
        for lbl, val, color, mx, sfx in [
            ("Future Score",  sc,  sc_c,     100, "/100"),
            ("Market Demand", dem, dem_c,    100, "/100"),
            ("Annual Growth", grw, "#22d3ee", 70, "%/yr"),
        ]:
            pct = int(val/mx*100)
            st.markdown(f"""
            <div class="prog-wrap">
              <div class="prog-hdr"><span>{lbl}</span><span style="color:{color}">{val}{sfx}</span></div>
              {bar_html(pct, color)}
            </div>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="display:flex;gap:0.85rem;margin-top:0.65rem;font-size:0.76rem;color:#64748b;flex-wrap:wrap">
          <span>📊 Market 2024: <b style="color:#94a3b8">${m24}B</b></span>
          <span>🚀 2030 Proj: <b style="color:#34d399">${m30}B</b></span>
          <span>💼 Openings: <b style="color:#a5b4fc">{jobs}K+</b></span>
        </div></div>""", unsafe_allow_html=True)
 
    
    st.markdown("**🛠 Required Skills**")
    st.markdown(skills_html(row["required_skills"]), unsafe_allow_html=True)
 
    
    if groq_out:
        st.markdown(f"""
        <div class="stage-box sb-groq">
          <div class="stage-label">🟠 Stage 2 — Groq LLaMA3-70B · Raw Career Intelligence</div>
          <div class="stage-content">{groq_out}</div>
        </div>""", unsafe_allow_html=True)
 

    if openai_out and openai_out != groq_out:
        st.markdown(f"""
        <div class="stage-box sb-oai">
          <div class="stage-label">🟢 Stage 3 — OpenAI GPT-4o-mini · Refined Final Insight</div>
          <div class="stage-content">{openai_out}</div>
        </div>""", unsafe_allow_html=True)
    elif openai_out:
        st.markdown(f"""
        <div class="stage-box sb-oai">
          <div class="stage-label">🟢 AI Career Insight (Final)</div>
          <div class="stage-content">{openai_out}</div>
        </div>""", unsafe_allow_html=True)
 

    if pipe_log:
        with st.expander("🔭 LangGraph Pipeline Trace Log  (LangSmith → Groq → OpenAI)"):
            for entry in pipe_log:
                st.markdown(
                    f'<div style="font-family:JetBrains Mono,monospace;font-size:0.76rem;'
                    f'color:#64748b;padding:2px 0">{entry}</div>',
                    unsafe_allow_html=True
                )
 

 
def render_compare(row_a: dict, row_b: dict):
    metrics = [
        ("Future Score",    "future_score",    100, "/100"),
        ("Market Demand",   "demand_score",    100, "/100"),
        ("Growth Rate",     "growth_rate",      70, "%/yr"),
        ("Automation Risk", "automation_risk", 100, "%"),
    ]
    winner = row_a["career_domain"] if int(row_a["future_score"]) >= int(row_b["future_score"]) else row_b["career_domain"]
 
    st.markdown(f"""
    <div class="glass-card gc-purple">
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.9rem;margin-bottom:0.9rem">
        <div style="text-align:center;padding:0.65rem;border-radius:10px;
                    background:rgba(99,102,241,0.14);border:1px solid rgba(99,102,241,0.35);
                    color:#818cf8;font-weight:700">{row_a['career_domain']}</div>
        <div style="text-align:center;padding:0.65rem;border-radius:10px;
                    background:rgba(52,211,153,0.12);border:1px solid rgba(52,211,153,0.3);
                    color:#34d399;font-weight:700">{row_b['career_domain']}</div>
      </div>""", unsafe_allow_html=True)
 
    for lbl, key, mx, sfx in metrics:
        va, vb = int(row_a[key]), int(row_b[key])
        pa, pb = int(va/mx*100), int(vb/mx*100)
        ca = "#f87171" if key=="automation_risk" and va>vb else "#818cf8"
        cb = "#f87171" if key=="automation_risk" and vb>va else "#34d399"
        st.markdown(f"""
        <div class="cmp-row">
          <div class="cmp-mname">{lbl}</div>
          <div class="cmp-bwrap">
            <span class="cmp-val" style="color:{ca}">{va}{sfx}</span>
            {bar_html(pa, ca, 6)}
          </div>
          <div class="cmp-bwrap">
            <span class="cmp-val" style="color:{cb}">{vb}{sfx}</span>
            {bar_html(pb, cb, 6)}
          </div>
        </div>""", unsafe_allow_html=True)
 
    st.markdown(f"""
    <div style="text-align:center;padding:0.9rem 0">
      <div style="font-size:0.7rem;color:#64748b;margin-bottom:0.45rem">RECOMMENDED CHOICE</div>
      <span class="winner-badge">🏆 {winner}</span>
    </div></div>""", unsafe_allow_html=True)
 

 
groq_ok   = bool(os.getenv("GROQ_API_KEY",     ""))
openai_ok = bool(os.getenv("OPENAI_API_KEY",   ""))
ls_ok     = bool(os.getenv("LANGSMITH_API_KEY",""))
 
with st.sidebar:
    st.markdown("""
    <div style="padding:0.7rem 0 0.4rem">
      <div style="font-size:1.4rem;font-weight:900;
                  background:linear-gradient(135deg,#818cf8,#34d399);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent">
        🔮 CareerAI Oracle
      </div>
      <div style="font-size:0.7rem;color:#475569;margin-top:2px">AI-Powered Career Forecasting</div>
    </div>
    <hr style="margin:0.4rem 0 0.65rem">
    """, unsafe_allow_html=True)
 
    st.markdown('<div style="font-size:0.69rem;font-weight:700;letter-spacing:0.1em;color:#475569;text-transform:uppercase;margin-bottom:0.45rem">AI Pipeline Status</div>', unsafe_allow_html=True)
 
    def srow(icon, label, ok):
        c   = "#34d399" if ok else "#f87171"
        txt = "Connected" if ok else "No Key"
        return (f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'padding:5px 8px;border-radius:8px;margin-bottom:3px;'
                f'background:rgba(30,41,59,0.4);border:1px solid rgba(99,102,241,0.1)">'
                f'<span style="font-size:0.76rem">{icon} <b style="color:#94a3b8">{label}</b></span>'
                f'<span style="font-size:0.7rem;color:{c}">{"●" if ok else "○"} {txt}</span></div>')
 
    st.markdown(
        srow("🔵","LangSmith (Tracing)", ls_ok) +
        srow("🟠","Groq LLaMA3-70B",     groq_ok) +
        srow("🟢","OpenAI GPT-4o-mini",  openai_ok) +
        f'<div style="font-size:0.66rem;color:#334155;padding:4px 8px">⚡ LangGraph · 3-node pipeline · {len(DF)} careers</div>',
        unsafe_allow_html=True
    )
 
    st.markdown('<hr style="margin:0.65rem 0">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.69rem;font-weight:700;letter-spacing:0.1em;color:#475569;text-transform:uppercase;margin-bottom:0.45rem">Top Careers by Score</div>', unsafe_allow_html=True)
 
    for _, r in DF.nlargest(9,"future_score")[["career_domain","future_score"]].iterrows():
        c = score_color(int(r["future_score"]))
        st.markdown(f'<div class="trend-item"><span class="trend-name">{r["career_domain"]}</span>'
                    f'<span class="trend-badge" style="color:{c};border:1px solid {c}40;background:{c}18">{int(r["future_score"])}</span></div>',
                    unsafe_allow_html=True)
 
    st.markdown("""<hr style="margin:0.65rem 0">
    <div style="font-size:0.68rem;color:#334155;line-height:1.65">
      💡 <b style="color:#475569">Try asking:</b><br>
      · "Future of AI jobs"<br>
      · "Is cybersecurity worth it?"<br>
      · "B.Tech degree in 5 years"<br>
      · "Scope of prompt engineering"
    </div>""", unsafe_allow_html=True)
 


 
st.markdown("""
<div class="hero-title">
  <h1>🔮 CareerAI Oracle</h1>
  <div class="hero-sub">AI-powered career forecasting · LangGraph intelligence · Data-backed predictions</div>
</div>""", unsafe_allow_html=True)
 
st.markdown("""
<div class="pipe-banner">
  <span class="pipe-step ps-ls">🔵 LangSmith</span>
  <span class="pipe-arrow">traces</span>
  <span class="pipe-step ps-gr">🟠 Groq LLaMA3-70B</span>
  <span class="pipe-arrow">→</span>
  <span class="pipe-step ps-oai">🟢 OpenAI GPT-4o-mini</span>
  <span class="pipe-arrow">→</span>
  <span style="font-size:0.78rem;color:#475569">Final Insight</span>
</div>""", unsafe_allow_html=True)
 
tab1, tab2, tab3 = st.tabs(["🤖  Career Prediction", "⚔️  Compare Careers", "📊  Career Explorer"])
 

with tab1:
    cq, cb = st.columns([4, 1])
    with cq:
        query = st.text_input("", placeholder="e.g. How effective will a B.E degree be in 5 years? · Future scope of AI jobs · Is cybersecurity worth it?",
                              label_visibility="collapsed", key="main_query")
    with cb:
        predict_btn = st.button("Predict 🔮", key="predict_btn")
 
    examples = ["Future of AI jobs","Is cybersecurity good?","B.Tech in 5 years",
                "Data science outlook","Cloud computing scope","Prompt engineering career"]
    ecols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        if ecols[i].button(ex, key=f"ex_{i}"):
            query = ex; predict_btn = True
 
    if predict_btn and query.strip():
        with st.spinner("🔵 LangSmith tracing  →  🟠 Groq analyzing  →  🟢 OpenAI refining…"):
            result = run_pipeline(query)
 
        if result.get("error") == "no_match" or not result.get("matched_career"):
            st.markdown("""
            <div class="glass-card" style="text-align:center;padding:2.5rem">
              <div style="font-size:3rem">🔍</div>
              <div style="color:#94a3b8;margin-top:0.75rem;font-size:1rem">No matching career domain found.</div>
              <div style="color:#475569;font-size:0.8rem;margin-top:0.4rem">
                Try: AI, cybersecurity, data science, cloud, blockchain, MBA, B.Tech…
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            render_career_card(
                result["matched_career"],
                result.get("groq_output",""),
                result.get("openai_output",""),
                result.get("pipeline_log",[])
            )
 
    elif not query.strip():
        st.markdown("""
        <div class="glass-card" style="text-align:center;padding:3rem">
          <div style="font-size:4rem">🔮</div>
          <div style="color:#94a3b8;font-size:1.05rem;margin-top:0.75rem">Ask anything about career futures</div>
          <div style="color:#475569;font-size:0.8rem;margin-top:0.4rem">
            🔵 LangSmith traces every call &nbsp;→&nbsp; 🟠 Groq generates raw intelligence
            &nbsp;→&nbsp; 🟢 OpenAI refines the final insight
          </div>
        </div>""", unsafe_allow_html=True)
 

with tab2:
    st.markdown('<div class="glass-card" style="padding:1.15rem">', unsafe_allow_html=True)
    ca_col, cb_col = st.columns(2)
    career_names = sorted(DF["career_domain"].tolist())
    with ca_col:
        st.markdown('<p style="color:#818cf8;font-size:0.76rem;font-weight:700;margin-bottom:4px;text-transform:uppercase;letter-spacing:0.08em">Career A</p>', unsafe_allow_html=True)
        sel_a = st.selectbox("", career_names, index=0, key="sela", label_visibility="collapsed")
    with cb_col:
        st.markdown('<p style="color:#34d399;font-size:0.76rem;font-weight:700;margin-bottom:4px;text-transform:uppercase;letter-spacing:0.08em">Career B</p>', unsafe_allow_html=True)
        sel_b = st.selectbox("", career_names, index=min(1,len(career_names)-1), key="selb", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
 
    if st.button("Compare Careers ⚔️", key="cmp_btn"):
        if sel_a == sel_b:
            st.warning("⚠️ Select two different career domains.")
        else:
            ra = DF[DF["career_domain"]==sel_a].iloc[0].to_dict()
            rb = DF[DF["career_domain"]==sel_b].iloc[0].to_dict()
            render_compare(ra, rb)
            st.markdown("---")
            cc1, cc2 = st.columns(2)
            with cc1:
                render_career_card(ra, ra.get("summary",""), "", [])
            with cc2:
                render_career_card(rb, rb.get("summary",""), "", [])
 

with tab3:
    f1, f2, f3 = st.columns(3)
    with f1: risk_f = st.selectbox("Risk Level", ["All","Low","Medium","High"], key="rf")
    with f2: min_sc = st.slider("Min Future Score", 0, 100, 60, key="sff")
    with f3: sort_f = st.selectbox("Sort By", ["future_score","demand_score","growth_rate","avg_salary_usd"], key="sortf")
 
    filtered = DF.copy()
    if risk_f != "All":
        filtered = filtered[filtered["risk_level"].str.lower() == risk_f.lower()]
    filtered = filtered[filtered["future_score"] >= min_sc].sort_values(sort_f, ascending=False)
 
    st.markdown(f'<div style="color:#64748b;font-size:0.79rem;margin-bottom:0.7rem">Showing <b style="color:#e2e8f0">{len(filtered)}</b> career domains</div>', unsafe_allow_html=True)
    for _, row in filtered.iterrows():
        icon = "🟢" if row["risk_level"]=="Low" else "🟡" if row["risk_level"]=="Medium" else "🔴"
        with st.expander(f"{icon}  {row['career_domain']}  ·  Score {int(row['future_score'])}/100  ·  {row['growth_rate']}%/yr growth"):
            render_career_card(row.to_dict(), row.get("summary",""), "", [])
 

st.markdown("""
<hr>
<div style="text-align:center;padding:0.8rem;color:#1e293b;font-size:0.73rem">
  🔮 <b style="color:#334155">CareerAI Oracle</b> &nbsp;·&nbsp;
  🔵 LangSmith &nbsp;→&nbsp; 🟠 Groq LLaMA3-70B &nbsp;→&nbsp; 🟢 OpenAI GPT-4o-mini
  &nbsp;·&nbsp; Powered by LangGraph
</div>""", unsafe_allow_html=True)
