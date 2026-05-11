import os
import re
import operator
import traceback
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
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()

USD_TO_INR = 84

_CAREER_SALARY_MAP: dict[str, float] = {
    "ai":              18.5, "machine learning": 17.2, "ml":          17.2,
    "data science":    15.8, "data scientist":   15.8, "data":        13.0,
    "cloud":           14.5, "devops":           13.8, "sre":         14.0,
    "cybersecurity":   14.0, "security":         13.5, "blockchain":  15.0,
    "software":        12.5, "developer":        11.0, "engineer":    12.0,
    "web":             10.5, "frontend":         10.0, "backend":     12.0,
    "fullstack":       12.5, "mobile":           11.5, "android":     11.0,
    "ios":             11.5, "flutter":          10.5,
    "product":         16.0, "product manager":  17.5, "scrum":       11.0,
    "ux":               9.5, "ui":                9.0, "design":       9.0,
    "embedded":        11.0, "iot":              12.0, "robotics":    13.5,
    "quantum":         19.0, "ar":               13.0, "vr":          13.0,
    "game":             9.5, "animation":         8.0,
    "finance":         14.0, "fintech":          16.0, "accounting":   8.5,
    "hr":               7.5, "marketing":         8.0, "sales":        8.5,
    "content":          6.5, "seo":               6.0, "social media":  6.0,
    "supply chain":     9.0, "logistics":         8.0, "operations":    9.5,
    "biotech":         12.0, "pharma":           10.5, "healthcare":    9.0,
    "legal":           10.0, "law":              10.0,
    "mba":             14.0, "consulting":       15.5, "analyst":      10.5,
    "btech":           10.0, "bsc":               7.5, "research":     11.0,
    "prompt":          13.0, "nlp":              16.0, "llm":          17.0,
    "database":        10.5, "sql":              10.0, "dba":          11.0,
    "testing":          8.5, "qa":                8.5, "automation":   11.0,
    "networking":      10.0, "cisco":            11.0, "network":      10.5,
    "erp":             10.0, "sap":              13.0, "salesforce":   14.0,
}

def _salary_lakhs_from_domain(domain: str) -> float:
    """Return a realistic ₹ Lakhs salary for a career domain using keyword match."""
    d = domain.lower()
    for keyword, lakhs in _CAREER_SALARY_MAP.items():
        if keyword in d:
            return lakhs
    return 9.5

def _normalise_salary_to_lakhs(raw: float, domain: str = "") -> float:
    """
    Convert raw salary to Lakhs. Rules:
      raw <= 0 / 60000  -> domain keyword map (missing/default)
      raw <= 300        -> already in Lakhs  (e.g. 18 -> 18L)
      raw <= 100_000    -> USD amount        (e.g. 80000 -> 67.2L)
      raw > 100_000     -> already INR       (e.g. 2000000 -> 20L, 3500000 -> 35L)
    Real CSV values are ALWAYS trusted. No upper cap.
    """
    if raw <= 0 or raw == 60000:
        return _salary_lakhs_from_domain(domain)
    if raw <= 300:
        lakhs = float(raw)
    elif raw <= 100_000:
        lakhs = (raw * USD_TO_INR) / 100_000
    else:
        lakhs = raw / 100_000
    if lakhs <= 0:
        return _salary_lakhs_from_domain(domain)
    return round(lakhs, 1)

def format_inr(raw: float, domain: str = "") -> str:
    lakhs = _normalise_salary_to_lakhs(raw, domain)
    return f"₹{lakhs:.1f}L"

def format_inr_chip(raw: float, domain: str = "") -> str:
    """Always return in Lakhs (L) — realistic Indian range 4–20L."""
    lakhs = _normalise_salary_to_lakhs(raw, domain)
    return f"₹{lakhs:.1f}L"


def init_langsmith() -> Optional[LangSmithClient]:
    api_key = os.getenv("LANGSMITH_API_KEY", "")
    project = os.getenv("LANGSMITH_PROJECT", "CareerAI")
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
    page_title="CareerAI",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

IS_DARK = st.session_state.theme == "dark"


DARK_VARS = """
    --bg-base:
    --bg-card:       rgba(15,23,42,0.75);
    --bg-card-solid:
    --bg-input:      rgba(15,23,42,0.8);
    --bg-chip:       rgba(30,41,59,0.8);
    --bg-sidebar:    rgba(15,23,42,0.96);
    --border-main:   rgba(99,102,241,0.2);
    --border-input:  rgba(99,102,241,0.3);
    --border-chip:   rgba(99,102,241,0.18);
    --text-primary:
    --text-secondary:
    --text-muted:
    --text-faint:
    --text-faintest:
    --divider:       rgba(30,41,59,0.8);
    --trend-bg:      rgba(30,41,59,0.5);
    --trend-border:  rgba(99,102,241,0.13);
    --tab-list:      rgba(15,23,42,0.6);
    --tab-text:
    --tab-sel:       rgba(99,102,241,0.25);
    --tab-sel-text:
    --prog-bg:       rgba(30,41,59,0.8);
    --cmp-border:    rgba(30,41,59,0.6);
    --scrollbar-track:
    --scrollbar-thumb:
    --bg-radial1:    rgba(99,102,241,0.18);
    --bg-radial2:    rgba(16,185,129,0.14);
    --bg-radial3:    rgba(118,169,250,0.10);
    --bg-radial4:    rgba(6,182,212,0.07);
    --hero-sub:
    --pipe-bg:       rgba(15,23,42,0.75);
    --pipe-border:   rgba(99,102,241,0.22);
    --pipe-arrow:
"""

LIGHT_VARS = """
    --bg-base:
    --bg-card:       rgba(255,255,255,0.88);
    --bg-card-solid:
    --bg-input:      rgba(255,255,255,0.9);
    --bg-chip:       rgba(224,237,255,0.9);
    --bg-sidebar:    rgba(225,240,255,0.97);
    --border-main:   rgba(99,102,241,0.25);
    --border-input:  rgba(99,102,241,0.4);
    --border-chip:   rgba(99,102,241,0.22);
    --text-primary:
    --text-secondary:
    --text-muted:
    --text-faint:
    --text-faintest:
    --divider:       rgba(99,102,241,0.15);
    --trend-bg:      rgba(200,220,255,0.5);
    --trend-border:  rgba(99,102,241,0.2);
    --tab-list:      rgba(200,220,255,0.5);
    --tab-text:
    --tab-sel:       rgba(99,102,241,0.18);
    --tab-sel-text:
    --prog-bg:       rgba(200,220,255,0.5);
    --cmp-border:    rgba(99,102,241,0.15);
    --scrollbar-track:
    --scrollbar-thumb:
    --bg-radial1:    rgba(99,102,241,0.10);
    --bg-radial2:    rgba(16,185,129,0.08);
    --bg-radial3:    rgba(118,169,250,0.06);
    --bg-radial4:    rgba(6,182,212,0.04);
    --hero-sub:
    --pipe-bg:       rgba(255,255,255,0.75);
    --pipe-border:   rgba(99,102,241,0.3);
    --pipe-arrow:
"""

THEME_VARS = DARK_VARS if IS_DARK else LIGHT_VARS

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

:root {{ {THEME_VARS} }}

*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
html,body,.stApp{{
    background:var(--bg-base)!important;
    font-family:'Inter',sans-serif!important;
    color:var(--text-primary)!important;
}}

.stApp::before{{
    content:'';position:fixed;inset:0;z-index:-1;
    background:
        radial-gradient(ellipse 80% 60% at 10% 10%, var(--bg-radial1) 0%,transparent 60%),
        radial-gradient(ellipse 60% 50% at 90% 85%, var(--bg-radial2) 0%,transparent 55%),
        radial-gradient(ellipse 50% 40% at 70% 20%, var(--bg-radial3) 0%,transparent 50%),
        radial-gradient(ellipse 70% 50% at 50% 50%, var(--bg-radial4) 0%,transparent 70%),
        var(--bg-base);
    animation:bgP 12s ease-in-out infinite alternate;
}}
@keyframes bgP{{0%{{opacity:1}}100%{{opacity:0.85}}}}

::-webkit-scrollbar{{width:6px}}
::-webkit-scrollbar-track{{background:var(--scrollbar-track)}}
::-webkit-scrollbar-thumb{{background:var(--scrollbar-thumb);border-radius:3px}}

header[data-testid="stHeader"]{{display:none!important;height:0!important}}
.block-container{{padding-top:0.5rem!important;margin-top:0!important}}
[data-testid="stAppViewContainer"]{{padding-top:0!important}}
[data-testid="stMain"]{{padding-top:0!important}}
.stDeployButton,
section[data-testid="stSidebar"]{{
    background:var(--bg-sidebar)!important;
    border-right:1px solid var(--border-main)!important;
    backdrop-filter:blur(20px);
}}

.theme-btn{{
    display:inline-flex;align-items:center;gap:7px;
    padding:7px 15px;border-radius:99px;cursor:pointer;
    font-family:'Inter',sans-serif;font-size:0.73rem;font-weight:700;
    letter-spacing:0.04em;border:1.5px solid var(--border-input);
    background:var(--bg-card);color:var(--text-primary);
    box-shadow:0 2px 16px rgba(99,102,241,0.18);
    transition:all 0.22s;backdrop-filter:blur(12px);
}}
.theme-btn:hover{{
    background:rgba(99,102,241,0.18);
    border-color:rgba(99,102,241,0.6);
    box-shadow:0 4px 24px rgba(99,102,241,0.28);
}}

.pipe-banner{{
    display:flex;align-items:center;justify-content:center;
    gap:0.35rem;flex-wrap:wrap;
    background:var(--pipe-bg);
    border:1px solid var(--pipe-border);
    border-radius:14px;padding:0.65rem 1.3rem;
    margin:0.6rem auto 1.5rem;max-width:820px;font-size:0.76rem;
    box-shadow:0 0 30px rgba(99,102,241,0.08);
}}
.pipe-step{{
    display:flex;align-items:center;gap:5px;
    padding:5px 14px;border-radius:99px;
    font-family:'JetBrains Mono',monospace;font-weight:600;
    font-size:0.75rem;
}}
.ps-ls {{background:rgba(99,102,241,0.18);border:1px solid rgba(99,102,241,0.45);color:#a5b4fc}}
.ps-gr {{background:rgba(251,146,60,0.15);border:1px solid rgba(251,146,60,0.45);color:#fb923c}}
.ps-oai{{background:rgba(52,211,153,0.13);border:1px solid rgba(52,211,153,0.45);color:#34d399}}
.ps-nv {{background:rgba(118,169,250,0.14);border:1px solid rgba(118,169,250,0.45);color:#76a9fa}}
.pipe-arrow{{color:var(--pipe-arrow);font-size:0.95rem;font-weight:700}}

.hero-title{{text-align:center;padding:2rem 1rem 0.5rem;animation:fadeInDown 0.7s ease-out}}
.hero-title h1{{
    font-size:clamp(2rem,5vw,3.2rem);font-weight:900;
    background:linear-gradient(135deg,#818cf8 0%,#34d399 40%,#76a9fa 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
    letter-spacing:-1px;line-height:1.1;
}}
.hero-sub{{font-size:0.9rem;color:var(--hero-sub);margin-top:0.4rem}}

.glass-card{{
    background:var(--bg-card);border:1px solid var(--border-main);
    border-radius:16px;padding:1.5rem;backdrop-filter:blur(16px);
    box-shadow:0 4px 40px rgba(0,0,0,0.15);margin-bottom:1.1rem;
    animation:fadeInUp 0.55s ease-out;
}}
.gc-green {{border-color:rgba(52,211,153,0.3);box-shadow:0 0 30px rgba(52,211,153,0.07),0 4px 40px rgba(0,0,0,0.12)}}
.gc-purple{{border-color:rgba(139,92,246,0.3);box-shadow:0 0 30px rgba(139,92,246,0.07),0 4px 40px rgba(0,0,0,0.12)}}

.stage-box{{border-radius:12px;padding:1rem 1.3rem;margin-top:0.65rem;position:relative;overflow:hidden}}
.sb-groq{{background:linear-gradient(135deg,rgba(251,146,60,0.08),rgba(245,158,11,0.04));border:1px solid rgba(251,146,60,0.25)}}
.sb-oai {{background:linear-gradient(135deg,rgba(52,211,153,0.08),rgba(34,211,238,0.04));border:1px solid rgba(52,211,153,0.25)}}
.sb-nv  {{background:linear-gradient(135deg,rgba(118,169,250,0.10),rgba(99,102,241,0.05));border:1px solid rgba(118,169,250,0.30)}}
.sb-err {{background:linear-gradient(135deg,rgba(239,68,68,0.08),rgba(220,38,38,0.04));border:1px solid rgba(239,68,68,0.25)}}
.stage-label{{font-size:0.66rem;font-weight:700;letter-spacing:0.13em;text-transform:uppercase;margin-bottom:0.45rem;display:flex;align-items:center;gap:6px}}
.sb-groq .stage-label{{color:#fb923c}}
.sb-oai  .stage-label{{color:#34d399}}
.sb-nv   .stage-label{{color:#76a9fa}}
.sb-err  .stage-label{{color:#f87171}}
.stage-content{{font-size:0.85rem;color:var(--text-secondary);line-height:1.72}}

.roadmap-container{{padding:0.25rem 0}}
.roadmap-step{{display:flex;align-items:flex-start;gap:0.85rem;padding:0.75rem 0;border-bottom:1px solid rgba(118,169,250,0.1)}}
.roadmap-step:last-child{{border-bottom:none}}
.roadmap-icon{{width:34px;height:34px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:0.85rem;font-weight:800;flex-shrink:0;font-family:'JetBrains Mono',monospace}}
.rs-1{{background:rgba(99,102,241,0.2);border:1.5px solid rgba(99,102,241,0.5);color:#818cf8}}
.rs-2{{background:rgba(52,211,153,0.2);border:1.5px solid rgba(52,211,153,0.5);color:#34d399}}
.rs-3{{background:rgba(251,146,60,0.2);border:1.5px solid rgba(251,146,60,0.5);color:#fb923c}}
.rs-4{{background:rgba(118,169,250,0.2);border:1.5px solid rgba(118,169,250,0.5);color:#76a9fa}}
.rs-5{{background:rgba(251,191,36,0.2);border:1.5px solid rgba(251,191,36,0.5);color:#fbbf24}}
.roadmap-phase{{font-size:0.68rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:var(--text-muted);margin-bottom:3px}}
.roadmap-content{{font-size:0.83rem;color:var(--text-secondary);line-height:1.6}}

.career-name{{font-size:1.6rem;font-weight:800;color:var(--text-primary);line-height:1.2;margin-bottom:0.28rem}}
.sec-label{{font-size:0.66rem;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;color:#6366f1;margin-bottom:0.28rem}}
.career-summary{{font-size:0.86rem;color:var(--text-muted);line-height:1.65;margin-top:0.42rem}}

.risk-badge{{display:inline-flex;align-items:center;gap:6px;border-radius:99px;padding:4px 13px;font-size:0.73rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em}}
.risk-low   {{background:rgba(16,185,129,0.12);border:1px solid rgba(16,185,129,0.4);color:#34d399}}
.risk-medium{{background:rgba(251,191,36,0.12);border:1px solid rgba(251,191,36,0.4);color:#fbbf24}}
.risk-high  {{background:rgba(239,68,68,0.12); border:1px solid rgba(239,68,68,0.4); color:#f87171}}

.score-wrapper{{display:flex;justify-content:center;padding:0.55rem 0}}
.score-ring{{width:110px;height:110px;border-radius:50%;display:flex;flex-direction:column;align-items:center;justify-content:center;font-weight:900;position:relative}}
.score-ring::before{{content:'';position:absolute;inset:5px;border-radius:50%;background:var(--bg-card-solid)}}
.snum{{font-size:1.8rem;position:relative;z-index:1}}
.slbl{{font-size:0.59rem;letter-spacing:0.1em;text-transform:uppercase;position:relative;z-index:1;color:var(--text-muted)}}

.metric-grid{{display:flex;flex-wrap:wrap;gap:0.48rem;margin:0.7rem 0}}
.metric-chip{{background:var(--bg-chip);border:1px solid var(--border-chip);border-radius:10px;padding:0.48rem 0.82rem;flex:1;min-width:82px;text-align:center}}
.mc-val{{font-size:1.22rem;font-weight:800;line-height:1}}
.mc-lbl{{font-size:0.59rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.08em;margin-top:3px}}

.prog-wrap{{margin:0.38rem 0}}
.prog-hdr{{display:flex;justify-content:space-between;font-size:0.73rem;margin-bottom:4px;color:var(--text-secondary)}}
.prog-bg{{height:8px;border-radius:99px;background:var(--prog-bg);overflow:hidden}}
.prog-fill{{height:100%;border-radius:99px}}

.skills-wrap{{display:flex;flex-wrap:wrap;gap:0.36rem;margin-top:0.55rem}}
.skill-pill{{background:rgba(99,102,241,0.12);border:1px solid rgba(99,102,241,0.25);border-radius:99px;padding:3px 10px;font-size:0.7rem;color:#a5b4fc;font-family:'JetBrains Mono',monospace}}

.cmp-row{{display:grid;grid-template-columns:1fr 2fr 2fr;gap:0.42rem;align-items:center;padding:0.48rem 0;border-bottom:1px solid var(--cmp-border)}}
.cmp-mname{{font-size:0.71rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.06em}}
.cmp-bwrap{{display:flex;align-items:center;gap:7px;font-size:0.77rem}}
.cmp-val{{font-weight:700;min-width:38px}}
.winner-badge{{display:inline-flex;align-items:center;gap:6px;background:linear-gradient(135deg,rgba(251,191,36,0.2),rgba(245,158,11,0.1));border:1px solid rgba(251,191,36,0.5);border-radius:99px;padding:6px 18px;color:#fbbf24;font-weight:700;font-size:0.82rem;animation:glow 2s ease-in-out infinite alternate}}
@keyframes glow{{0%{{box-shadow:0 0 8px rgba(251,191,36,0.3)}}100%{{box-shadow:0 0 20px rgba(251,191,36,0.5)}}}}

.trend-item{{background:var(--trend-bg);border:1px solid var(--trend-border);border-radius:10px;padding:0.52rem 0.78rem;margin-bottom:0.4rem;display:flex;justify-content:space-between;align-items:center}}
.trend-name{{font-size:0.78rem;color:var(--text-primary);font-weight:600}}
.trend-badge{{border-radius:99px;padding:2px 9px;font-size:0.67rem;font-weight:700}}

.stTextInput>div>div>input,.stTextArea>div>div>textarea{{
    background:var(--bg-input)!important;
    border:1.5px solid var(--border-input)!important;
    border-radius:10px!important;
    color:var(--text-primary)!important;
    font-family:'Inter',sans-serif!important;font-size:0.91rem!important
}}
.stTextInput>div>div>input:focus{{border-color:rgba(99,102,241,0.7)!important;box-shadow:0 0 0 3px rgba(99,102,241,0.1)!important}}
.stSelectbox>div>div{{background:var(--bg-input)!important;border:1.5px solid var(--border-input)!important;border-radius:10px!important}}
.stButton>button{{background:linear-gradient(135deg,#6366f1,#4f46e5)!important;color:white!important;border:none!important;border-radius:10px!important;padding:0.56rem 1.6rem!important;font-weight:700!important;font-size:0.86rem!important;width:100%!important;box-shadow:0 4px 20px rgba(99,102,241,0.35)!important;transition:all 0.2s!important}}
.stButton>button:hover{{transform:translateY(-2px)!important;box-shadow:0 8px 30px rgba(99,102,241,0.5)!important}}

.stTabs [data-baseweb="tab-list"]{{background:var(--tab-list)!important;border-radius:12px!important;padding:4px!important;gap:4px!important;border:1px solid var(--border-main)!important}}
.stTabs [data-baseweb="tab"]{{background:transparent!important;color:var(--tab-text)!important;border-radius:8px!important;font-weight:600!important;font-size:0.82rem!important;padding:0.44rem 1rem!important}}
.stTabs [aria-selected="true"]{{background:var(--tab-sel)!important;color:var(--tab-sel-text)!important}}
.stTabs [data-baseweb="tab-panel"]{{padding-top:1rem!important}}

hr{{border-color:var(--divider)!important}}
@keyframes fadeInDown{{from{{opacity:0;transform:translateY(-18px)}}to{{opacity:1;transform:translateY(0)}}}}
@keyframes fadeInUp  {{from{{opacity:0;transform:translateY(18px)}} to{{opacity:1;transform:translateY(0)}}}}
</style>
""", unsafe_allow_html=True)


toggle_col, _ = st.columns([1, 8])
with toggle_col:
    btn_icon  = "☀️" if IS_DARK else "🌙"
    btn_label = "Light Blue" if IS_DARK else "Dark Blue"
    if st.button(f"{btn_icon} {btn_label}", key="theme_toggle", on_click=toggle_theme):
        pass


_COL_ALIASES: dict[str, list[str]] = {
    "avg_salary_usd": [
        "avg_salary_usd", "average_salary_usd", "salary_usd",
        "avg_salary_inr", "average_salary_inr", "salary_inr",
        "avg_salary", "average_salary", "salary", "mean_salary_usd",
        "median_salary_usd", "annual_salary_usd", "annual_salary",
        "mean_salary_inr", "median_salary_inr", "annual_salary_inr",
    ],
    "future_score": [
        "future_score", "futurescore", "future score",
        "career_score", "score",
    ],
    "demand_score": [
        "demand_score", "demandscore", "demand score", "demand",
    ],
    "growth_rate": [
        "growth_rate", "growthrate", "growth rate", "growth",
        "annual_growth", "yoy_growth",
    ],
    "automation_risk": [
        "automation_risk", "automationrisk", "automation risk",
        "auto_risk", "ai_risk",
    ],
    "market_size_2024_b": [
        "market_size_2024_b", "market_size_2024", "market2024",
        "market_size_b", "market_size",
    ],
    "projected_market_2030_b": [
        "projected_market_2030_b", "projected_market_2030",
        "market2030", "projected_market",
    ],
    "job_openings_k": [
        "job_openings_k", "job_openings", "openings_k", "openings",
        "jobs_k", "jobs",
    ],
    "career_domain": [
        "career_domain", "careerdomain", "career domain",
        "domain", "career", "job_title", "title",
    ],
    "risk_level": [
        "risk_level", "risklevel", "risk level", "risk",
    ],
    "keywords": [
        "keywords", "keyword", "tags",
    ],
    "required_skills": [
        "required_skills", "requiredskills", "required skills",
        "skills", "key_skills",
    ],
    "summary": [
        "summary", "description", "overview", "about",
    ],
}

_DEFAULTS: dict[str, object] = {
    "avg_salary_usd":          60000,
    "future_score":            50,
    "demand_score":            50,
    "growth_rate":             10,
    "automation_risk":         30,
    "market_size_2024_b":      10,
    "projected_market_2030_b": 15,
    "job_openings_k":          50,
    "risk_level":              "Medium",
    "keywords":                "",
    "required_skills":         "",
    "summary":                 "",
}


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    alias_lookup: dict[str, str] = {}
    for canonical, aliases in _COL_ALIASES.items():
        for alias in aliases:
            alias_lookup[alias.lower().strip()] = canonical

    rename_map: dict[str, str] = {}
    for col in df.columns:
        canonical = alias_lookup.get(col.lower().strip())
        if canonical and canonical not in df.columns:
            rename_map[col] = canonical

    if rename_map:
        df = df.rename(columns=rename_map)

    for canonical, default in _DEFAULTS.items():
        if canonical not in df.columns:
            df[canonical] = default

    return df


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv("data.csv")
    df.columns = df.columns.str.strip()
    df = _normalise_columns(df)
    for col in ["avg_salary_usd", "future_score", "demand_score",
                "growth_rate", "automation_risk",
                "market_size_2024_b", "projected_market_2030_b", "job_openings_k"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(_DEFAULTS[col])
    return df


DF = load_data()


def get_groq_llm():
    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        return None
    try:
        return ChatGroq(model="llama3-70b-8192", groq_api_key=key, temperature=0.5, max_tokens=500)
    except Exception:
        return None

def get_openai_llm():
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        return None
    try:
        return ChatOpenAI(model="gpt-4o-mini", openai_api_key=key, temperature=0.3, max_tokens=500)
    except Exception:
        return None

def get_nvidia_llm():
    key = os.getenv("NVIDIA_API_KEY", "")
    if not key:
        return None
    try:
        return ChatNVIDIA(
            model="meta/llama-3.3-70b-instruct",
            api_key=key,
            base_url="https://integrate.api.nvidia.com/v1",
            temperature=0.4,
            max_tokens=700,
        )
    except Exception:
        return None

def _extract_content(resp) -> str:
    content = resp.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(p for p in parts if p).strip()
    return str(content)


def match_career(query: str) -> Optional[pd.Series]:
    q = query.lower()
    for stop in ["how effective will", "future scope of", "what about", "tell me about",
                 "career in", "job in", "will", "be in", "years", "the"]:
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
    nvidia_output:  str
    nvidia_error:   str
    error:          str
    pipeline_log:   Annotated[List[str], operator.add]


def node_match_career(state: CareerState) -> CareerState:
    row = match_career(state["query"])
    if row is not None:
        return {**state,
                "matched_career": row.to_dict(),
                "error":          "",
                "nvidia_error":   "",
                "pipeline_log":   [f"✅ Node 1 | Match → {row['career_domain']}"]}
    return {**state,
            "matched_career": None,
            "error":          "no_match",
            "nvidia_error":   "",
            "pipeline_log":   ["❌ Node 1 | Match → No domain found"]}

@traceable(name="groq_analyze_node")
def node_groq_analyze(state: CareerState) -> CareerState:
    if state.get("error"):
        return state
    row = state["matched_career"]
    llm = get_groq_llm()
    sal_inr = format_inr(int(row.get("avg_salary_usd", 60000)), str(row.get("career_domain","")))

    if llm is None:
        fallback = row.get("summary", "Add GROQ_API_KEY to enable Groq intelligence.")
        return {**state, "groq_output": fallback,
                "pipeline_log": ["⚠️  Node 2 | Groq → No API key (CSV fallback)"]}

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "You are a rapid career intelligence engine focused on the Indian job market. "
            "Generate 2-3 sentences of raw, direct, data-backed intelligence covering: "
            "market trajectory, salary outlook in Indian Rupees (₹ Lakhs), and the single biggest risk factor. "
            "Be specific and numbers-focused. Reference Indian market conditions where relevant. No fluff."
        )),
        HumanMessage(content=(
            f"Career Domain   : {row['career_domain']}\n"
            f"Future Score    : {row['future_score']}/100  |  Demand: {row['demand_score']}/100\n"
            f"Annual Growth   : {row['growth_rate']}%  |  Risk: {row['risk_level']}\n"
            f"Avg Salary (IN) : {sal_inr}/yr  |  Automation Risk: {row['automation_risk']}%\n"
            f"Market 2024→2030: ₹{float(row['market_size_2024_b'])*USD_TO_INR:.0f}B → ₹{float(row['projected_market_2030_b'])*USD_TO_INR:.0f}B (INR)\n"
            f"Job Openings    : {row['job_openings_k']}K+\n"
            f"Top Skills      : {row['required_skills']}\n\n"
            "Provide rapid raw career intelligence for an Indian professional."
        ))
    ])
    try:
        resp = (prompt | llm).invoke({})
        out  = _extract_content(resp)
        return {**state, "groq_output": out,
                "pipeline_log": [f"🟠 Node 2 | Groq LLaMA3-70B → {len(out)} chars"]}
    except Exception as e:
        fallback = row.get("summary", str(e))
        return {**state, "groq_output": fallback,
                "pipeline_log": [f"⚠️  Node 2 | Groq error: {str(e)[:70]}"]}

@traceable(name="openai_refine_node")
def node_openai_refine(state: CareerState) -> CareerState:
    if state.get("error"):
        return state
    row         = state["matched_career"]
    groq_output = state.get("groq_output", "")
    llm         = get_openai_llm()
    sal_inr     = format_inr(int(row.get("avg_salary_usd", 60000)), str(row.get("career_domain","")))

    if llm is None:
        return {**state, "openai_output": groq_output,
                "pipeline_log": ["⚠️  Node 3 | OpenAI → No API key (Groq output used)"]}

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "You are the refinement layer of CareerAI, focused on the Indian career landscape. "
            "Transform raw career intelligence into 3 polished, inspiring sentences. "
            "Structure: (1) Core opportunity in India, (2) Key challenge, (3) One actionable next step. "
            "Always mention salary in Indian Rupees (₹ Lakhs). "
            "Tone: empathetic, realistic, motivating. Flowing prose. No bullets."
        )),
        HumanMessage(content=(
            f"User query     : \"{state['query']}\"\n"
            f"Career domain  : {row['career_domain']}\n"
            f"Future Score   : {row['future_score']}/100  |  Avg Salary: {sal_inr}/yr\n\n"
            f"Raw Groq output:\n{groq_output}\n\n"
            "Refine into polished user-ready career advice for an Indian professional."
        ))
    ])
    try:
        resp = (prompt | llm).invoke({})
        out  = _extract_content(resp)
        return {**state, "openai_output": out,
                "pipeline_log": [f"🟢 Node 3 | OpenAI GPT-4o-mini → {len(out)} chars"]}
    except Exception as e:
        return {**state, "openai_output": groq_output,
                "pipeline_log": [f"⚠️  Node 3 | OpenAI error: {str(e)[:70]}"]}

@traceable(name="nvidia_roadmap_node")
def node_nvidia_roadmap(state: CareerState) -> CareerState:
    if state.get("error"):
        return state
    row           = state["matched_career"]
    openai_output = state.get("openai_output", "")
    groq_output   = state.get("groq_output",   "")
    llm           = get_nvidia_llm()
    sal_inr       = format_inr(int(row.get("avg_salary_usd", 60000)), str(row.get("career_domain","")))

    _FALLBACK = (
        "Phase 1 (0–6 months): Build foundational skills — focus on the top 3 skills listed above.\n"
        "Phase 2 (6–18 months): Complete a relevant certification or project portfolio.\n"
        "Phase 3 (1.5–3 years): Land your first role or transition; aim for mid-level growth.\n"
        "Phase 4 (3–5 years): Specialise in a high-demand sub-domain and build your network.\n"
        "Phase 5 (5+ years): Target senior/lead roles or start your own consultancy."
    )

    if llm is None:
        return {**state, "nvidia_output": _FALLBACK, "nvidia_error": "",
                "pipeline_log": ["⚠️  Node 4 | NVIDIA → No API key (default roadmap shown)"]}

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "You are NVIDIA CareerAI — a strategic career roadmap architect for Indian professionals. "
            "Create a precise, actionable 5-phase career roadmap. "
            "Label each phase exactly as:\n"
            "Phase 1 (0–6 months): ...\n"
            "Phase 2 (6–18 months): ...\n"
            "Phase 3 (1.5–3 years): ...\n"
            "Phase 4 (3–5 years): ...\n"
            "Phase 5 (5+ years): ...\n"
            "Each phase: 1-2 specific sentences. Include certifications, tools, or milestones. "
            "Mention salary expectations in Indian Rupees (₹ Lakhs) where relevant. "
            "Be concrete and achievable. No preamble. Start directly with Phase 1."
        )),
        HumanMessage(content=(
            f"Career Domain  : {row['career_domain']}\n"
            f"Future Score   : {row['future_score']}/100  |  Growth: {row['growth_rate']}%/yr\n"
            f"Avg Salary(IN) : {sal_inr}/yr  |  Risk: {row['risk_level']}\n"
            f"Key Skills     : {row['required_skills']}\n"
            f"Market 2030    : ₹{float(row['projected_market_2030_b'])*USD_TO_INR:.0f}B projected (INR)\n\n"
            f"Career Insight (from Groq + OpenAI):\n{openai_output or groq_output}\n\n"
            f"User's question: \"{state['query']}\"\n\n"
            "Build a precise 5-phase strategic career roadmap for an Indian professional."
        ))
    ])

    try:
        resp    = (prompt | llm).invoke({})
        content = _extract_content(resp)
        if not content.strip():
            raise ValueError("NVIDIA returned empty content")
        return {**state, "nvidia_output": content, "nvidia_error": "",
                "pipeline_log": [f"🔷 Node 4 | NVIDIA Llama-3.3-70B → {len(content)} chars"]}
    except Exception as e:
        err_detail = traceback.format_exc()
        return {**state, "nvidia_output": _FALLBACK, "nvidia_error": err_detail,
                "pipeline_log": [f"⚠️  Node 4 | NVIDIA error: {str(e)[:120]}"]}


def build_graph():
    g = StateGraph(CareerState)
    g.add_node("match",  node_match_career)
    g.add_node("groq",   node_groq_analyze)
    g.add_node("openai", node_openai_refine)
    g.add_node("nvidia", node_nvidia_roadmap)
    g.set_entry_point("match")
    g.add_conditional_edges(
        "match",
        lambda s: "end" if s.get("error") == "no_match" else "groq",
        {"groq": "groq", "end": END}
    )
    g.add_edge("groq",   "openai")
    g.add_edge("openai", "nvidia")
    g.add_edge("nvidia", END)
    return g.compile()

@st.cache_resource
def get_graph():
    return build_graph()

def run_pipeline(query: str) -> CareerState:
    return get_graph().invoke({
        "query":          query,
        "matched_career": None,
        "groq_output":    "",
        "openai_output":  "",
        "nvidia_output":  "",
        "nvidia_error":   "",
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
            f'<div class="score-ring" style="background:conic-gradient({c} 0deg {deg}deg,'
            f'rgba(30,41,59,0.6) {deg}deg 360deg)">'
            f'<span class="snum" style="color:{c}">{score}</span>'
            f'<span class="slbl">/ 100</span></div></div>')

def roadmap_html(nvidia_text: str) -> str:
    phase_icons  = ["rs-1","rs-2","rs-3","rs-4","rs-5"]
    phase_labels = ["0 – 6 Months","6 – 18 Months","1.5 – 3 Years","3 – 5 Years","5+ Years"]
    phase_nums   = ["01","02","03","04","05"]
    parts = re.split(r'Phase\s+\d+[^:]*:', nvidia_text, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) < 2:
        parts = [p.strip() for p in nvidia_text.split("\n") if p.strip()]
    html_steps = ""
    for i, content in enumerate(parts[:5]):
        cls = phase_icons[i]  if i < len(phase_icons)  else "rs-5"
        lbl = phase_labels[i] if i < len(phase_labels) else f"Phase {i+1}"
        num = phase_nums[i]   if i < len(phase_nums)   else str(i+1)
        html_steps += f"""
        <div class="roadmap-step">
          <div class="roadmap-icon {cls}">{num}</div>
          <div class="roadmap-text">
            <div class="roadmap-phase">📅 {lbl}</div>
            <div class="roadmap-content">{content}</div>
          </div>
        </div>"""
    return f'<div class="roadmap-container">{html_steps}</div>'


def render_career_card(row: dict, groq_out: str, openai_out: str,
                       nvidia_out: str, pipe_log: list,
                       nvidia_error: str = ""):
    sc   = int(row.get("future_score",            50))
    dem  = int(row.get("demand_score",            50))
    grw  = int(row.get("growth_rate",             10))
    risk = str(row.get("risk_level",          "Medium"))
    sal    = int(row.get("avg_salary_usd",       60000))
    domain = str(row.get("career_domain",            ""))
    auto   = int(row.get("automation_risk",          30))
    m24    = float(row.get("market_size_2024_b",     10))
    m30    = float(row.get("projected_market_2030_b",15))
    jobs   = float(row.get("job_openings_k",         50))
    sc_c   = score_color(sc)
    dem_c  = score_color(dem)

    sal_chip = format_inr_chip(sal, domain)

    def mkt_fmt(val_b_usd):
        val_cr = val_b_usd * USD_TO_INR * 100
        if val_cr >= 10_000:
            return f"₹{val_b_usd * USD_TO_INR:.0f}B"
        return f"₹{val_cr:,.0f}Cr"
    m24_str = mkt_fmt(m24)
    m30_str = mkt_fmt(m30)

    st.markdown(f"""
    <div class="glass-card gc-green">
      <div class="sec-label">Career Domain Identified</div>
      <div class="career-name">{row.get('career_domain','')}</div>
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
          <div class="metric-chip"><div class="mc-val" style="color:#fbbf24">{sal_chip}</div><div class="mc-lbl">Avg Salary</div></div>
          <div class="metric-chip"><div class="mc-val" style="color:#f87171">{auto}%</div><div class="mc-lbl">Auto-Risk</div></div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card" style="padding:1.05rem;margin-bottom:0">', unsafe_allow_html=True)
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
        <div style="display:flex;gap:0.8rem;margin-top:0.6rem;font-size:0.74rem;color:var(--text-muted);flex-wrap:wrap">
          <span>📊 Market 2024: <b style="color:var(--text-secondary)">{m24_str}</b></span>
          <span>🚀 2030 Proj: <b style="color:#34d399">{m30_str}</b></span>
          <span>💼 Openings: <b style="color:#a5b4fc">{jobs}K+</b></span>
        </div></div>""", unsafe_allow_html=True)

    st.markdown("**🛠 Required Skills**")
    st.markdown(skills_html(row.get("required_skills","")), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if groq_out:
        st.markdown(f"""
        <div class="stage-box sb-groq">
          <div class="stage-label">🟠 Stage 2 — Groq LLaMA3-70B · Raw Career Intelligence</div>
          <div class="stage-content">{groq_out}</div>
        </div>""", unsafe_allow_html=True)

    if openai_out and openai_out != groq_out:
        st.markdown(f"""
        <div class="stage-box sb-oai">
          <div class="stage-label">🟢 Stage 3 — OpenAI GPT-4o-mini · Refined Career Insight</div>
          <div class="stage-content">{openai_out}</div>
        </div>""", unsafe_allow_html=True)
    elif openai_out:
        st.markdown(f"""
        <div class="stage-box sb-oai">
          <div class="stage-label">🟢 AI Career Insight (Final)</div>
          <div class="stage-content">{openai_out}</div>
        </div>""", unsafe_allow_html=True)

    if nvidia_out:
        st.markdown(f"""
        <div class="stage-box sb-nv">
          <div class="stage-label">🔷 Stage 4 — NVIDIA Llama-3.3-70B · Strategic Career Roadmap</div>
          {roadmap_html(nvidia_out)}
        </div>""", unsafe_allow_html=True)

    if nvidia_error:
        with st.expander("⚠️ NVIDIA Debug Info — click to inspect"):
            st.markdown(f"""
            <div class="stage-box sb-err">
              <div class="stage-label">🔴 NVIDIA Error Detail</div>
              <div class="stage-content" style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;white-space:pre-wrap">{nvidia_error}</div>
            </div>""", unsafe_allow_html=True)

    if pipe_log:
        with st.expander("🔭 LangGraph Pipeline Trace"):
            for entry in pipe_log:
                st.markdown(
                    f'<div style="font-family:JetBrains Mono,monospace;font-size:0.74rem;'
                    f'color:var(--text-muted);padding:2px 0;border-bottom:1px solid var(--divider)">{entry}</div>',
                    unsafe_allow_html=True
                )


def render_compare(row_a: dict, row_b: dict):
    metrics = [
        ("Future Score",    "future_score",    100, "/100"),
        ("Market Demand",   "demand_score",    100, "/100"),
        ("Growth Rate",     "growth_rate",      70, "%/yr"),
        ("Automation Risk", "automation_risk", 100, "%"),
    ]
    winner = row_a["career_domain"] if int(row_a.get("future_score",0)) >= int(row_b.get("future_score",0)) else row_b["career_domain"]

    st.markdown(f"""
    <div class="glass-card gc-purple">
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.85rem;margin-bottom:0.85rem">
        <div style="text-align:center;padding:0.62rem;border-radius:10px;
                    background:rgba(99,102,241,0.14);border:1px solid rgba(99,102,241,0.35);
                    color:
        <div style="text-align:center;padding:0.62rem;border-radius:10px;
                    background:rgba(52,211,153,0.12);border:1px solid rgba(52,211,153,0.3);
                    color:
      </div>""", unsafe_allow_html=True)

    for lbl, key, mx, sfx in metrics:
        va, vb = int(row_a.get(key,0)), int(row_b.get(key,0))
        pa, pb = int(va/mx*100), int(vb/mx*100)
        ca = "#f87171" if key == "automation_risk" and va > vb else "#818cf8"
        cb = "#f87171" if key == "automation_risk" and vb > va else "#34d399"
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

    dom_a    = str(row_a.get("career_domain", ""))
    dom_b    = str(row_b.get("career_domain", ""))
    sal_a    = format_inr_chip(int(row_a.get("avg_salary_usd", 60000)), dom_a)
    sal_b    = format_inr_chip(int(row_b.get("avg_salary_usd", 60000)), dom_b)
    lakhs_a  = _normalise_salary_to_lakhs(int(row_a.get("avg_salary_usd", 60000)), dom_a)
    lakhs_b  = _normalise_salary_to_lakhs(int(row_b.get("avg_salary_usd", 60000)), dom_b)
    higher_a = lakhs_a >= lakhs_b
    st.markdown(f"""
    <div class="cmp-row">
      <div class="cmp-mname">Avg Salary</div>
      <div class="cmp-bwrap">
        <span class="cmp-val" style="color:{'#34d399' if higher_a else '#818cf8'}">{sal_a}/yr</span>
      </div>
      <div class="cmp-bwrap">
        <span class="cmp-val" style="color:{'#34d399' if not higher_a else '#818cf8'}">{sal_b}/yr</span>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="text-align:center;padding:0.85rem 0">
      <div style="font-size:0.68rem;color:var(--text-muted);margin-bottom:0.42rem">RECOMMENDED CHOICE</div>
      <span class="winner-badge">🏆 {winner}</span>
    </div></div>""", unsafe_allow_html=True)


groq_ok   = bool(os.getenv("GROQ_API_KEY",     ""))
openai_ok = bool(os.getenv("OPENAI_API_KEY",   ""))
ls_ok     = bool(os.getenv("LANGSMITH_API_KEY",""))
nvidia_ok = bool(os.getenv("NVIDIA_API_KEY",   ""))

with st.sidebar:
    st.markdown("""
    <div style="padding:0.65rem 0 0.35rem">
      <div style="font-size:1.35rem;font-weight:900;
                  background:linear-gradient(135deg,#818cf8,#34d399,#76a9fa);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent">
        🔮 CareerAI
      </div>
      <div style="font-size:0.68rem;color:var(--text-faint);margin-top:2px">4-Stage AI Pipeline · Career Forecasting</div>
    </div>
    <hr style="margin:0.38rem 0 0.6rem">
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-size:0.67rem;font-weight:700;letter-spacing:0.1em;color:var(--text-faint);text-transform:uppercase;margin-bottom:0.42rem">Pipeline Status</div>', unsafe_allow_html=True)

    def srow(icon, stage, label, ok):
        c   = "#34d399" if ok else "#f87171"
        txt = "Connected" if ok else "No Key"
        dot = "●" if ok else "○"
        return (f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'padding:5px 8px;border-radius:8px;margin-bottom:3px;'
                f'background:var(--trend-bg);border:1px solid var(--trend-border)">'
                f'<span style="font-size:0.74rem">{icon} <span style="color:var(--text-faint);font-size:0.62rem">{stage}</span> '
                f'<b style="color:var(--text-secondary)">{label}</b></span>'
                f'<span style="font-size:0.68rem;color:{c}">{dot} {txt}</span></div>')

    st.markdown(
        srow("🔵","S1","LangSmith Tracing",    ls_ok)     +
        srow("🟠","S2","Groq LLaMA3-70B",      groq_ok)   +
        srow("🟢","S3","OpenAI GPT-4o-mini",   openai_ok) +
        srow("🔷","S4","NVIDIA Llama-3.3-70B", nvidia_ok) +
        f'<div style="font-size:0.63rem;color:var(--text-faintest);padding:4px 8px">'
        f'⚡ LangGraph · 4-node pipeline · {len(DF)} careers loaded</div>',
        unsafe_allow_html=True
    )

    st.markdown('<hr style="margin:0.6rem 0">', unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:0.67rem;font-weight:700;letter-spacing:0.1em;color:var(--text-faint);text-transform:uppercase;margin-bottom:0.42rem">Try asking</div>
    <div style="font-size:0.66rem;color:var(--text-faintest);line-height:1.85">
      · Future of AI jobs in India<br>
      · Is cybersecurity worth it?<br>
      · B.Tech degree in 5 years<br>
      · Scope of prompt engineering<br>
      · Data science salary in India<br>
      · Cloud computing outlook<br>
      · Blockchain career scope
    </div>""", unsafe_allow_html=True)


st.markdown("""
<div class="hero-title">
  <h1>🔮 CareerAI</h1>
  <div class="hero-sub">4-Stage AI pipeline · ₹-based predictions for India · Strategic career roadmaps</div>
</div>""", unsafe_allow_html=True)

st.markdown("""
<div class="pipe-banner">
  <span class="pipe-step ps-ls">🔵 LangSmith</span>
  <span class="pipe-arrow">traces</span>
  <span class="pipe-step ps-gr">🟠 Groq LLaMA3</span>
  <span class="pipe-arrow">→</span>
  <span class="pipe-step ps-oai">🟢 OpenAI GPT-4o</span>
  <span class="pipe-arrow">→</span>
  <span class="pipe-step ps-nv">🔷 NVIDIA Llama-3.3</span>
  <span class="pipe-arrow">→</span>
  <span style="font-size:0.76rem;color:var(--text-faint);padding:0 4px">₹ Career Roadmap</span>
</div>""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🤖  Career Prediction", "⚔️  Compare Careers", "📊  Career Explorer"])

with tab1:
    cq, cb = st.columns([4, 1])
    with cq:
        query = st.text_input(
            "", placeholder="e.g. Scope of AI jobs in India · Is cybersecurity good? · B.Tech in 5 years · Data science salary",
            label_visibility="collapsed", key="main_query"
        )
    with cb:
        predict_btn = st.button("Predict 🔮", key="predict_btn")

    examples = ["AI jobs in India", "Cybersecurity scope", "B.Tech in 5 years",
                "Data science outlook", "Cloud computing", "Prompt engineering"]
    ecols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        if ecols[i].button(ex, key=f"ex_{i}"):
            query = ex; predict_btn = True

    if predict_btn and query.strip():
        with st.spinner("🔵 LangSmith tracing → 🟠 Groq analyzing → 🟢 OpenAI refining → 🔷 NVIDIA building roadmap…"):
            result = run_pipeline(query)

        if result.get("error") == "no_match" or not result.get("matched_career"):
            st.markdown("""
            <div class="glass-card" style="text-align:center;padding:2.5rem">
              <div style="font-size:3rem">🔍</div>
              <div style="color:var(--text-secondary);margin-top:0.75rem;font-size:1rem">No matching career domain found.</div>
              <div style="color:var(--text-faint);font-size:0.79rem;margin-top:0.4rem">
                Try: AI, cybersecurity, data science, cloud, blockchain, MBA, B.Tech, robotics…
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            render_career_card(
                result["matched_career"],
                result.get("groq_output",   ""),
                result.get("openai_output", ""),
                result.get("nvidia_output", ""),
                result.get("pipeline_log",  []),
                result.get("nvidia_error",  ""),
            )
    elif not query.strip():
        st.markdown("""
        <div class="glass-card" style="text-align:center;padding:3rem">
          <div style="font-size:4rem">🔮</div>
          <div style="color:var(--text-secondary);font-size:1.05rem;margin-top:0.75rem">Ask anything about career futures in India</div>
          <div style="color:var(--text-faint);font-size:0.79rem;margin-top:0.4rem;line-height:1.8">
            🔵 LangSmith traces every call &nbsp;→&nbsp; 🟠 Groq raw intelligence<br>
            🟢 OpenAI refined insight &nbsp;→&nbsp; 🔷 NVIDIA strategic roadmap<br>
            💰 All salaries shown in Indian Rupees (₹ Lakhs)
          </div>
        </div>""", unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="glass-card" style="padding:1.1rem">', unsafe_allow_html=True)
    ca_col, cb_col = st.columns(2)
    career_names = sorted(DF["career_domain"].tolist())
    with ca_col:
        st.markdown('<p style="color:#818cf8;font-size:0.74rem;font-weight:700;margin-bottom:4px;text-transform:uppercase;letter-spacing:0.08em">Career A</p>', unsafe_allow_html=True)
        sel_a = st.selectbox("", career_names, index=0, key="sela", label_visibility="collapsed")
    with cb_col:
        st.markdown('<p style="color:#34d399;font-size:0.74rem;font-weight:700;margin-bottom:4px;text-transform:uppercase;letter-spacing:0.08em">Career B</p>', unsafe_allow_html=True)
        sel_b = st.selectbox("", career_names, index=min(1, len(career_names)-1), key="selb", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Compare Careers ⚔️", key="cmp_btn"):
        if sel_a == sel_b:
            st.warning("⚠️ Select two different career domains.")
        else:
            ra = DF[DF["career_domain"] == sel_a].iloc[0].to_dict()
            rb = DF[DF["career_domain"] == sel_b].iloc[0].to_dict()
            render_compare(ra, rb)
            st.markdown("---")
            cc1, cc2 = st.columns(2)
            with cc1:
                render_career_card(ra, ra.get("summary",""), "", "", [])
            with cc2:
                render_career_card(rb, rb.get("summary",""), "", "", [])

with tab3:
    f1, f2, f3 = st.columns(3)
    with f1: risk_f = st.selectbox("Risk Level", ["All","Low","Medium","High"], key="rf")
    with f2: min_sc = st.slider("Min Future Score", 0, 100, 60, key="sff")
    with f3: sort_f = st.selectbox("Sort By", ["future_score","demand_score","growth_rate","avg_salary_usd"], key="sortf")

    filtered = DF.copy()
    if risk_f != "All":
        filtered = filtered[filtered["risk_level"].str.lower() == risk_f.lower()]
    filtered = filtered[filtered["future_score"] >= min_sc].sort_values(sort_f, ascending=False)

    st.markdown(f'<div style="color:var(--text-muted);font-size:0.78rem;margin-bottom:0.65rem">Showing <b style="color:var(--text-primary)">{len(filtered)}</b> career domains</div>', unsafe_allow_html=True)
    for _, row in filtered.iterrows():
        icon    = "🟢" if row["risk_level"]=="Low" else "🟡" if row["risk_level"]=="Medium" else "🔴"
        sal_lbl = format_inr_chip(int(row.get("avg_salary_usd", 60000)), str(row.get("career_domain","")))
        with st.expander(f"{icon}  {row['career_domain']}  ·  Score {int(row['future_score'])}/100  ·  {sal_lbl}/yr  ·  {row['growth_rate']}%/yr growth"):
            render_career_card(row.to_dict(), row.get("summary",""), "", "", [])

st.markdown("""
<hr>
<div style="text-align:center;padding:0.75rem;color:var(--text-faintest);font-size:0.71rem">
  🔮 <b style="color:var(--text-faint)">CareerAI</b> &nbsp;·&nbsp; 💰 Salaries in ₹ Lakhs (Indian Rupees) &nbsp;·&nbsp;
  🔵 LangSmith &nbsp;→&nbsp; 🟠 Groq LLaMA3-70B &nbsp;→&nbsp; 🟢 OpenAI GPT-4o-mini
  &nbsp;→&nbsp; 🔷 NVIDIA Llama-3.3-70B &nbsp;·&nbsp; Powered by LangGraph
</div>""", unsafe_allow_html=True)
