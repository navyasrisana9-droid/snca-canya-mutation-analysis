"""
Alpha-Synuclein Explorer
AI-Based Analysis of Alpha-Synuclein Aggregation and Neurodegenerative Disease Risk
By Sana Manchikanti

Single-file Streamlit dashboard. Put this file next to run_all.py (inside the
snca_project folder) and run:

    streamlit run app.py

Everything shown comes from files in results/. Where a file is missing the app
says so. The only invented numbers are the cross-disease "demo" charts, which
appear only when no real cross-protein file exists and are labelled as demo.
Project progress (the four phases and five options) is inferred from which
output files exist.
"""

import html as _html
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Alpha-Synuclein Explorer", page_icon="🧬", layout="wide")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
FASTA = BASE / "data" / "snca_P37840.fasta"
MISSING = "File not found. Run the pipeline to generate outputs."

PURPLE, PINK, BLUE, GREEN, BG = "#8b5cf6", "#ec4899", "#60a5fa", "#22c55e", "#fdf4ff"
INK = "#2b1b4e"

AA = "ACDEFGHIKLMNPQRSTVWY"
KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
      "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
      "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}

FAMILIAL = ["A30P", "E46K", "H50Q", "G51D", "A53T", "A53E"]
VALIDATION_SET = ["A30P", "E46K", "H50Q", "G51D", "A53T"]  # set used in the original test
LITERATURE = {
    "A30P": "Slower fibril formation than wild type in vitro (Sahay et al., J Biol Chem 2015).",
    "E46K": "Faster fibril formation than wild type in vitro (Sahay et al., J Biol Chem 2015).",
    "A53T": "Faster fibril formation than wild type in vitro (Sahay et al., J Biol Chem 2015).",
    "H50Q": "Aggregation rate not checked in this project.",
    "G51D": "Aggregation rate not checked in this project.",
    "A53E": "Aggregation rate not checked in this project.",
}

SCORE_FILES = ["step11_scores_with_ranks.csv", "step14_robustness.csv",
               "step9_mutation_scores.csv", "step15_features.csv"]

# Names for the research journey (four build stages) and the five discovery tracks
PHASE_TITLES = {1: "Meet the Protein", 2: "Mutation Forge", 3: "Oracle Scoring", 4: "Dataset Assembly"}
PHASE_SUBS = {1: "Get alpha-synuclein", 2: "Create every single mutation",
              3: "Score them with CANYA", 4: "Build the dataset"}
PHASE_ICONS = {1: "🧫", 2: "🧬", 3: "🔮", 4: "🗄️"}
OPTION_TITLES = {1: "Hotspot Hunt", 2: "Ask the Oracle", 3: "Neuro Galaxy",
                 4: "Truth Check", 5: "Mutation Radar"}
OPTION_SUBS = {1: "Mutation landscape mapping", 2: "Explainable machine learning",
               3: "Cross-disease comparison", 4: "Validation study", 5: "Mutation scanner tool"}
OPTION_ICONS = {1: "🔥", 2: "🔮", 3: "🌌", 4: "📊", 5: "📡"}


def phase_tag(n):
    return PHASE_TITLES[n]


def option_tag(n):
    return OPTION_TITLES[n]


P_HOME = "🏠 Home"
P_GLANCE = "✨ At a Glance"
P_EXPLORE = "🧬 Mutation Forge"
P_HOT = "🔥 Hotspot Hunt"
P_XAI = "🔮 Ask the Oracle"
P_PD = "🏆 Parkinson Legends"
P_CROSS = "🌌 Neuro Galaxy"
P_STATS = "📊 Truth Check"
P_DL = "📦 Data Vault"
P_ABOUT = "ℹ️ About"
NAV = [P_HOME, P_GLANCE, P_EXPLORE, P_HOT, P_XAI, P_PD, P_CROSS, P_STATS, P_DL, P_ABOUT]

SUBTITLES = {
    P_EXPLORE: "Exploring how tiny changes reshape an entire protein.",
    P_HOT: "Finding the most mutation-sensitive regions of alpha-synuclein.",
    P_XAI: "Let explainable AI reveal what drives aggregation risk.",
    P_PD: "Known Parkinson’s mutations and how they compare.",
    P_CROSS: "Comparing proteins across multiple neurodegenerative diseases.",
    P_STATS: "Testing whether our findings hold up under statistical scrutiny.",
    P_DL: "Download datasets, figures, and research outputs.",
}

MASCOT = {
    P_HOME: "Hi, I'm Helix! Pick a page and let's explore.",
    P_GLANCE: "The whole project on one screen.",
    P_EXPLORE: "Try Build a mutation, or press Surprise me!",
    P_HOT: "Hover over the tiles to peek at each residue.",
    P_XAI: "Pink pushes the score up, blue pushes it down. Then try the challenge!",
    P_PD: "These are the famous familial mutations.",
    P_CROSS: "Only alpha-synuclein so far. More to come!",
    P_STATS: "Small sample, so we stay humble here.",
    P_DL: "Grab anything you like.",
    P_ABOUT: "The honest limits live here.",
}

DONE, PROG, TODO = "done", "prog", "todo"
STATUS_TEXT = {DONE: "Complete", PROG: "In progress", TODO: "Not started"}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Unbounded:wght@500;700&family=Nunito:wght@400;600;700;800&display=swap');

html, body, .stApp, [data-testid="stSidebar"] { font-family: 'Nunito', 'Segoe UI', sans-serif; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4 { font-family: 'Unbounded', 'Nunito', sans-serif; letter-spacing: -0.01em; }

.stApp {
  color: #2b1b4e;
  background:
    linear-gradient(rgba(139,92,246,.055) 1px, transparent 1px) 0 0 / 46px 46px,
    linear-gradient(90deg, rgba(139,92,246,.055) 1px, transparent 1px) 0 0 / 46px 46px,
    radial-gradient(900px 520px at 8% -8%, rgba(236,72,153,.16), transparent 62%),
    radial-gradient(900px 520px at 100% 0%, rgba(96,165,250,.22), transparent 62%),
    radial-gradient(1000px 600px at 50% 115%, rgba(139,92,246,.16), transparent 62%),
    #fdf4ff;
  background-attachment: fixed;
}
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p, .stApp li, .stApp label,
.stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stMarkdownContainer"] { color: #2b1b4e; }
.stApp h3 { font-size: 1.15rem; font-weight: 700; margin-top: 1.4rem; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stHeader"] * { color: #2b1b4e; }

/* ---------- sidebar: companion console ---------- */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(255,255,255,.82) 0%, rgba(250,232,255,.78) 55%, rgba(224,236,255,.8) 100%);
  backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px);
  border-right: 1px solid rgba(139,92,246,.16);
}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding-top: .6rem; }
.brand { display: flex; align-items: center; gap: .8rem; padding: .9rem 1rem; margin-bottom: 1rem;
  background: rgba(255,255,255,.72); border-radius: 26px;
  border: 1px solid rgba(255,255,255,.9); box-shadow: 0 10px 28px rgba(139,92,246,.16); }
.avatar { flex: none; width: 3rem; height: 3rem; border-radius: 50%; display: grid; place-items: center;
  font-size: 1.5rem; background: linear-gradient(140deg, #8b5cf6, #ec4899 60%, #60a5fa);
  box-shadow: 0 6px 16px rgba(236,72,153,.35), inset 0 0 0 3px rgba(255,255,255,.55);
  animation: bob 3.6s ease-in-out infinite; }
@keyframes bob { 0%,100% { transform: translateY(0);} 50% { transform: translateY(-4px);} }
.brand .t1 { font-family: 'Unbounded', sans-serif; font-weight: 700; font-size: .92rem; line-height: 1.15; }
.brand .t2 { font-size: .78rem; opacity: .7; margin-top: .15rem; }
.bubble { position: relative; background: #ffffff; border-radius: 18px; padding: .6rem .85rem;
  font-size: .86rem; font-weight: 600; margin: 0 0 1rem 0; box-shadow: 0 6px 16px rgba(139,92,246,.14); }
.bubble::before { content: ""; position: absolute; top: -7px; left: 24px; border: 7px solid transparent;
  border-top: 0; border-bottom-color: #ffffff; }

[data-testid="stSidebar"] [role="radiogroup"] { gap: .3rem; }
[data-testid="stSidebar"] label[data-baseweb="radio"] {
  width: 100%; margin: 0; padding: .5rem .85rem; border-radius: 999px;
  background: rgba(255,255,255,.5); border: 1px solid rgba(139,92,246,.1);
  transition: transform .18s ease, background .18s ease, box-shadow .18s ease;
}
[data-testid="stSidebar"] label[data-baseweb="radio"] > div:first-child { display: none; }
[data-testid="stSidebar"] label[data-baseweb="radio"]:hover {
  transform: translateX(4px); background: rgba(255,255,255,.95); box-shadow: 0 6px 16px rgba(139,92,246,.16); }
[data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked) {
  background: linear-gradient(90deg, #8b5cf6, #ec4899); border-color: transparent;
  box-shadow: 0 8px 20px rgba(236,72,153,.32); }
[data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked) * { color: #ffffff !important; font-weight: 700; }
[data-testid="stSidebar"] label[data-baseweb="radio"] p { font-size: .95rem; }

.sidebox { background: rgba(255,255,255,.62); border-radius: 22px; padding: .9rem 1rem; margin-top: 1rem;
  border: 1px solid rgba(255,255,255,.9); box-shadow: 0 8px 22px rgba(139,92,246,.1); }
.sidebox .h { font-family: 'Unbounded', sans-serif; font-weight: 500; font-size: .8rem; margin-bottom: .55rem; }
.steps { border-left: 2px dashed rgba(139,92,246,.3); margin-left: .45rem; padding-left: 1rem; }
.step { display: flex; align-items: center; gap: .5rem; font-size: .86rem; padding: .18rem 0; }
.dot { flex: none; width: .78rem; height: .78rem; border-radius: 50%; margin-left: -1.4rem;
  border: 2px solid #c4b5fd; background: #fdf4ff; }
.dot.done { background: #22c55e; border-color: #22c55e; box-shadow: 0 0 0 3px rgba(34,197,94,.18); }
.dot.prog { background: linear-gradient(90deg, #ec4899 50%, #fdf4ff 50%); border-color: #ec4899; }
.opt { display: flex; align-items: center; gap: .5rem; font-size: .86rem; padding: .16rem 0; }
.opt .dot { margin-left: 0; }
.live { display: flex; align-items: center; gap: .5rem; font-size: .8rem; opacity: .8; margin-top: 1rem; padding-left: .3rem; }
.live .dot { margin-left: 0; width: .6rem; height: .6rem; }
.live .dot.done { animation: pulse 2.4s ease-in-out infinite; }
@keyframes pulse { 0%,100% { box-shadow: 0 0 0 0 rgba(34,197,94,.45);} 50% { box-shadow: 0 0 0 7px rgba(34,197,94,0);} }

/* ---------- surfaces ---------- */
.glass {
  background: rgba(255,255,255,.66);
  backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
  border: 1px solid rgba(255,255,255,.9);
  border-radius: 24px;
  box-shadow: 0 10px 30px rgba(139,92,246,.12);
  padding: 1.1rem 1.3rem;
  margin-bottom: 1rem;
  transition: border-color .2s ease, box-shadow .2s ease;
}
.glass:hover { border-color: rgba(236,72,153,.4); box-shadow: 0 14px 34px rgba(236,72,153,.16); }
.cardgrid { display: grid; gap: 1rem; margin-bottom: 1rem; }
.cardgrid .glass { margin-bottom: 0; word-break: normal; overflow-wrap: break-word; }

.m-title { font-size: .85rem; font-weight: 700; opacity: .75; }
.m-value { font-family: 'Unbounded', sans-serif; font-weight: 700; color: #8b5cf6; line-height: 1.25;
  font-size: clamp(1.1rem, 1.7vw, 1.55rem); margin: .25rem 0; overflow-wrap: break-word; word-break: normal; }
.m-sub { font-size: .88rem; opacity: .78; overflow-wrap: break-word; word-break: normal; }
.pill { display: inline-block; padding: .15rem .7rem; border-radius: 999px; font-size: .8rem;
  font-weight: 700; color: white; background: #8b5cf6; margin-top: .4rem; }
.tag { display: inline-block; padding: .1rem .6rem; border-radius: 999px; font-size: .78rem; font-weight: 700;
  background: rgba(139,92,246,.12); color: #6d28d9; margin-bottom: .4rem; }
.tag.o { background: rgba(236,72,153,.12); color: #be185d; }
.status { display: inline-block; padding: .1rem .65rem; border-radius: 999px; font-size: .78rem; font-weight: 700; }
.status.done { background: rgba(34,197,94,.16); color: #15803d; }
.status.prog { background: rgba(250,204,21,.25); color: #a16207; }
.status.todo { background: rgba(107,91,149,.14); color: #6b5b95; }
.num { font-family: 'Unbounded', sans-serif; font-weight: 700; font-size: 2rem; line-height: 1;
  background: linear-gradient(135deg, #8b5cf6, #ec4899); -webkit-background-clip: text; background-clip: text; color: transparent; }
.ctitle { font-family: 'Unbounded', sans-serif; font-weight: 500; font-size: 1rem; margin: .45rem 0 .1rem 0; }
.csub { font-size: .85rem; opacity: .7; margin-bottom: .5rem; }

.pbar { height: 16px; border-radius: 999px; background: rgba(139,92,246,.15); overflow: hidden; }
.pbar > span { display: block; height: 100%; border-radius: 999px;
  background: linear-gradient(90deg, #8b5cf6, #ec4899, #60a5fa); transition: width .6s ease; }
.check { font-size: .9rem; padding: .12rem 0; }
.check.ok { color: #15803d; }
.check.no { opacity: .6; }

/* ---------- hero: the one big moment ---------- */
.hero {
  position: relative; overflow: hidden; display: flex; flex-wrap: wrap; align-items: center; gap: 1.5rem;
  justify-content: space-between;
  background: radial-gradient(520px 260px at 90% 0%, rgba(96,165,250,.55), transparent 70%),
              radial-gradient(420px 260px at 0% 100%, rgba(236,72,153,.55), transparent 70%),
              linear-gradient(130deg, #4c1d95 0%, #7c3aed 48%, #c026d3 100%);
  border-radius: 34px; padding: 2.4rem 2.4rem; margin-bottom: 1.4rem;
  box-shadow: 0 22px 54px rgba(124,58,237,.38);
}
.hero * { color: #ffffff !important; }
.hero h1 { font-size: clamp(1.7rem, 3.4vw, 2.7rem); font-weight: 700; line-height: 1.12; margin: 0; }
.hero .sub { font-size: 1.05rem; opacity: .95; margin-top: .8rem; max-width: 34rem; line-height: 1.5; }
.hero .by { margin-top: .9rem; font-weight: 700; opacity: .92; }
.chips { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: 1.1rem; }
.chip { padding: .25rem .8rem; border-radius: 999px; font-size: .82rem; font-weight: 700;
  background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.36); }
.helix { display: flex; gap: 7px; height: 96px; align-items: flex-start; padding: 0 .6rem; }
.rung { position: relative; width: 11px; height: 96px; }
.rung i { position: absolute; left: 0; width: 11px; height: 11px; border-radius: 50%; background: #fff;
  box-shadow: 0 0 12px rgba(255,255,255,.8); animation: strand 2.8s ease-in-out infinite; }
.rung i.b { background: #fbcfe8; animation-direction: reverse; }
@keyframes strand {
  0%,100% { transform: translateY(0) scale(1); opacity: 1; }
  50% { transform: translateY(84px) scale(.65); opacity: .55; }
}
@media (prefers-reduced-motion: reduce) { .rung i, .live .dot.done, .avatar { animation: none; } }

/* ---------- protein ribbon ---------- */
.ribbon { display: flex; flex-wrap: wrap; gap: .8rem 1.1rem; }
.grp .gnum { font-size: .7rem; opacity: .6; margin-bottom: 2px; }
.grp .row { display: flex; gap: 2px; }
.tile { width: 1.5rem; height: 1.75rem; display: grid; place-items: center; font-weight: 800; font-size: .82rem;
  border-radius: 8px; border-bottom: 3px solid transparent; transition: transform .12s ease, box-shadow .12s ease; }
.tile.nac { border-bottom-color: #8b5cf6; }
.tile.win { outline: 2px solid #ec4899; outline-offset: 1px; }
.tile.sel { outline: 3px solid #2b1b4e; outline-offset: 1px; position: relative; z-index: 2; }
.tile:hover { transform: scale(1.5); position: relative; z-index: 3; box-shadow: 0 8px 18px rgba(139,92,246,.4); }

/* ---------- page header ---------- */
.pagehead { margin: .2rem 0 1.3rem 0; }
.pagehead h2 { font-size: clamp(1.4rem, 2.6vw, 2rem); font-weight: 700; margin: 0;
  background: linear-gradient(90deg, #7c3aed, #ec4899); -webkit-background-clip: text; background-clip: text; color: transparent; }
.pagehead p { margin: .35rem 0 .5rem 0; opacity: .8; font-size: 1.02rem; max-width: 44rem; }

.banner-demo { background: rgba(250,204,21,.25); border: 2px dashed #ca8a04; border-radius: 18px;
  padding: .8rem 1.1rem; margin: .5rem 0 1rem 0; font-weight: 700; }
.flow { display: flex; flex-wrap: wrap; align-items: center; gap: .5rem; margin: .4rem 0 1.2rem 0; }
.flow .node { padding: .35rem .9rem; border-radius: 999px; background: rgba(255,255,255,.75);
  border: 1px solid rgba(139,92,246,.25); font-weight: 700; font-size: .88rem; }
.flow .arrow { color: #ec4899; font-weight: 800; }
.footer { text-align: center; opacity: .65; font-size: .82rem; margin: 2.4rem 0 .6rem 0; }

/* ---------- widgets ---------- */
.stButton > button { border-radius: 999px; font-weight: 700; color: #2b1b4e;
  background: rgba(255,255,255,.8); border: 1px solid rgba(139,92,246,.3); transition: all .18s ease; }
.stButton > button:hover { border-color: #ec4899; color: #be185d; box-shadow: 0 6px 16px rgba(236,72,153,.2); }
[data-baseweb="input"] { border-radius: 16px !important; }
.stTabs [data-baseweb="tab"] { font-weight: 700; }

/* ---------- startup polish: gentle motion ---------- */
@keyframes fadeUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
@keyframes drift { 0%,100% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } }
@keyframes floaty { 0%,100% { transform: translateY(0) rotate(0deg); } 50% { transform: translateY(-9px) rotate(6deg); } }
[data-testid="stMainBlockContainer"], .block-container { animation: fadeUp .5s ease both; }
.glass:hover { transform: translateY(-3px); }
.hero { background-size: 100% 100%, 100% 100%, 220% 220%; animation: drift 14s ease-in-out infinite; }
.hero .float { position: absolute; font-size: 1.5rem; opacity: .35; animation: floaty 6s ease-in-out infinite; pointer-events: none; }
.eyebrow { display: inline-block; font-size: .74rem; font-weight: 800; letter-spacing: .14em; text-transform: uppercase;
  padding: .15rem .7rem; border-radius: 999px; background: rgba(255,255,255,.2); border: 1px solid rgba(255,255,255,.4); margin-bottom: .8rem; }
.sparkle { font-size: 1.4rem; }
@media (prefers-reduced-motion: reduce) {
  .hero, .hero .float, .glass, [data-testid="stMainBlockContainer"], .block-container { animation: none; transition: none; }
}
</style>
"""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def esc(x):
    return _html.escape(str(x))


def html(block):
    """Render an HTML block (leading spaces stripped so Markdown never makes code blocks)."""
    lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
    st.markdown("\n".join(lines), unsafe_allow_html=True)


def card_html(title, value, sub="", accent=PURPLE):
    return (f'<div class="glass" style="border-top:5px solid {accent}">'
            f'<div class="m-title">{esc(title)}</div>'
            f'<div class="m-value" style="color:{accent}">{esc(value)}</div>'
            f'<div class="m-sub">{esc(sub)}</div></div>')


def card_grid(cards, min_px=220):
    """Responsive row of cards: wraps onto new lines instead of squeezing text."""
    st.markdown(
        f'<div class="cardgrid" style="grid-template-columns: repeat(auto-fit, minmax({min_px}px, 1fr));">'
        + "".join(cards) + "</div>",
        unsafe_allow_html=True,
    )


def info_card_html(title, body, accent=PURPLE, icon="", tag="", option=False):
    tag_html = f'<div class="tag{" o" if option else ""}">{esc(tag)}</div>' if tag else ""
    return (f'<div class="glass" style="border-left:6px solid {accent}">{tag_html}'
            f'<div style="font-weight:800;font-size:1.02rem">{icon} {esc(title)}</div>'
            f'<div style="opacity:.85;margin-top:.3rem">{esc(body)}</div></div>')


def section(title):
    st.markdown(f"### {title}")


def page_header(title, subtitle="", tags=()):
    chips = "".join(f'<span class="tag{" o" if i % 2 else ""}" '
                    f'style="margin-right:.35rem">{esc(t)}</span>' for i, t in enumerate(tags))
    html(f'<div class="pagehead"><h2>{esc(title)}</h2><p>{esc(subtitle)}</p>{chips}</div>')


def warn_missing(extra=""):
    st.warning(MISSING + (f" ({extra})" if extra else ""))


def demo_banner(text):
    html(f'<div class="banner-demo">🧪 Demo data, not results from this study. {esc(text)}</div>')


def style_plot(fig):
    """Make every chart readable: white card, black text, light grid, clear axis labels."""
    black = "#000000"
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0.92)", plot_bgcolor="#ffffff",
        font=dict(color=black, family="Nunito, sans-serif", size=14),
        title=dict(font=dict(color=black, size=18, family="Nunito, sans-serif"), x=0.01, xanchor="left"),
        legend=dict(font=dict(color=black)),
        margin=dict(l=70, r=30, t=70, b=70),
        hoverlabel=dict(bgcolor="#ffffff", font=dict(color=black)),
    )
    axis = dict(
        title_font=dict(color=black, size=14), tickfont=dict(color=black, size=12),
        gridcolor="rgba(0,0,0,0.10)", zerolinecolor="rgba(0,0,0,0.35)",
        linecolor="rgba(0,0,0,0.45)", showline=True, automargin=True,
    )
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    fig.update_annotations(font=dict(color=black, size=13))
    try:
        fig.update_traces(colorbar=dict(tickfont=dict(color=black), title_font=dict(color=black)),
                          selector=dict(type="heatmap"))
    except Exception:
        pass
    return fig


def show_plot(fig, key=None):
    style_plot(fig)
    try:
        st.plotly_chart(fig, use_container_width=True, key=key, theme=None)
    except Exception:
        try:
            st.plotly_chart(fig, width="stretch", key=key, theme=None)
        except Exception:
            st.plotly_chart(fig, key=key)


def show_df(df, height=None):
    kwargs = {} if height is None else {"height": height}
    try:
        st.dataframe(df, use_container_width=True, hide_index=True, **kwargs)
    except Exception:
        st.dataframe(df, width="stretch", hide_index=True, **kwargs)


def show_image(name, caption=""):
    path = RESULTS / name
    if not path.exists():
        warn_missing(name)
        return False
    try:
        st.image(str(path), caption=caption or None, use_container_width=True)
    except Exception:
        try:
            st.image(str(path), caption=caption or None, width="stretch")
        except Exception:
            warn_missing(name)
            return False
    return True


@st.cache_data(show_spinner=False)
def read_csv_cached(path_str, mtime):
    return pd.read_csv(path_str)


def read_csv_safe(path):
    path = Path(path)
    try:
        if path.exists():
            return read_csv_cached(str(path), path.stat().st_mtime)
    except Exception:
        return None
    return None


def get_scores():
    """Per-mutation table from the pipeline (None if no usable file)."""
    for name in SCORE_FILES:
        df = read_csv_safe(RESULTS / name)
        if df is None or not {"mutation", "position", "delta"}.issubset(df.columns):
            continue
        df = df.dropna(subset=["mutation", "position", "delta"]).copy()
        df["position"] = df["position"].astype(int)
        df["mutation"] = df["mutation"].astype(str)
        if "wt_aa" not in df.columns:
            df["wt_aa"] = df["mutation"].str[0]
        if "mut_aa" not in df.columns:
            df["mut_aa"] = df["mutation"].str[-1]
        df["delta_rank_pct"] = df["delta"].rank(pct=True) * 100
        df["rank"] = df["delta"].rank(ascending=False, method="min").astype(int)
        return df
    return None


def effect_category(pct):
    if pct >= 90:
        return "Strong increase (top 10%)"
    if pct >= 75:
        return "Moderate increase (top 10-25%)"
    if pct <= 10:
        return "Strong decrease (bottom 10%)"
    if pct <= 25:
        return "Moderate decrease (bottom 10-25%)"
    return "Near typical"


def region_of(pos):
    if pos <= 60:
        return "N-terminal region (1-60)"
    if pos <= 95:
        return "NAC region (61-95)" + (", core 71-82" if 71 <= pos <= 82 else "")
    return "C-terminal region (96-140)"


def find_cross_file():
    if not RESULTS.exists():
        return None
    for p in sorted(RESULTS.glob("*.csv")):
        n = p.name.lower()
        if "cross" in n or "proteins" in n:
            return p
    return None


def results_files():
    if not RESULTS.exists():
        return []
    return sorted(p for p in RESULTS.rglob("*")
                  if p.is_file() and p.suffix.lower() in {".csv", ".png", ".txt"})


def fasta_length():
    try:
        if FASTA.exists():
            return sum(len(ln.strip()) for ln in FASTA.read_text().splitlines() if not ln.startswith(">"))
    except Exception:
        pass
    return 0


def count_lines(path, header=False):
    try:
        if Path(path).exists():
            with open(path, "r", errors="ignore") as f:
                n = sum(1 for _ in f)
            return max(n - (1 if header else 0), 0)
    except Exception:
        pass
    return 0


# ------------------- project progress: 4 phases and 5 options ------------------- #
def phase_info():
    df = get_scores()
    seq_len = fasta_length()
    n_mut = count_lines(RESULTS / "mutations.csv", header=True)
    n_pep = count_lines(RESULTS / "step6_canya_output.tsv")
    n_ds = len(df) if df is not None else 0
    items = [
        (DONE if seq_len else TODO,
         f"UniProt P37840, {seq_len} residues" if seq_len else "data/snca_P37840.fasta not found", P_HOME,
         ["data/snca_P37840.fasta"]),
        (DONE if n_mut else TODO,
         f"{n_mut:,} single substitutions generated" if n_mut else "results/mutations.csv not found", P_EXPLORE,
         ["results/mutations.csv"]),
        (DONE if n_pep else TODO,
         f"{n_pep:,} peptide windows scored" if n_pep else "results/step6_canya_output.tsv not found", P_XAI,
         ["results/step6_canya_input.tsv", "results/step6_canya_output.tsv", "results/step8_scored_windows.csv"]),
        (DONE if n_ds else TODO,
         f"{n_ds:,} mutations with a score and a change vs wild type" if n_ds else "no score table found", P_DL,
         ["results/step9_mutation_scores.csv", "results/step11_scores_with_ranks.csv"]),
    ]
    return [dict(n=i + 1, title=PHASE_TITLES[i + 1], icon=PHASE_ICONS[i + 1], sub=PHASE_SUBS[i + 1], status=s,
                 detail=d, page=pg, files=f)
            for i, (s, d, pg, f) in enumerate(items)]


def option_info():
    df = get_scores()
    cross = find_cross_file()

    # Hotspot Hunt: landscape mapping
    o1 = TODO
    o1_text = "Position-by-position sensitivity map not found."
    if (RESULTS / "step10_heatmap.png").exists() or (RESULTS / "step19_position_summary.csv").exists():
        o1 = DONE
        o1_text = "Sensitivity map built for all positions."
        if df is not None:
            core = df[df["position"].between(61, 82)]["delta"].abs().mean()
            rest = df[~df["position"].between(61, 82)]["delta"].abs().mean()
            if rest:
                o1_text = f"Residues 61-82 change the score {core / rest:.1f} times more than the rest of the protein."

    # Ask the Oracle: explainable ML
    o2 = DONE if (RESULTS / "step17_shap_summary.png").exists() else TODO
    o2_text = "No SHAP output found."
    if o2 == DONE:
        o2_text = "SHAP shows which amino-acid properties drive the predictions."
        pred = read_csv_safe(RESULTS / "step16_predictions.csv")
        if pred is not None and {"delta", "predicted_delta"}.issubset(pred.columns):
            try:
                rho = pred["delta"].corr(pred["predicted_delta"], method="spearman")
                resid = ((pred["delta"] - pred["predicted_delta"]) ** 2).sum()
                total = ((pred["delta"] - pred["delta"].mean()) ** 2).sum()
                r2 = 1 - resid / total if total else float("nan")
                o2_text = f"Model ranks effects with Spearman {rho:.2f} on held-out regions"
                o2_text += ("; it does not predict their size (R-squared near 0)." if r2 < 0.1
                            else f"; R-squared is {r2:.2f}.")
            except Exception:
                pass

    # Neuro Galaxy: cross-disease
    if cross is not None:
        o3, o3_text = DONE, f"Results loaded from {cross.name}."
    else:
        o3, o3_text = TODO, "Tau, amyloid-beta, huntingtin and TDP-43 have not been run yet."

    # Truth Check: validation
    o4, o4_text = TODO, "Score table needed for validation."
    if df is not None:
        hit = df[df["mutation"].isin(FAMILIAL)]
        o4 = DONE
        if len(hit):
            lo_p, hi_p = hit["delta_rank_pct"].min(), hit["delta_rank_pct"].max()
            o4_text = f"Known Parkinson's mutations sit at the {lo_p:.0f}th to {hi_p:.0f}th percentile"
            if hi_p >= 90:
                o4_text += ", and at least one ranks in the top 10%."
            elif hi_p <= 75:
                o4_text += ", so CANYA does not single them out."
            else:
                o4_text += "."
        else:
            o4_text = "Known mutations not found in the score table."

    # Mutation Radar: scanner tool
    if (BASE / "run_all.py").exists():
        o5, o5_text = PROG, ("run_all.py rebuilds the analysis and this dashboard searches every single "
                             "mutation. Scoring any new sequence is not built yet.")
    else:
        o5, o5_text = TODO, "No run_all.py found."

    cross_name = f"results/{cross.name}" if cross is not None else "results/ a CSV with 'cross' in its name"
    items = [
        (o1, o1_text, P_HOT, ["results/step10_heatmap.png", "results/step12_wt_profile.png",
                              "results/step19_position_sensitivity.png", "results/step19_position_summary.csv"]),
        (o2, o2_text, P_XAI, ["results/step15_features.csv", "results/step16_predictions.csv",
                              "results/step17_shap_summary.png"]),
        (o3, o3_text, P_CROSS, [cross_name]),
        (o4, o4_text, P_STATS, ["results/step11_scores_with_ranks.csv", "results/step14_robustness.csv",
                                "results/step13_top10_increase.csv", "results/step13_top10_decrease.csv"]),
        (o5, o5_text, P_EXPLORE, ["run_all.py", "app.py"]),
    ]
    return [dict(n=i + 1, title=OPTION_TITLES[i + 1], icon=OPTION_ICONS[i + 1], sub=OPTION_SUBS[i + 1],
                 status=s, text=t, page=pg, files=f)
            for i, (s, t, pg, f) in enumerate(items)]


@st.cache_data(show_spinner=False)
def permutation_test(all_vals, known_vals, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    k = len(known_vals)
    means = np.array([rng.choice(all_vals, size=k, replace=False).mean() for _ in range(n)])
    obs = float(np.mean(known_vals))
    return obs, float((means >= obs).mean()), float((means <= obs).mean()), means


@st.cache_data(show_spinner=False)
def bootstrap_ci(vals, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.choice(vals, size=(n, len(vals)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def heat_colorscale():
    return [[0, BLUE], [0.5, "#ffffff"], [1, PINK]]


def goto(page):
    st.session_state["nav"] = page


def helix_html(rungs=12):
    parts = []
    for k in range(rungs):
        d = round(k * 0.23, 2)
        parts.append(f'<span class="rung"><i style="top:0;animation-delay:-{d}s"></i>'
                     f'<i class="b" style="top:0;animation-delay:-{d}s"></i></span>')
    return '<div class="helix">' + "".join(parts) + "</div>"


def progress_value():
    items = phase_info() + option_info()
    score = sum(1.0 if i["status"] == DONE else 0.5 if i["status"] == PROG else 0.0 for i in items)
    return score, len(items)


def ribbon_html(df, highlight=None, window=None):
    """Clickless 'hover to explore' strip: one tile per residue, pink = more sensitive."""
    per = df.groupby("position").agg(wt=("wt_aa", "first"), v=("delta", lambda s: s.abs().mean()))
    vmax = float(per["v"].max()) or 1.0
    L = int(per.index.max())
    groups = []
    for start in range(1, L + 1, 10):
        tiles = []
        for pos in range(start, min(start + 10, L + 1)):
            if pos not in per.index:
                continue
            wt, v = per.loc[pos, "wt"], float(per.loc[pos, "v"])
            alpha = 0.08 + 0.92 * (v / vmax)
            cls = "tile"
            if 61 <= pos <= 95:
                cls += " nac"
            if window and window[0] <= pos <= window[1]:
                cls += " win"
            if highlight is not None and pos == highlight:
                cls += " sel"
            tiles.append(f'<span class="{cls}" title="{esc(wt)}{pos}: mean |change| {v:.4f}" '
                         f'style="background:rgba(236,72,153,{alpha:.2f})">{esc(wt)}</span>')
        groups.append(f'<div class="grp"><div class="gnum">{start}-{min(start + 9, L)}</div>'
                      f'<div class="row">{"".join(tiles)}</div></div>')
    return '<div class="ribbon">' + "".join(groups) + "</div>"


# --------------------------------------------------------------------------- #
# Oracle Challenge (guessing game) logic
# --------------------------------------------------------------------------- #
def game_init():
    ss = st.session_state
    for k, v in {"g_score": 0, "g_rounds": 0, "g_streak": 0, "g_best": 0,
                 "g_pair": None, "g_result": None}.items():
        if k not in ss:
            ss[k] = v


def game_new_round():
    df = get_scores()
    if df is None or len(df) < 2:
        return
    rng = np.random.default_rng()
    a = b = None
    for _ in range(300):
        i, j = rng.choice(len(df), size=2, replace=False)
        a, b = df.iloc[int(i)], df.iloc[int(j)]
        if abs(float(a["delta"]) - float(b["delta"])) >= 0.008:
            break
    st.session_state["g_pair"] = (str(a["mutation"]), str(b["mutation"]))
    st.session_state["g_result"] = None


def game_guess(which):
    ss = st.session_state
    df = get_scores()
    if df is None or ss.get("g_pair") is None:
        return
    a = df[df["mutation"] == ss["g_pair"][0]].iloc[0]
    b = df[df["mutation"] == ss["g_pair"][1]].iloc[0]
    correct = 0 if float(a["delta"]) >= float(b["delta"]) else 1
    ok = which == correct
    ss["g_rounds"] += 1
    if ok:
        ss["g_score"] += 1
        ss["g_streak"] += 1
        ss["g_best"] = max(ss["g_best"], ss["g_streak"])
    else:
        ss["g_streak"] = 0
    ss["g_result"] = (which, correct, ok)


def game_reset():
    for k in ["g_score", "g_rounds", "g_streak", "g_best"]:
        st.session_state[k] = 0
    game_new_round()


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_home():
    df = get_scores()
    cross = find_cross_file()
    n_mut = f"{len(df):,}" if df is not None else "n/a"
    n_prot = "1"
    prot_sub = "Alpha-synuclein (others not yet analyzed)"
    if cross is not None:
        cdf = read_csv_safe(cross)
        if cdf is not None and len(cdf):
            n_prot = str(cdf.iloc[:, 0].nunique())
            prot_sub = f"from {cross.name}"
    xgb = bool(list(RESULTS.glob("step24*"))) if RESULTS.exists() else False
    models = "Random Forest + XGBoost" if xgb else "Random Forest"
    shap_on = (RESULTS / "step17_shap_summary.png").exists()

    chips = ["CANYA nucleation model", models, "UniProt P37840"]
    html(
        '<div class="hero"><span class="float" style="top:12%;left:46%">🧬</span>'
        '<span class="float" style="top:68%;left:58%;animation-delay:-2s">✨</span>'
        '<span class="float" style="top:20%;left:80%;animation-delay:-4s">🧠</span><div>'
        '<span class="eyebrow">Computational neuroscience platform</span>'
        '<h1>🧬 Alpha-Synuclein Explorer</h1>'
        '<div class="sub">AI-Based Analysis of Alpha-Synuclein Aggregation and Neurodegenerative Disease Risk</div>'
        '<div class="by">By Sana Manchikanti</div>'
        '<div class="chips">' + "".join(f'<span class="chip">{esc(c)}</span>' for c in chips) + '</div>'
        '</div>' + helix_html() + '</div>'
    )

    card_grid([
        card_html("Mutations analyzed", n_mut, "single amino-acid substitutions", PURPLE),
        card_html("Proteins studied", n_prot, prot_sub, PINK),
        card_html("Models", models, "rank mutation effects", BLUE),
        card_html("SHAP", "Enabled" if shap_on else "Not run yet", "feature explanations", GREEN),
    ], 210)

    section("🧬 The Research Journey")
    phases = phase_info()
    card_grid([
        f'<div class="glass"><div class="sparkle">{p["icon"]}</div>'
        f'<div class="ctitle">{esc(p["title"])}</div><div class="csub">{esc(p["sub"])}</div>'
        f'<span class="status {p["status"]}">{STATUS_TEXT[p["status"]]}</span>'
        f'<div class="m-sub" style="margin-top:.5rem">{esc(p["detail"])}</div></div>'
        for p in phases
    ], 210)
    st.caption("Open ✨ At a Glance to click through every stage and discovery track.")

    section("🚀 Pick an adventure")
    cards = [
        ("Mutation Forge", "Build a mutation, search one, or press Surprise me.",
         PURPLE, "🧬", OPTION_TITLES[5], False),
        ("Hotspot Hunt", "Hover over every residue. The hot zone is residues about 61-82.",
         PINK, "🔥", OPTION_TITLES[1], True),
        ("Ask the Oracle", "A random forest and SHAP show which properties drive the predictions. Then play the challenge.",
         GREEN, "🔮", OPTION_TITLES[2], False),
        ("Parkinson Legends", "Familial mutations compared with CANYA's predictions and published work.",
         BLUE, "🏆", "Known mutations", True),
        ("Neuro Galaxy", "Tau, amyloid-beta, huntingtin and TDP-43 through the same pipeline.",
         PURPLE, "🌌", OPTION_TITLES[3], False),
        ("Truth Check", "Permutation tests and bootstrap intervals on the known mutations.",
         PINK, "📊", OPTION_TITLES[4], True),
    ]
    card_grid([info_card_html(t, b, a, i, tag, option=o) for (t, b, a, i, tag, o) in cards], 260)

    st.info("All scores are model predictions, not laboratory measurements, and this site is "
            "not medical advice. See the About page for limitations.")


def page_glance():
    page_header("At a Glance", "The whole project, its key numbers and its progress on one screen.",
                tags=("Project dashboard",))
    df = get_scores()
    cross = find_cross_file()
    n_mut = f"{len(df):,}" if df is not None else "2,660"
    n_mut_sub = "single amino-acid substitutions" if df is not None else "expected: 140 positions x 19 substitutions"
    n_prot, prot_sub = "5", "1 analyzed (alpha-synuclein); tau, Aβ, huntingtin, TDP-43 planned"
    if cross is not None:
        cdf = read_csv_safe(cross)
        if cdf is not None and len(cdf):
            n_prot, prot_sub = str(max(cdf.iloc[:, 0].nunique(), 5)), f"results loaded from {cross.name}"
    xgb = bool(list(RESULTS.glob("step24*"))) if RESULTS.exists() else False
    shap_on = (RESULTS / "step17_shap_summary.png").exists()
    card_grid([
        card_html("🧬 Mutations analyzed", n_mut, n_mut_sub, PURPLE),
        card_html("🌌 Disease proteins", n_prot, prot_sub, PINK),
        card_html("🌲 Machine learning", "Random Forest", "held-out protein blocks", BLUE),
        card_html("⚡ Cross-check", "XGBoost", "check complete" if xgb else "check not run yet", GREEN),
        card_html("🔮 Explainability", "SHAP", "feature explanations ready" if shap_on else "not run yet", PURPLE),
    ], 200)
    mission_map()


def mission_map():
    phases = phase_info()
    options = option_info()
    score, total = progress_value()
    pct = score / total * 100 if total else 0
    html(f'<div class="glass"><div style="display:flex;justify-content:space-between;font-weight:800">'
         f'<span>Mission progress</span><span>{score:g} of {total} steps</span></div>'
         f'<div class="pbar" style="margin-top:.6rem"><span style="width:{pct:.0f}%"></span></div>'
         f'<div class="m-sub" style="margin-top:.5rem">Status comes from the files in results/. '
         f'A step counts as complete when its output exists; half credit for in progress.</div></div>')

    html('<div class="flow">'
         + '<span class="arrow">&rarr;</span>'.join(
             f'<span class="node">{p["icon"]} {esc(p["title"])}</span>' for p in phases)
         + '<span class="arrow">&rarr;</span><span class="node" style="border-color:#ec4899">🔭 Discovery tracks</span></div>')

    section("🧬 The Research Journey")
    card_grid([
        f'<div class="glass" style="border-top:5px solid {PURPLE}"><div class="sparkle">{p["icon"]}</div>'
        f'<div class="ctitle">{esc(p["title"])}</div><div class="csub">{esc(p["sub"])}</div>'
        f'<span class="status {p["status"]}">{STATUS_TEXT[p["status"]]}</span>'
        f'<div class="m-sub" style="margin-top:.55rem">{esc(p["detail"])}</div></div>'
        for p in phases
    ], 220)

    section("🔭 Discovery Tracks")
    card_grid([
        f'<div class="glass" style="border-top:5px solid {PINK}"><div class="sparkle">{o["icon"]}</div>'
        f'<div class="ctitle">{esc(o["title"])}</div><div class="csub">{esc(o["sub"])}</div>'
        f'<span class="status {o["status"]}">{STATUS_TEXT[o["status"]]}</span>'
        f'<div class="m-sub" style="margin-top:.55rem">{esc(o["text"])}</div></div>'
        for o in options
    ], 220)

    section("🔎 Inspect a step")
    entries = ([(f"{p['icon']} {p['title']}", p) for p in phases]
               + [(f"{o['icon']} {o['title']}", o) for o in options])
    labels = [e[0] for e in entries]
    pick = st.selectbox("Pick a step to look inside", labels, key="map_pick")
    chosen = dict(entries).get(pick, entries[0][1])
    detail = chosen.get("detail") or chosen.get("text", "")
    checks = ""
    for rel in chosen["files"]:
        ok = (BASE / rel).exists()
        if "a CSV with" in rel:
            ok = False
        checks += f'<div class="check {"ok" if ok else "no"}">{"✓" if ok else "○"} {esc(rel)}</div>'
    html(f'<div class="glass"><div class="ctitle">{esc(pick)}</div>'
         f'<div class="csub">{esc(chosen["sub"])}</div>'
         f'<span class="status {chosen["status"]}">{STATUS_TEXT[chosen["status"]]}</span>'
         f'<div style="margin:.7rem 0">{esc(detail)}</div>'
         f'<div class="m-title">Files that show this step is done</div>{checks}</div>')
    st.button(f"Open {chosen['title']}" if chosen["page"] != P_HOME else "Open Home", key="map_go", on_click=goto, args=(chosen["page"],))

    if st.button("Celebrate progress", key="map_celebrate"):
        if all(p["status"] == DONE for p in phases):
            st.balloons()
            st.success("All four phases are complete. Nice work!")
        else:
            st.info("Finish the remaining phases first, then try this button again.")

    st.caption("Mutation Radar is partly built: this dashboard and run_all.py cover the project's own protein, "
               "but scoring a brand-new sequence is future work.")


def page_explorer():
    page_header("🧬 Mutation Forge", SUBTITLES[P_EXPLORE], tags=(phase_tag(2), option_tag(5)))
    df = get_scores()
    if df is None:
        warn_missing("results/step11_scores_with_ranks.csv")
        return

    wt_map = dict(zip(df["position"], df["wt_aa"]))
    ss = st.session_state
    if "mut_query" not in ss:
        ss["mut_query"] = "A53T"
    if "b_pos" not in ss:
        ss["b_pos"] = 53
    if "b_aa" not in ss:
        ss["b_aa"] = "T"

    def _sync(q):
        m = re.fullmatch(r"[A-Z](\d+)([A-Z])", q)
        if m:
            ss["b_pos"] = int(m.group(1))
            ss["b_aa"] = m.group(2)

    def _set(q):
        ss["mut_query"] = q
        _sync(q)

    def _build():
        pos, aa = int(ss["b_pos"]), ss["b_aa"]
        wt = wt_map.get(pos, "?")
        ss["mut_query"] = str(pos) if aa == wt else f"{wt}{pos}{aa}"

    def _surprise():
        r = df.sample(1).iloc[0]
        _set(str(r["mutation"]))

    section("Build a mutation")
    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        st.slider("Position in the protein", 1, 140, key="b_pos", on_change=_build)
    with c2:
        st.selectbox("Swap in", list(AA), key="b_aa", on_change=_build)
    with c3:
        st.write("")
        st.button("Surprise me", key="surprise", on_click=_surprise)

    st.caption("Or try one of the familial Parkinson's mutations:")
    cols = st.columns(len(FAMILIAL))
    for col, m in zip(cols, FAMILIAL):
        with col:
            st.button(m, key=f"quick_{m}", on_click=_set, args=(m,))

    query = st.text_input("Or type a mutation or position (for example A53T, 73 or V71)", key="mut_query")
    q = re.sub(r"\s+", "", str(query)).upper()
    if not q:
        st.info("Type a mutation such as A30P, E46K, H50Q, G51D or A53T.")
        return

    sel_pos = None
    exact = df[df["mutation"] == q]
    if re.fullmatch(r"[A-Z]\d+[A-Z]", q) and not exact.empty:
        r = exact.iloc[0]
        sel_pos = int(r["position"])
        pct = float(r["delta_rank_pct"])
        mut_score = r["mut_score"] if "mut_score" in r.index and pd.notna(r.get("mut_score")) else None
        sub = f"change vs wild type: {r['delta']:+.4f}"
        card_grid([
            card_html("Position", int(r["position"]), region_of(int(r["position"])), PURPLE),
            card_html("Aggregation score (CANYA)",
                      f"{mut_score:.3f}" if mut_score is not None else f"{r['delta']:+.4f}",
                      sub if mut_score is not None else "change vs wild type", PINK),
            card_html("Percentile", f"{pct:.1f}",
                      f"rank #{int(r['rank'])} of {len(df):,} (100 = largest increase)", BLUE),
            card_html("Predicted effect", effect_category(pct),
                      "relative to all mutations; not clinical risk", GREEN),
        ], 210)

        same_pos = df[df["position"] == r["position"]].sort_values("mut_aa")
        colors = [PINK if m == q else PURPLE for m in same_pos["mutation"]]
        fig = go.Figure(go.Bar(x=same_pos["mut_aa"], y=same_pos["delta"], marker_color=colors,
                               hovertext=same_pos["mutation"]))
        fig.add_hline(y=0, line_color="gray")
        fig.update_layout(title=f"All substitutions at position {int(r['position'])} ({r['wt_aa']})",
                          xaxis_title="Amino acid substituted in",
                          yaxis_title="Change in CANYA score")
        show_plot(fig, key="explorer_pos")
        table = same_pos
    elif re.fullmatch(r"\d+", q):
        sel_pos = int(q)
        table = df[df["position"] == int(q)].sort_values("mut_aa")
    elif re.fullmatch(r"[A-Z]\d+", q):
        sel_pos = int(q[1:])
        table = df[(df["position"] == int(q[1:])) & (df["wt_aa"] == q[0])].sort_values("mut_aa")
    else:
        table = pd.DataFrame()

    if table.empty:
        st.error(f"No mutation matches '{q}'. Positions run 1-140. Try A53T, E46K or 73.")
        return

    st.markdown("#### Where it sits in the protein")
    html(ribbon_html(df, highlight=sel_pos))
    st.caption("Each tile is a residue. Pink strength shows how much mutations there change the score, "
               "the purple underline marks the NAC region, and the dark outline is your position.")

    st.markdown("#### Results table")
    show_cols = [c for c in ["mutation", "position", "wt_aa", "mut_aa", "wt_score", "mut_score",
                             "delta", "delta_rank_pct", "rank"] if c in table.columns]
    out = table[show_cols].copy()
    out["effect"] = out["delta_rank_pct"].map(effect_category)
    show_df(out.round(4))


def game_section():
    section("🎮 Oracle Challenge")
    st.write("Two mutations appear. Pick the one you think raises predicted aggregation more.")
    df = get_scores()
    if df is None:
        warn_missing("results/step11_scores_with_ranks.csv")
        return
    game_init()
    ss = st.session_state
    if ss["g_pair"] is None:
        game_new_round()
    pair = ss["g_pair"]
    if pair is None:
        warn_missing("score table")
        return

    acc = f"{ss['g_score'] / ss['g_rounds'] * 100:.0f}%" if ss["g_rounds"] else "-"
    card_grid([
        card_html("Score", f"{ss['g_score']} of {ss['g_rounds']}", "correct guesses", PURPLE),
        card_html("Accuracy", acc, "your guesses vs CANYA", PINK),
        card_html("Streak", ss["g_streak"], "in a row", GREEN),
        card_html("Best streak", ss["g_best"], "this session", BLUE),
    ], 170)

    rows = [df[df["mutation"] == m].iloc[0] for m in pair]
    result = ss["g_result"]
    cards = []
    for k, r in enumerate(rows):
        wt, mt = str(r["wt_aa"]), str(r["mut_aa"])
        hyd = KD.get(mt, 0) - KD.get(wt, 0)
        body = (f'<div class="tag{" o" if k else ""}">Mutation {"AB"[k]}</div>'
                f'<div class="m-value" style="color:{[PURPLE, PINK][k]}">{esc(r["mutation"])}</div>'
                f'<div class="m-sub">{esc(region_of(int(r["position"])))}</div>'
                f'<div class="m-sub">Swaps {esc(wt)} for {esc(mt)}. Hydrophobicity change: {hyd:+.1f}</div>')
        if result is not None:
            won = result[1] == k
            body += (f'<div class="pill" style="background:{GREEN if won else "#94a3b8"}">'
                     f'{"Higher effect" if won else "Lower effect"}</div>'
                     f'<div class="m-sub" style="margin-top:.4rem">Change in CANYA score: {r["delta"]:+.4f}, '
                     f'{r["delta_rank_pct"]:.0f}th percentile</div>')
        cards.append(f'<div class="glass" style="border-top:5px solid {[PURPLE, PINK][k]}">{body}</div>')
    card_grid(cards, 260)

    if result is None:
        st.write("Which mutation does CANYA predict raises aggregation more?")
        c1, c2 = st.columns(2)
        with c1:
            st.button(f"A: {pair[0]}", key="guess_a", on_click=game_guess, args=(0,))
        with c2:
            st.button(f"B: {pair[1]}", key="guess_b", on_click=game_guess, args=(1,))
        st.caption("Hint: aggregation-prone stretches like greasy (hydrophobic) residues, and prolines tend to break them up.")
    else:
        if result[2]:
            st.success("Right! You and the Oracle agree." + (" Hot streak!" if ss["g_streak"] >= 3 else ""))
        else:
            st.info("Not this time. The Oracle ranks the other mutation higher.")
        st.button("Next round", key="game_next", on_click=game_new_round)

    st.button("Reset score", key="game_reset", on_click=game_reset)
    st.caption("This is a game about CANYA's predictions, not about real laboratory aggregation.")


def page_hotspots():
    page_header("🔥 Hotspot Hunt", SUBTITLES[P_HOT], tags=(option_tag(1),))
    df = get_scores()

    if df is not None:
        section("Hover over the protein")
        lo, hi = st.slider("Zoom into a region", 1, 140, (61, 82))
        html(ribbon_html(df, window=(lo, hi)))
        st.caption("Each tile is one residue, pink strength is the mean size of the change when that residue "
                   "is mutated, and the purple underline is the NAC region. Hover for the exact value.")
        win = df[df["position"].between(lo, hi)]
        if len(win):
            up = win.loc[win["delta"].idxmax()]
            down = win.loc[win["delta"].idxmin()]
            card_grid([
                card_html(f"Mean |change|, residues {lo}-{hi}", f"{win['delta'].abs().mean():.4f}",
                          f"{len(win):,} mutations in this window", PURPLE),
                card_html("Biggest predicted increase", str(up["mutation"]), f"{up['delta']:+.4f}", PINK),
                card_html("Biggest predicted decrease", str(down["mutation"]), f"{down['delta']:+.4f}", BLUE),
            ], 220)

    ok = show_image("step19_position_sensitivity.png", "Mean |change in score| by position (shaded: NAC region)")
    if df is None:
        if not ok:
            st.info("Interactive charts need results/step11_scores_with_ranks.csv.")
        return

    pos = df.groupby("position")["delta"].agg(mean_abs=lambda s: s.abs().mean(), mean_delta="mean").reset_index()
    fig = go.Figure(go.Bar(x=pos["position"], y=pos["mean_abs"], marker_color=PURPLE))
    fig.add_vrect(x0=61, x1=95, fillcolor=PINK, opacity=0.12, line_width=0, annotation_text="NAC region (61-95)", annotation_position="top left")
    fig.update_layout(title="Interactive: mutational sensitivity by position",
                      xaxis_title="Position", yaxis_title="Mean |change in CANYA score|")
    show_plot(fig, key="hot_bar")

    top = pos.nlargest(10, "mean_abs").round(4)
    inside = pos[pos["position"].between(61, 95)]["mean_abs"].mean()
    outside = pos[~pos["position"].between(61, 95)]["mean_abs"].mean()
    card_grid([
        card_html("Mean |delta| inside NAC (61-95)", f"{inside:.4f}", "", PINK),
        card_html("Mean |delta| outside NAC", f"{outside:.4f}", "", BLUE),
        card_html("Ratio", f"{inside / outside:.1f}x" if outside else "n/a", "NAC vs rest", GREEN),
    ], 220)

    piv = df.pivot_table(index="mut_aa", columns="position", values="delta").reindex(list(AA))
    lim = float(np.nanmax(np.abs(piv.values)))
    heat = go.Figure(go.Heatmap(z=piv.values, x=piv.columns, y=piv.index, zmin=-lim, zmax=lim,
                                colorscale=heat_colorscale(), xgap=1, ygap=1, hoverongaps=False,
                                colorbar=dict(title="Change in<br>CANYA score")))
    heat.update_layout(title="Every mutation: pink = predicted increase, blue = decrease",
                       xaxis_title="Position", yaxis_title="Amino acid substituted in", height=520)
    show_plot(heat, key="hot_heat")

    st.markdown("#### Ten most sensitive positions")
    show_df(top)


def page_xai():
    page_header("🔮 Ask the Oracle", SUBTITLES[P_XAI], tags=("SHAP explainability", "Random Forest"))
    show_image("step17_shap_summary.png", "SHAP summary: each dot is one mutation")

    st.markdown("### How to read the SHAP plot")
    st.write("Each dot is one mutation. Dots to the right push the predicted score change up, "
             "dots to the left push it down. Red means a high value of that feature, blue a low value.")
    st.markdown("### SHAP analysis")
    st.write("In this project the strongest driver was the hydrophobicity of the amino acid "
             "introduced, followed by whether the position lies in the NAC region. "
             "Charge and size changes mattered less.")
    st.caption("SHAP explains the random forest, which imitates CANYA. It describes what drives "
               "CANYA's predictions, not what drives aggregation in cells. Hydrophobicity features "
               "overlap, so credit is shared among them.")

    df = get_scores()
    if df is None:
        warn_missing("results/step11_scores_with_ranks.csv")
        return

    st.markdown("### A direct look at the data")
    nac = df[df["position"].between(61, 82)]
    g = nac.groupby("mut_aa")["delta"].mean().reindex(list(AA)).reset_index()
    g["hydrophobicity"] = g["mut_aa"].map(KD)
    fig = px.scatter(g, x="hydrophobicity", y="delta", text="mut_aa", color_discrete_sequence=[PURPLE])
    fig.update_traces(textposition="top center", marker=dict(size=11))
    fig.add_hline(y=0, line_color="gray")
    fig.update_layout(title="Mean change in score by introduced amino acid, residues 61-82",
                      xaxis_title="Kyte-Doolittle hydrophobicity of introduced residue",
                      yaxis_title="Mean change in CANYA score")
    show_plot(fig, key="xai_scatter")
    try:
        rho = g["hydrophobicity"].corr(g["delta"], method="spearman")
        st.caption(f"Spearman correlation across the 20 amino acids: {rho:.2f}")
    except Exception:
        pass

    pred = read_csv_safe(RESULTS / "step16_predictions.csv")
    if pred is not None and {"delta", "predicted_delta"}.issubset(pred.columns):
        st.markdown("### Random forest: predicted vs actual (held-out blocks)")
        fig2 = px.scatter(pred, x="delta", y="predicted_delta", opacity=0.5,
                          color_discrete_sequence=[PINK], hover_name="mutation" if "mutation" in pred else None)
        fig2.update_layout(title="Random forest: predicted vs CANYA change in score", xaxis_title="CANYA change in score (actual)", yaxis_title="Model-predicted change in score")
        show_plot(fig2, key="xai_pred")
        st.caption("Ranking is moderately captured (Spearman about 0.45 in this project) "
                   "but magnitudes are not (R-squared about 0).")

    game_section()


def page_parkinson():
    page_header("🏆 Parkinson Legends", SUBTITLES[P_PD], tags=("Familial alpha-synuclein mutations",))
    df = get_scores()
    accents = [PURPLE, PINK, BLUE, GREEN, PURPLE, PINK]
    pcards = []
    for m in FAMILIAL:
        accent = accents[FAMILIAL.index(m)]
        hit = df[df["mutation"] == m] if df is not None else pd.DataFrame()
        if hit.empty:
            pcards.append(
                f'<div class="glass" style="border-top:5px dashed {accent}">'
                f'<div class="m-title">{esc(m)}</div>'
                f'<div class="m-value" style="color:{accent}">No data</div>'
                f'<div class="m-sub">{esc(MISSING)}</div></div>')
            continue
        r = hit.iloc[0]
        score = (f"mutant score {r['mut_score']:.3f}, "
                 if "mut_score" in r.index and pd.notna(r.get("mut_score")) else "")
        pcards.append(
            f'<div class="glass" style="border-top:5px solid {accent}">'
            f'<div class="m-title">{esc(m)}</div>'
            f'<div class="m-value" style="color:{accent}">{r["delta_rank_pct"]:.1f}th percentile</div>'
            f'<div class="m-sub">Rank {int(r["rank"])} of {len(df):,}; {esc(score)}change {r["delta"]:+.4f}</div>'
            f'<div class="pill" style="background:{accent}">{esc(effect_category(float(r["delta_rank_pct"])))}</div>'
            f'<div class="m-sub" style="margin-top:.5rem">Literature: {esc(LITERATURE[m])}</div></div>')
    card_grid(pcards, 290)

    if df is not None:
        sub = df[df["mutation"].isin(FAMILIAL)].set_index("mutation").reindex(
            [m for m in FAMILIAL if m in set(df["mutation"])]).reset_index()
        fig = go.Figure(go.Bar(y=sub["mutation"], x=sub["delta_rank_pct"], orientation="h",
                               marker_color=PINK))
        fig.add_vline(x=50, line_dash="dash", line_color="black")
        fig.update_layout(title="Rank percentile of each mutation (100 = largest predicted increase)",
                          xaxis=dict(range=[0, 100], title="Percentile"), yaxis=dict(autorange="reversed"))
        show_plot(fig, key="pd_bar")

    if df is not None and df["mutation"].isin(FAMILIAL).any():
        pcts = df[df["mutation"].isin(FAMILIAL)]["delta_rank_pct"]
        if pcts.max() <= 75:
            st.info("Finding: CANYA does not single out these mutations, and "
                    f"{int((pcts < 50).sum())} of {len(pcts)} rank below the median. "
                    "CANYA scores nucleation of 20-residue peptides, while disease mutations may act "
                    "through the full-length protein, oligomers or membranes. This is a limit of the "
                    "method, not evidence against the mutations.")
        else:
            st.info("At least one of these mutations ranks in the upper quarter of predicted effects. "
                    "Treat this cautiously: the set is small and CANYA scores short peptides only.")


def page_cross():
    page_header("🌌 Neuro Galaxy", SUBTITLES[P_CROSS],
                tags=("Alpha-synuclein", "Tau", "Amyloid-beta", "Huntingtin", "TDP-43"))
    proteins = ["Alpha-Synuclein", "Tau", "Amyloid-Beta", "Huntingtin", "TDP-43"]
    cross = find_cross_file()
    cdf = read_csv_safe(cross) if cross else None

    if cdf is not None and len(cdf) and cdf.shape[1] >= 2:
        st.success(f"Loaded {cross.name}")
        name_col = next((c for c in cdf.columns if "protein" in c.lower()), cdf.columns[0])
        nums = [c for c in cdf.columns if c != name_col and pd.api.types.is_numeric_dtype(cdf[c])]
        show_df(cdf)
        if nums:
            fig = px.bar(cdf, x=name_col, y=nums[0], color_discrete_sequence=[PURPLE])
            fig.update_layout(title=f"{nums[0]} by protein", xaxis_title="Protein", yaxis_title=nums[0])
            show_plot(fig, key="cross_bar")
            if len(nums) >= 2:
                heat = go.Figure(go.Heatmap(z=cdf[nums].values, x=nums, y=cdf[name_col],
                                            colorscale=heat_colorscale()))
                heat.update_layout(title="Protein comparison heatmap", xaxis_title="Metric", yaxis_title="Protein")
                show_plot(heat, key="cross_heat")
        return

    df = get_scores()
    st.markdown("### Protein status")
    status = []
    for p, accent in zip(proteins, [PURPLE, PINK, BLUE, GREEN, PURPLE]):
        if p == "Alpha-Synuclein" and df is not None:
            m = df[df["position"].between(61, 95)]["delta"].abs().mean()
            status.append(card_html(p, "Analyzed", f"{len(df):,} variants; NAC mean |delta| {m:.3f}", accent))
        else:
            status.append(card_html(p, "Not yet run", "no results file found", accent))
    card_grid(status, 190)

    st.warning("No cross-protein results file was found in results/ (a CSV with 'cross' or 'proteins' "
               "in its name). The charts below show only the intended layout.")
    demo_banner("Values below are made up to show the layout. Do not cite them.")

    rng = np.random.default_rng(7)
    demo = pd.DataFrame({"protein": proteins, "demo metric": np.round(rng.uniform(0.2, 0.9, 5), 2)})
    fig = px.bar(demo, x="protein", y="demo metric", color_discrete_sequence=[PURPLE])
    fig.update_layout(title="Demo: protein comparison (not real data)", xaxis_title="Protein", yaxis_title="Demo metric")
    show_plot(fig, key="cross_demo_bar")
    regions = ["N-term", "Region 2", "Region 3", "Region 4", "C-term"]
    z = np.round(rng.uniform(0, 1, (5, 5)), 2)
    heat = go.Figure(go.Heatmap(z=z, x=regions, y=proteins, colorscale=heat_colorscale()))
    heat.update_layout(title="Demo: region sensitivity heatmap (not real data)", xaxis_title="Protein region", yaxis_title="Protein")
    show_plot(heat, key="cross_demo_heat")
    st.caption("To fill this page, save a CSV such as results/cross_disease_summary.csv with one row "
               "per protein and numeric columns.")


def page_stats():
    page_header("📊 Truth Check", SUBTITLES[P_STATS], tags=("Permutation test", "Bootstrap CI"))
    df = get_scores()
    if df is None:
        warn_missing("results/step11_scores_with_ranks.csv")
        demo_banner("Example layout only.")
        show_df(pd.DataFrame({"test": ["Permutation test", "Bootstrap 95% CI"],
                              "result": ["(run the pipeline)", "(run the pipeline)"]}))
        return

    present = [m for m in FAMILIAL if m in set(df["mutation"])]
    default = [m for m in VALIDATION_SET if m in present]
    chosen = st.multiselect("Mutations treated as the known set", present, default=default)
    if len(chosen) < 2:
        st.info("Pick at least two mutations.")
        return

    all_vals = df["delta"].to_numpy(dtype=float)
    known_vals = df[df["mutation"].isin(chosen)]["delta"].to_numpy(dtype=float)
    obs, p_high, p_low, means = permutation_test(all_vals, known_vals)
    lo, hi = bootstrap_ci(known_vals, n=10000)
    glo, ghi = bootstrap_ci(all_vals, n=2000)

    card_grid([
        card_html("Mean change, known set", f"{obs:+.4f}", f"n = {len(chosen)}", PURPLE),
        card_html("Mean change, all mutations", f"{all_vals.mean():+.4f}", f"n = {len(all_vals):,}", BLUE),
        card_html("P(random set ≥ known)", f"{p_high:.3f}", "10,000 permutations", PINK),
        card_html("P(random set ≤ known)", f"{p_low:.3f}", "10,000 permutations", GREEN),
    ], 210)

    fig = go.Figure(go.Histogram(x=means, nbinsx=60, marker_color=PURPLE, opacity=0.8))
    fig.add_vline(x=obs, line_color=PINK, line_width=3, annotation_text="known set", annotation_position="top left")
    fig.update_layout(title="Permutation test: mean change in random sets of the same size",
                      xaxis_title="Mean change in CANYA score", yaxis_title="Count")
    show_plot(fig, key="stats_hist")

    st.markdown("#### Bootstrap 95% confidence intervals for the mean change")
    show_df(pd.DataFrame({
        "group": ["Known set", "All mutations"],
        "n": [len(known_vals), len(all_vals)],
        "mean": [round(obs, 4), round(float(all_vals.mean()), 4)],
        "95% CI low": [round(lo, 4), round(glo, 4)],
        "95% CI high": [round(hi, 4), round(ghi, 4)],
    }))

    rob = read_csv_safe(RESULTS / "step14_robustness.csv")
    if rob is not None and "mutation" in rob.columns:
        st.markdown("#### Robustness across ways of summarising windows")
        rows = []
        for col in ["delta", "delta_mean", "delta_peak"]:
            if col in rob.columns:
                a = rob[col].dropna().to_numpy(dtype=float)
                k = rob[rob["mutation"].isin(chosen)][col].dropna().to_numpy(dtype=float)
                if len(k) >= 2:
                    o, ph, pl, _ = permutation_test(a, k)
                    rows.append({"summary": col, "mean known": round(o, 4),
                                 "P(random ≥ known)": round(ph, 3), "P(random ≤ known)": round(pl, 3)})
        if rows:
            show_df(pd.DataFrame(rows))

    st.caption("Small sample: with only a handful of known mutations these tests have little power. "
               "We tried three window summaries, so a single low p-value would need cautious reading. "
               "The known set scoring slightly lower, not higher, is exploratory.")


def page_downloads():
    page_header("📦 Data Vault", SUBTITLES[P_DL], tags=(phase_tag(4),))
    files = results_files()
    if not files:
        warn_missing("results/ is empty or missing")
        return
    mimes = {".csv": "text/csv", ".png": "image/png", ".txt": "text/plain"}
    tabs = st.tabs(["CSV files", "PNG files", "TXT files"])
    for tab, ext in zip(tabs, [".csv", ".png", ".txt"]):
        with tab:
            group = [p for p in files if p.suffix.lower() == ext]
            if not group:
                st.info(f"No {ext} files found in results/.")
                continue
            for p in group:
                rel = p.relative_to(RESULTS).as_posix()
                try:
                    data = p.read_bytes()
                except Exception:
                    continue
                c1, c2 = st.columns([4, 1])
                with c1:
                    st.write(f"**{rel}**  ({len(data) / 1024:,.0f} KB)")
                with c2:
                    st.download_button("Download", data=data, file_name=p.name,
                                       mime=mimes[ext], key=f"dl_{rel}")


def page_about():
    page_header("ℹ️ About", "Research question, methods and honest limits.", tags=("Methods", "Limitations"))
    st.markdown("### Research question")
    st.write("Can a model of amyloid nucleation (CANYA) show which positions and which kinds of "
             "substitutions in alpha-synuclein change aggregation propensity, and does it flag "
             "mutations linked to familial Parkinson's disease?")
    st.markdown("### Methods")
    st.write("Sequence: UniProt P37840 (140 residues). All 2,660 single substitutions were generated. "
             "CANYA, trained on 20-residue peptides, scored sliding 20-residue windows covering each "
             "mutated residue; the per-mutation score is the highest window, and the effect is mutant "
             "minus wild type. Known mutations were ranked and tested by permutation. A random forest "
             "with held-out protein blocks and SHAP examined which properties explain the predictions.")
    st.markdown("### Pipeline workflow")
    st.code("Meet the Protein   get the sequence\n"
            "Mutation Forge     generate 2,660 mutations\n"
            "Oracle Scoring     score 20-residue windows with CANYA\n"
            "Dataset Assembly   build the dataset\n"
            "Hotspot Hunt       landscape mapping\n"
            "Ask the Oracle     explainable machine learning\n"
            "Neuro Galaxy       cross-disease comparison\n"
            "Truth Check        validation study\n"
            "Mutation Radar     scanner tool",
            language="text")
    st.markdown("### Key findings")
    st.write("Predicted sensitivity concentrates in residues about 61-82 (NAC). Prolines lower and "
             "hydrophobic substitutions raise the predicted score there. Amino-acid properties rank "
             "effects moderately but do not predict their size. The familial mutations were not "
             "flagged.")
    st.markdown("### Limitations")
    st.write("Scores are predictions, not measurements. CANYA scores short peptides, not the whole "
             "protein, oligomers or cells. The validation set is small. This dashboard reports "
             "predicted aggregation propensity, not clinical disease risk, and is not medical advice.")
    st.markdown("### Future work")
    st.write("Run tau, amyloid-beta, huntingtin and TDP-43 through the same pipeline; compare with "
             "experimental mutational scans; test other aggregation predictors; score new sequences "
             "directly.")
    st.markdown("### Applications")
    st.write("Prioritising mutations for lab testing, teaching how sequence shapes aggregation, and "
             "benchmarking aggregation predictors.")
    st.markdown("### Credits")
    st.write("CANYA: Thompson et al., Science Advances 11(18):eadt5111 (2025). Sequence: UniProt P37840. "
             "Built by Sana Manchikanti.")


PAGES = {
    P_HOME: page_home,
    P_GLANCE: page_glance,
    P_EXPLORE: page_explorer,
    P_HOT: page_hotspots,
    P_XAI: page_xai,
    P_PD: page_parkinson,
    P_CROSS: page_cross,
    P_STATS: page_stats,
    P_DL: page_downloads,
    P_ABOUT: page_about,
}


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def sidebar():
    html('<div class="brand"><div class="avatar">🧬</div><div>'
         '<div class="t1">Alpha-Synuclein Explorer</div>'
         '<div class="t2">Neuro-AI lab · Sana Manchikanti</div></div></div>')
    current = st.session_state.get("nav", P_HOME)
    html(f'<div class="bubble">{esc(MASCOT.get(current, MASCOT[P_HOME]))}</div>')

    st.radio("Navigate", NAV, key="nav", label_visibility="collapsed")

    try:
        phases, options = phase_info(), option_info()
        score, total = progress_value()
        steps = "".join(
            f'<div class="step"><span class="dot {p["status"]}"></span>{p["icon"]} {esc(p["title"])}</div>'
            for p in phases)
        opts = "".join(
            f'<div class="opt"><span class="dot {o["status"]}"></span>{o["icon"]} {esc(o["title"])}</div>'
            for o in options)
        html(f'<div class="sidebox"><div class="h">Research journey</div><div class="steps">{steps}</div></div>')
        html(f'<div class="sidebox"><div class="h">Discovery tracks</div>{opts}</div>')
        pct = score / total * 100 if total else 0
        html(f'<div class="sidebox"><div class="h">Mission progress</div>'
             f'<div class="pbar"><span style="width:{pct:.0f}%"></span></div>'
             f'<div class="m-sub" style="margin-top:.4rem">{score:g} of {total} steps</div></div>')
    except Exception:
        pass

    n_files = len(results_files())
    if n_files:
        html(f'<div class="live"><span class="dot done"></span>{n_files} output files found</div>')
    else:
        html('<div class="live"><span class="dot"></span>No outputs yet. Run the pipeline.</div>')


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    if "nav" not in st.session_state or st.session_state["nav"] not in NAV:
        st.session_state["nav"] = P_HOME
    html(CSS)
    with st.sidebar:
        sidebar()
    choice = st.session_state.get("nav", P_HOME)

    try:
        PAGES.get(choice, page_home)()
    except Exception as exc:  # the app must never crash
        st.error("Something went wrong while building this page.")
        st.caption(f"{type(exc).__name__}: {exc}")

    html('<div class="footer">Made with 💜 using predictions from the CANYA model. Not medical advice.</div>')


main()