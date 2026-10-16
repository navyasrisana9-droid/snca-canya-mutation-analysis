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

What's new in this version
- Project at a Glance page (one screen for posters, fairs and judges)
- Mutation Explorer now has three tabs: single mutation (with sequence context),
  side-by-side comparison, and a filterable browser with CSV export
- Hotspot Mapping split into tabs: residue strip, position profile, heatmap,
  substitution types (amino-acid classes + wild-type x mutant matrix),
  regions and leaderboards
- Protein architecture diagram showing N-terminal / NAC / C-terminal regions
  and the familial mutation sites
- Oracle Challenge has Easy / Medium / Hard difficulty and a result chart
- Statistical Validation adds an effect size and a plain-language reading
- Parkinson Mutations shows where each known mutation falls in the full distribution
- Explainable AI adds a feature-vs-effect correlation chart (if step15_features.csv exists)
- Downloads adds a one-click ZIP and image previews
- Sidebar "Jump to a mutation" search box
- Faster: the score table is loaded and processed once, then cached
"""

import html as _html
import io
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Alpha-Synuclein Explorer", page_icon="🧬", layout="wide",
                   initial_sidebar_state="expanded")

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

# Amino-acid chemistry classes used for grouping
AA_CLASSES = {
    "Hydrophobic": "AVILMC",
    "Aromatic": "FWY",
    "Polar": "STNQ",
    "Positive": "KRH",
    "Negative": "DE",
    "Special (G, P)": "GP",
}
AA_TO_CLASS = {a: c for c, letters in AA_CLASSES.items() for a in letters}

REGION_ORDER = ["N-terminal (1-60)", "NAC (61-95)", "C-terminal (96-140)"]
EFFECT_ORDER = ["Strong increase (top 10%)", "Moderate increase (top 10-25%)", "Near typical",
                "Moderate decrease (bottom 10-25%)", "Strong decrease (bottom 10%)"]

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

# Names for the four phases and five options of the research plan
PHASE_TITLES = {1: "Meet the Protein", 2: "Mutation Forge", 3: "Ask the Oracle", 4: "Data Vault"}
PHASE_SUBS = {1: "Get alpha-synuclein", 2: "Create every single mutation",
              3: "Score them with CANYA", 4: "Build the dataset"}
OPTION_TITLES = {1: "Hotspot Atlas", 2: "Decode the Oracle", 3: "Protein Crossover",
                 4: "Reality Check", 5: "Mutation Radar"}
OPTION_SUBS = {1: "Mutation landscape mapping", 2: "Explainable machine learning",
               3: "Cross-disease comparison", 4: "Validation study", 5: "Mutation scanner tool"}


def phase_tag(n):
    return f"Phase {n}: {PHASE_TITLES[n]}"


def option_tag(n):
    return f"Option {n}: {OPTION_TITLES[n]}"


P_HOME = "🏠 Home"
P_SUM = "📌 At a Glance"
P_MAP = "🗺 Mission Map"
P_EXPLORE = "🔬 Mutation Explorer"
P_RADAR = "📡 Mutation Radar"
P_GAME = "🎮 Oracle Challenge"
P_HOT = "🔥 Hotspot Mapping"
P_XAI = "🤖 Explainable AI"
P_PD = "🧠 Parkinson Mutations"
P_CROSS = "🧬 Cross-Disease Analysis"
P_STATS = "📊 Statistical Validation"
P_DL = "⬇ Downloads"
P_ABOUT = "ℹ About"
NAV = [P_HOME, P_SUM, P_MAP, P_EXPLORE, P_RADAR, P_GAME, P_HOT, P_XAI, P_PD, P_CROSS, P_STATS, P_DL, P_ABOUT]

MASCOT = {
    P_HOME: "Hi, I'm Helix! Pick a page and let's explore.",
    P_SUM: "The whole study on one screen.",
    P_MAP: "Choose a step and I'll show what it made.",
    P_EXPLORE: "Try Build a mutation, compare two, or press Surprise me!",
    P_RADAR: "Scan any mutation set or a brand-new sequence.",
    P_GAME: "Think you can out-guess the Oracle?",
    P_HOT: "Hover over the tiles to peek at each residue.",
    P_XAI: "Pink pushes the score up, blue pushes it down.",
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
.stApp h4 { font-size: 1rem; font-weight: 700; margin-top: 1.1rem; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stHeader"] * { color: #2b1b4e; }
.block-container { padding-top: 2.2rem; max-width: 1280px; }

:focus-visible { outline: 3px solid #ec4899 !important; outline-offset: 2px; }

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

/* ---------- sequence context line ---------- */
.seqline { display: flex; flex-wrap: wrap; align-items: center; gap: 3px; padding: .7rem .9rem;
  background: rgba(255,255,255,.7); border-radius: 18px; border: 1px solid rgba(139,92,246,.18);
  font-family: ui-monospace, 'SFMono-Regular', Menlo, Consolas, monospace; font-size: 1.05rem; }
.seqline .r { padding: .1rem .3rem; border-radius: 6px; font-weight: 700; }
.seqline .r.mut { background: linear-gradient(90deg, #8b5cf6, #ec4899); color: #fff; box-shadow: 0 4px 12px rgba(236,72,153,.35); }
.seqline .sn { font-size: .72rem; opacity: .55; padding: 0 .4rem; }

/* ---------- page header ---------- */
.pagehead { margin: .2rem 0 1.3rem 0; }
.pagehead h2 { font-size: clamp(1.4rem, 2.6vw, 2rem); font-weight: 700; margin: 0;
  background: linear-gradient(90deg, #7c3aed, #ec4899); -webkit-background-clip: text; background-clip: text; color: transparent; }
.pagehead p { margin: .35rem 0 .5rem 0; opacity: .8; font-size: 1.02rem; max-width: 44rem; }

.banner-demo { background: rgba(250,204,21,.25); border: 2px dashed #ca8a04; border-radius: 18px;
  padding: .8rem 1.1rem; margin: .5rem 0 1rem 0; font-weight: 700; }
.callout { background: rgba(255,255,255,.72); border-left: 6px solid #8b5cf6; border-radius: 18px;
  padding: .85rem 1.1rem; margin: .6rem 0 1rem 0; font-size: .95rem; line-height: 1.5; }
.callout.warn { border-left-color: #ca8a04; background: rgba(250,204,21,.16); }
.callout.good { border-left-color: #22c55e; }
.flow { display: flex; flex-wrap: wrap; align-items: center; gap: .5rem; margin: .4rem 0 1.2rem 0; }
.flow .node { padding: .35rem .9rem; border-radius: 999px; background: rgba(255,255,255,.75);
  border: 1px solid rgba(139,92,246,.25); font-weight: 700; font-size: .88rem; }
.flow .arrow { color: #ec4899; font-weight: 800; }
.footer { text-align: center; opacity: .65; font-size: .82rem; margin: 2.4rem 0 .6rem 0; }

/* ---------- widgets ---------- */
.stButton > button, .stDownloadButton > button { border-radius: 999px; font-weight: 700; color: #2b1b4e;
  background: rgba(255,255,255,.8); border: 1px solid rgba(139,92,246,.3); transition: all .18s ease; }
.stButton > button:hover, .stDownloadButton > button:hover { border-color: #ec4899; color: #be185d;
  box-shadow: 0 6px 16px rgba(236,72,153,.2); }
[data-baseweb="input"] { border-radius: 16px !important; }
.stTabs [data-baseweb="tab-list"] { gap: .45rem; flex-wrap: wrap; }
.stTabs [data-baseweb="tab"] { font-weight: 700; border-radius: 999px; padding: .35rem 1.05rem; height: auto;
  background: rgba(255,255,255,.55); border: 1px solid rgba(139,92,246,.14); }
.stTabs [aria-selected="true"] { background: linear-gradient(90deg, #8b5cf6, #ec4899); border-color: transparent; }
.stTabs [aria-selected="true"] p, .stTabs [aria-selected="true"] { color: #ffffff !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
[data-testid="stExpander"] { border-radius: 18px; background: rgba(255,255,255,.55); }

/* inputs stay readable even when Streamlit is in dark mode */
.stApp textarea, .stApp input, .stApp [data-baseweb="select"] input {
  background: #ffffff !important; color: #2b1b4e !important; -webkit-text-fill-color: #2b1b4e !important; }
.stApp textarea::placeholder, .stApp input::placeholder { color: #8b7fb0 !important; -webkit-text-fill-color: #8b7fb0 !important; }
.stApp [data-baseweb="textarea"], .stApp [data-baseweb="input"], .stApp [data-baseweb="base-input"] {
  background: #ffffff !important; border-radius: 16px !important; border-color: rgba(139,92,246,.3) !important; }
.stApp [data-baseweb="select"] > div { background: #ffffff !important; border-radius: 16px !important; }
.stApp [data-baseweb="select"] *, .stApp [data-baseweb="tag"] * { color: #2b1b4e !important; }
[data-baseweb="popover"] [data-baseweb="menu"], [data-baseweb="popover"] ul { background: #ffffff !important; }
[data-baseweb="popover"] li, [data-baseweb="popover"] li * { color: #2b1b4e !important; }
[data-testid="stFileUploaderDropzone"] { background: rgba(255,255,255,.85) !important; border-radius: 18px;
  border: 1.5px dashed rgba(139,92,246,.45); }
[data-testid="stFileUploaderDropzone"] *, [data-testid="stFileUploader"] small { color: #2b1b4e !important; }
.stTabs [data-baseweb="tab"] { padding: .45rem 1.1rem !important; border-radius: 999px !important;
  height: auto !important; min-height: 0 !important; margin-right: .1rem; }
.stTabs [data-baseweb="tab-list"] { gap: .5rem !important; padding: .2rem 0 .6rem 0; }

@media (max-width: 640px) {
  .hero { padding: 1.5rem 1.2rem; border-radius: 24px; }
  .helix { display: none; }
  .tile { width: 1.25rem; height: 1.5rem; font-size: .72rem; }
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


def callout(text, kind=""):
    html(f'<div class="callout {kind}">{esc(text)}</div>')


def section(title):
    st.markdown(f"### {title}")


def page_header(title, subtitle="", tags=()):
    chips = "".join(f'<span class="tag{" o" if t.lower().startswith("option") else ""}" '
                    f'style="margin-right:.35rem">{esc(t)}</span>' for t in tags)
    html(f'<div class="pagehead"><h2>{esc(title)}</h2><p>{esc(subtitle)}</p>{chips}</div>')


def warn_missing(extra=""):
    st.warning(MISSING + (f" ({extra})" if extra else ""))


def demo_banner(text):
    html(f'<div class="banner-demo">🧪 Demo data, not results from this study. {esc(text)}</div>')


def show_plot(fig, key=None):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.55)",
        font=dict(color=INK, family="Nunito, sans-serif"), margin=dict(l=40, r=20, t=60, b=40),
    )
    fig.update_xaxes(tickfont=dict(color=INK), title_font=dict(color=INK), gridcolor="rgba(139,92,246,.12)")
    fig.update_yaxes(tickfont=dict(color=INK), title_font=dict(color=INK), gridcolor="rgba(139,92,246,.12)")
    fig.update_layout(title_font=dict(color=INK, size=16))
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


def norm_query(s):
    return re.sub(r"\s+", "", str(s)).upper()


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


def region_group(pos):
    if pos <= 60:
        return REGION_ORDER[0]
    if pos <= 95:
        return REGION_ORDER[1]
    return REGION_ORDER[2]


@st.cache_data(show_spinner=False)
def _load_scores(path_str, mtime):
    """Read one score file and add every derived column once (cached)."""
    df = pd.read_csv(path_str)
    if not {"mutation", "position", "delta"}.issubset(df.columns):
        return None
    df = df.dropna(subset=["mutation", "position", "delta"]).copy()
    if df.empty:
        return None
    df["position"] = df["position"].astype(int)
    df["mutation"] = df["mutation"].astype(str)
    if "wt_aa" not in df.columns:
        df["wt_aa"] = df["mutation"].str[0]
    if "mut_aa" not in df.columns:
        df["mut_aa"] = df["mutation"].str[-1]
    df["delta_rank_pct"] = df["delta"].rank(pct=True) * 100
    df["rank"] = df["delta"].rank(ascending=False, method="min").astype(int)
    df["region"] = df["position"].map(region_group)
    df["aa_class"] = df["mut_aa"].map(AA_TO_CLASS).fillna("Other")
    df["effect"] = df["delta_rank_pct"].map(effect_category)
    return df


def get_scores():
    """Per-mutation table from the pipeline (None if no usable file)."""
    for name in SCORE_FILES:
        p = RESULTS / name
        try:
            if p.exists():
                df = _load_scores(str(p), p.stat().st_mtime)
                if df is not None:
                    return df
        except Exception:
            continue
    return None


def get_sequence(df=None):
    """Wild-type sequence from the FASTA, or rebuilt from the score table."""
    try:
        if FASTA.exists():
            seq = "".join(ln.strip() for ln in FASTA.read_text().splitlines() if not ln.startswith(">"))
            if seq:
                return seq
    except Exception:
        pass
    if df is not None and len(df):
        m = dict(zip(df["position"], df["wt_aa"]))
        return "".join(m.get(i, "-") for i in range(1, max(m) + 1))
    return ""


def context_html(seq, pos, mut_aa=None, flank=12):
    """A short stretch of the sequence around one residue, with the change highlighted."""
    if not seq or pos < 1 or pos > len(seq):
        return ""
    lo, hi = max(1, pos - flank), min(len(seq), pos + flank)
    parts = []
    for i in range(lo, hi + 1):
        ch = seq[i - 1]
        if i == pos:
            txt = f"{ch}→{mut_aa}" if mut_aa and mut_aa != ch else ch
            parts.append(f'<span class="r mut" title="position {i}">{esc(txt)}</span>')
        else:
            parts.append(f'<span class="r">{esc(ch)}</span>')
    return f'<div class="seqline"><span class="sn">{lo}</span>{"".join(parts)}<span class="sn">{hi}</span></div>'


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
                  if p.is_file() and p.suffix.lower() in {".csv", ".png", ".txt", ".tsv"})


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
         f"{n_pep:,} peptide windows scored" if n_pep else "results/step6_canya_output.tsv not found", P_GAME,
         ["results/step6_canya_input.tsv", "results/step6_canya_output.tsv", "results/step8_scored_windows.csv"]),
        (DONE if n_ds else TODO,
         f"{n_ds:,} mutations with a score and a change vs wild type" if n_ds else "no score table found", P_DL,
         ["results/step9_mutation_scores.csv", "results/step11_scores_with_ranks.csv"]),
    ]
    return [dict(n=i + 1, title=PHASE_TITLES[i + 1], sub=PHASE_SUBS[i + 1], status=s, detail=d, page=pg, files=f)
            for i, (s, d, pg, f) in enumerate(items)]


def option_info():
    df = get_scores()
    cross = find_cross_file()

    # Option 1: landscape mapping
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

    # Option 2: explainable ML
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

    # Option 3: cross-disease
    if cross is not None:
        o3, o3_text = DONE, f"Results loaded from {cross.name}."
    else:
        o3, o3_text = TODO, "Tau, amyloid-beta, huntingtin and TDP-43 have not been run yet."

    # Option 4: validation
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

    # Option 5: scanner tool
    if get_sequence(df):
        o5, o5_text = DONE, ("Mutation Radar scans any single or combined mutation, or a brand-new sequence: "
                             "it prepares CANYA windows, scores them with your script or an uploaded file, and charts the result.")
    else:
        o5, o5_text = TODO, "No sequence or score table found for the scanner."

    cross_name = f"results/{cross.name}" if cross is not None else "results/ a CSV with 'cross' in its name"
    items = [
        (o1, o1_text, P_HOT, ["results/step10_heatmap.png", "results/step12_wt_profile.png",
                              "results/step19_position_sensitivity.png", "results/step19_position_summary.csv"]),
        (o2, o2_text, P_XAI, ["results/step15_features.csv", "results/step16_predictions.csv",
                              "results/step17_shap_summary.png"]),
        (o3, o3_text, P_CROSS, [cross_name]),
        (o4, o4_text, P_STATS, ["results/step11_scores_with_ranks.csv", "results/step14_robustness.csv",
                                "results/step13_top10_increase.csv", "results/step13_top10_decrease.csv"]),
        (o5, o5_text, P_RADAR, ["run_all.py", "app.py"]),
    ]
    return [dict(n=i + 1, title=OPTION_TITLES[i + 1], sub=OPTION_SUBS[i + 1], status=s, text=t, page=pg, files=f)
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


@st.cache_data(show_spinner=False)
def zip_results(sig):
    """Bundle every results file into one ZIP (sig = tuple of (relative path, mtime))."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for rel, _ in sig:
            try:
                z.write(RESULTS / rel, arcname=rel)
            except Exception:
                continue
    return buf.getvalue()


def heat_colorscale():
    return [[0, BLUE], [0.5, "#ffffff"], [1, PINK]]


def goto(page):
    st.session_state["_goto"] = page


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


def architecture_fig(height=360):
    """Protein map: the three regions plus the six familial mutation sites."""
    fig = go.Figure()
    regions = [(1, 60, "N-terminal (1-60)", BLUE), (61, 95, "NAC (61-95)", PINK),
               (96, 140, "C-terminal (96-140)", PURPLE)]
    for a, b, name, color in regions:
        fig.add_shape(type="rect", x0=a - 0.5, x1=b + 0.5, y0=0, y1=1, fillcolor=color,
                      line_width=0, layer="below")
        fig.add_annotation(x=(a + b) / 2, y=0.5, text=f"<b>{name}</b>", showarrow=False,
                           font=dict(color="white", size=13, family="Nunito, sans-serif"))
    # labels are grouped so the crowded 50-53 cluster gets one readable callout
    groups = [("A30P", [30], 30, 1.9), ("E46K", [46], 42, 1.9),
              ("H50Q · G51D<br>A53T / A53E", [50, 51, 53], 60, 2.6)]
    for label, xs, lx, ly in groups:
        fig.add_annotation(x=lx, y=ly, text=f"<b>{label}</b>", showarrow=False, yanchor="bottom",
                           font=dict(size=12, color=INK, family="Nunito, sans-serif"),
                           bgcolor="rgba(255,255,255,.9)", bordercolor=PINK, borderwidth=1, borderpad=4)
        for x in xs:
            fig.add_shape(type="line", x0=x, y0=1.0, x1=lx, y1=ly, line=dict(color=INK, width=1))
    fig.add_trace(go.Scatter(
        x=[int(m[1:-1]) for m in FAMILIAL], y=[1.0] * len(FAMILIAL), mode="markers", showlegend=False,
        marker=dict(symbol="circle", size=9, color="white", line=dict(color=INK, width=2)),
        text=FAMILIAL, hovertemplate="%{text} (position %{x})<extra></extra>"))
    fig.update_xaxes(range=[0, 141], title="Residue position", showgrid=False, dtick=20, zeroline=False)
    fig.update_yaxes(visible=False, range=[-0.15, 3.4])
    fig.update_layout(height=height, title="Protein map: regions and familial Parkinson's mutation sites",
                      margin=dict(l=20, r=20, t=60, b=50))
    return fig


def key_findings(df):
    """Headline numbers computed from the score table."""
    out = []
    nac = df[df["position"].between(61, 95)]["delta"].abs().mean()
    rest = df[~df["position"].between(61, 95)]["delta"].abs().mean()
    if rest:
        out.append(card_html("NAC sensitivity", f"{nac / rest:.1f}x",
                             "mean |change| inside residues 61-95 vs the rest", PINK))
    up, down = df.loc[df["delta"].idxmax()], df.loc[df["delta"].idxmin()]
    out.append(card_html("Largest predicted increase", str(up["mutation"]),
                         f"{up['delta']:+.4f} change in CANYA score", PURPLE))
    out.append(card_html("Largest predicted decrease", str(down["mutation"]),
                         f"{down['delta']:+.4f} change in CANYA score", BLUE))
    hit = df[df["mutation"].isin(FAMILIAL)]
    if len(hit):
        out.append(card_html("Familial mutations", f"median {hit['delta_rank_pct'].median():.0f}th pct",
                             f"{len(hit)} known mutations ranked among all {len(df):,}", GREEN))
    return out


# --------------------------------------------------------------------------- #
# Query helpers shared by the Explorer and the sidebar search
# --------------------------------------------------------------------------- #
def _wt_map():
    df = get_scores()
    return dict(zip(df["position"], df["wt_aa"])) if df is not None else {}


def sync_builder(q):
    """Keep the slider and dropdown in step with a typed mutation such as A53T."""
    ss = st.session_state
    m = re.fullmatch(r"[A-Z](\d+)([A-Z])", q)
    if m and 1 <= int(m.group(1)) <= 140 and m.group(2) in AA:
        ss["b_pos"] = int(m.group(1))
        ss["b_aa"] = m.group(2)


def set_query(q):
    st.session_state["mut_query"] = q
    sync_builder(q)


def build_query():
    ss = st.session_state
    pos, aa = int(ss["b_pos"]), ss["b_aa"]
    wt = _wt_map().get(pos, "?")
    ss["mut_query"] = str(pos) if aa == wt else f"{wt}{pos}{aa}"


def surprise_query():
    df = get_scores()
    if df is not None and len(df):
        set_query(str(df.sample(1).iloc[0]["mutation"]))


def jump_to_mutation():
    q = norm_query(st.session_state.get("jump_q", ""))
    if not q:
        return
    set_query(q)
    if re.fullmatch(r"\d+", q) and 1 <= int(q) <= 140:
        st.session_state["b_pos"] = int(q)
    st.session_state["_goto"] = P_EXPLORE
    st.session_state["jump_q"] = ""


# --------------------------------------------------------------------------- #
# Oracle Challenge (guessing game) logic
# --------------------------------------------------------------------------- #
# Size of the gap between the two mutations, as a multiple of the score spread.
GAME_GAPS = {"Easy": (1.0, 99.0), "Medium": (0.3, 1.0), "Hard": (0.02, 0.3)}


def game_init():
    ss = st.session_state
    for k, v in {"g_score": 0, "g_rounds": 0, "g_streak": 0, "g_best": 0,
                 "g_pair": None, "g_result": None, "g_diff": "Medium"}.items():
        if k not in ss:
            ss[k] = v


def game_new_round():
    df = get_scores()
    if df is None or len(df) < 2:
        return
    rng = np.random.default_rng()
    spread = float(df["delta"].std()) or 1.0
    lo, hi = GAME_GAPS.get(st.session_state.get("g_diff", "Medium"), GAME_GAPS["Medium"])
    a = b = None
    for _ in range(400):
        i, j = rng.choice(len(df), size=2, replace=False)
        a, b = df.iloc[int(i)], df.iloc[int(j)]
        gap = abs(float(a["delta"]) - float(b["delta"])) / spread
        if lo <= gap < hi:
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
        '<div class="hero"><div>'
        '<h1>🧬 Alpha-Synuclein Explorer</h1>'
        '<div class="sub">AI-Based Analysis of Alpha-Synuclein Aggregation and Neurodegenerative Disease Risk</div>'
        '<div class="by">By Sana Manchikanti</div>'
        '<div class="chips">' + "".join(f'<span class="chip">{esc(c)}</span>' for c in chips) + '</div>'
        '</div>' + helix_html() + '</div>'
    )

    # quick launch
    qcols = st.columns(5)
    for col, (label, page) in zip(qcols, [("🔬 Explore a mutation", P_EXPLORE), ("🎮 Play the Oracle", P_GAME),
                                           ("🔥 See the hotspots", P_HOT), ("🧠 Known mutations", P_PD),
                                           ("📌 One-screen summary", P_SUM)]):
        with col:
            st.button(label, key=f"home_go_{page}", on_click=goto, args=(page,), use_container_width=True)

    card_grid([
        card_html("Mutations analyzed", n_mut, "single amino-acid substitutions", PURPLE),
        card_html("Proteins studied", n_prot, prot_sub, PINK),
        card_html("Models", models, "rank mutation effects", BLUE),
        card_html("SHAP", "Enabled" if shap_on else "Not run yet", "feature explanations", GREEN),
    ], 210)

    if df is not None:
        section("What the data says")
        card_grid(key_findings(df), 210)
    show_plot(architecture_fig(), key="home_arch")

    section("The four phases")
    phases = phase_info()
    pcols = st.columns(4)
    for col, p in zip(pcols, phases):
        with col:
            st.markdown(
                f'<div class="glass"><div class="num">{p["n"]}</div>'
                f'<div class="ctitle">{esc(p["title"])}</div><div class="csub">{esc(p["sub"])}</div>'
                f'<span class="status {p["status"]}">{STATUS_TEXT[p["status"]]}</span>'
                f'<div class="m-sub" style="margin-top:.5rem">{esc(p["detail"])}</div></div>',
                unsafe_allow_html=True)
            if p["page"] != P_HOME:
                st.button("Open", key=f"phase_go_{p['n']}", on_click=goto, args=(p["page"],),
                          use_container_width=True)
    st.caption("Open the Mission Map to click through every phase and option.")

    section("Pick an adventure")
    cards = [
        ("Mutation Explorer", "Build a mutation, compare two, or filter all 2,660 of them.",
         PURPLE, "🔬", option_tag(5), True, P_EXPLORE),
        ("Mutation Radar", "Scan any mutation, a double mutant, or a brand-new protein sequence.",
         BLUE, "📡", option_tag(5), True, P_RADAR),
        ("Oracle Challenge", "Guess which mutation raises aggregation more, then see what CANYA says.",
         PINK, "🎮", "Play", False, P_GAME),
        ("Hotspot Mapping", "Hover over every residue. The hot zone is residues about 61-82.",
         PINK, "🔥", option_tag(1), True, P_HOT),
        ("Explainable AI", "A random forest and SHAP show which properties drive the predictions.",
         GREEN, "🤖", option_tag(2), True, P_XAI),
        ("Parkinson Mutations", "Familial mutations compared with CANYA's predictions and published work.",
         BLUE, "🧠", option_tag(4), True, P_PD),
        ("Cross-Disease Analysis", "Tau, amyloid-beta, huntingtin and TDP-43 through the same pipeline.",
         PURPLE, "🧬", option_tag(3), True, P_CROSS),
    ]
    for row in range(0, len(cards), 3):
        acols = st.columns(3)
        for col, (t, b, a, i, tag, o, pg) in zip(acols, cards[row:row + 3]):
            with col:
                st.markdown(info_card_html(t, b, a, i, tag, option=o), unsafe_allow_html=True)
                st.button(f"Open {t}", key=f"adv_{pg}", on_click=goto, args=(pg,), use_container_width=True)

    st.info("All scores are model predictions, not laboratory measurements, and this site is "
            "not medical advice. See the About page for limitations.")


def page_summary():
    page_header("Project at a Glance", "The whole study on one screen: question, method, findings and limits.",
                tags=("Phases 1-4", "Options 1-5"))
    df = get_scores()
    seq_len = fasta_length() or (int(df["position"].max()) if df is not None else 0)
    n_win = count_lines(RESULTS / "step6_canya_output.tsv")
    cards = [
        card_html("Protein", "α-synuclein", f"UniProt P37840, {seq_len} residues" if seq_len else "UniProt P37840", PURPLE),
        card_html("Mutations scored", f"{len(df):,}" if df is not None else "n/a", "every single substitution", PINK),
        card_html("Windows scored", f"{n_win:,}" if n_win else "n/a", "20-residue peptides through CANYA", BLUE),
    ]
    if df is not None:
        cards.append(card_html("Known mutations ranked", str(int(df["mutation"].isin(FAMILIAL).sum())),
                               "familial Parkinson's set", GREEN))
    card_grid(cards, 200)

    section("The question")
    callout("Can a model of amyloid nucleation (CANYA) show which positions and which kinds of substitutions "
            "in alpha-synuclein change aggregation propensity, and does it flag mutations linked to "
            "familial Parkinson's disease?")

    section("How it works")
    steps = ["Get the sequence", "Generate every mutation", "Score 20-residue windows with CANYA",
             "Mutation score = best window; effect = mutant minus wild type",
             "Map, explain and validate"]
    html('<div class="flow">' + '<span class="arrow">&rarr;</span>'.join(
        f'<span class="node">{esc(s)}</span>' for s in steps) + '</div>')

    show_plot(architecture_fig(), key="sum_arch")

    if df is not None:
        section("What we found")
        card_grid(key_findings(df), 210)
        win = df.groupby("position")["delta"].agg(lambda s: s.abs().mean()).reset_index(name="v")
        fig = go.Figure(go.Scatter(x=win["position"], y=win["v"], mode="lines", fill="tozeroy",
                                   line=dict(color=PURPLE, width=2), fillcolor="rgba(139,92,246,.25)"))
        fig.add_vrect(x0=61, x1=95, fillcolor=PINK, opacity=0.12, line_width=0, annotation_text="NAC")
        fig.update_layout(title="Predicted sensitivity by position", xaxis_title="Position",
                          yaxis_title="Mean |change in CANYA score|", height=320)
        show_plot(fig, key="sum_profile")
    else:
        warn_missing("results/step11_scores_with_ranks.csv")

    section("Honest limits")
    callout("Scores are predictions, not measurements. CANYA scores short peptides, not the whole protein, "
            "oligomers or cells. The set of known mutations is small. This reports predicted aggregation "
            "propensity, not clinical disease risk, and is not medical advice.", "warn")
    st.caption("Credit: CANYA, Thompson et al., Science Advances 11(18):eadt5111 (2025). Sequence: UniProt P37840.")


def page_map():
    page_header("Mission Map", "Click through the four phases and five options of the project and see what each one made.",
                tags=("Phases 1-4", "Options 1-5"))

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
             f'<span class="node">{p["n"]}. {esc(p["title"])}</span>' for p in phases)
         + '<span class="arrow">&rarr;</span><span class="node" style="border-color:#ec4899">Options 1-5</span></div>')

    section("The four phases")
    card_grid([
        f'<div class="glass" style="border-top:5px solid {PURPLE}"><div class="num">{p["n"]}</div>'
        f'<div class="ctitle">{esc(p["title"])}</div><div class="csub">{esc(p["sub"])}</div>'
        f'<span class="status {p["status"]}">{STATUS_TEXT[p["status"]]}</span>'
        f'<div class="m-sub" style="margin-top:.55rem">{esc(p["detail"])}</div></div>'
        for p in phases
    ], 220)

    section("The five research options")
    card_grid([
        f'<div class="glass" style="border-top:5px solid {PINK}"><div class="tag o">Option {o["n"]}</div>'
        f'<div class="ctitle">{esc(o["title"])}</div><div class="csub">{esc(o["sub"])}</div>'
        f'<span class="status {o["status"]}">{STATUS_TEXT[o["status"]]}</span>'
        f'<div class="m-sub" style="margin-top:.55rem">{esc(o["text"])}</div></div>'
        for o in options
    ], 220)

    section("Inspect a step")
    entries = ([(f"Phase {p['n']}: {p['title']}", p) for p in phases]
               + [(f"Option {o['n']}: {o['title']}", o) for o in options])
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
    st.button(f"Open {chosen['title']}", key="map_go", on_click=goto, args=(chosen["page"],))

    if st.button("Celebrate progress", key="map_celebrate"):
        if all(p["status"] == DONE for p in phases):
            st.balloons()
            st.success("All four phases are complete. Nice work!")
        else:
            st.info("Finish the remaining phases first, then try this button again.")

    st.caption("Option 5 lives on the Mutation Radar page. CANYA itself runs in your pipeline, so the radar "
               "prepares the windows and scores them with your CANYA script or an uploaded file.")


# ----------------------------- Mutation Explorer ----------------------------- #
def explorer_single(df):
    ss = st.session_state
    seq = get_sequence(df)
    if "mut_query" not in ss:
        ss["mut_query"] = "A53T"
    if "b_pos" not in ss:
        ss["b_pos"] = 53
    if "b_aa" not in ss:
        ss["b_aa"] = "T"

    section("Build a mutation")
    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        st.slider("Position in the protein", 1, 140, key="b_pos", on_change=build_query)
    with c2:
        st.selectbox("Swap in", list(AA), key="b_aa", on_change=build_query)
    with c3:
        st.write("")
        st.button("Surprise me", key="surprise", on_click=surprise_query)

    st.caption("Or try one of the familial Parkinson's mutations:")
    cols = st.columns(len(FAMILIAL))
    for col, m in zip(cols, FAMILIAL):
        with col:
            st.button(m, key=f"quick_{m}", on_click=set_query, args=(m,))

    query = st.text_input("Or type a mutation or position (for example A53T, 73 or V71)", key="mut_query")
    q = norm_query(query)
    if not q:
        st.info("Type a mutation such as A30P, E46K, H50Q, G51D or A53T.")
        return

    sel_pos, sel_aa = None, None
    exact = df[df["mutation"] == q]
    if re.fullmatch(r"[A-Z]\d+[A-Z]", q) and not exact.empty:
        r = exact.iloc[0]
        sel_pos, sel_aa = int(r["position"]), str(r["mut_aa"])
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
        n_below = int((same_pos["delta"] < r["delta"]).sum())
        st.caption(f"At position {int(r['position'])}, {q} raises the predicted score more than "
                   f"{n_below} of the other {len(same_pos) - 1} substitutions.")
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

    if sel_pos is not None:
        st.markdown("#### Sequence neighbourhood")
        ctx = context_html(seq, sel_pos, sel_aa)
        if ctx:
            html(ctx)
            st.caption("The highlighted residue is your position (wild type, then the swapped-in amino acid). "
                       "Numbers at the ends show the residue range.")

    st.markdown("#### Where it sits in the protein")
    html(ribbon_html(df, highlight=sel_pos))
    st.caption("Each tile is a residue. Pink strength shows how much mutations there change the score, "
               "the purple underline marks the NAC region, and the dark outline is your position.")

    st.markdown("#### Results table")
    show_cols = [c for c in ["mutation", "position", "wt_aa", "mut_aa", "wt_score", "mut_score",
                             "delta", "delta_rank_pct", "rank"] if c in table.columns]
    out = table[show_cols].copy()
    out["effect"] = out["delta_rank_pct"].map(effect_category)
    out = out.round(4)
    show_df(out)
    st.download_button("Download this table (CSV)", data=out.to_csv(index=False).encode("utf-8"),
                       file_name=f"snca_{q}.csv", mime="text/csv", key="dl_single")


def explorer_compare(df):
    st.write("Put two mutations side by side and see how they differ.")
    c1, c2 = st.columns(2)
    with c1:
        qa = norm_query(st.text_input("Mutation A", value="A53T", key="cmp_a"))
    with c2:
        qb = norm_query(st.text_input("Mutation B", value="E46K", key="cmp_b"))
    ra, rb = df[df["mutation"] == qa], df[df["mutation"] == qb]
    if ra.empty or rb.empty:
        bad = [x for x, r in [(qa, ra), (qb, rb)] if r.empty]
        st.info(f"No exact match for {', '.join(repr(b) for b in bad)}. Use the form A53T (old amino acid, "
                "position, new amino acid).")
        return
    ra, rb = ra.iloc[0], rb.iloc[0]
    cards = []
    for r, accent in [(ra, PURPLE), (rb, PINK)]:
        cards.append(
            f'<div class="glass" style="border-top:5px solid {accent}">'
            f'<div class="m-title">{esc(region_of(int(r["position"])))}</div>'
            f'<div class="m-value" style="color:{accent}">{esc(r["mutation"])}</div>'
            f'<div class="m-sub">Change {r["delta"]:+.4f}, rank #{int(r["rank"])} of {len(df):,}</div>'
            f'<div class="pill" style="background:{accent}">{r["delta_rank_pct"]:.1f}th percentile</div>'
            f'<div class="m-sub" style="margin-top:.5rem">{esc(effect_category(float(r["delta_rank_pct"])))}</div></div>')
    card_grid(cards, 260)

    hyd_a = KD.get(str(ra["mut_aa"]), 0) - KD.get(str(ra["wt_aa"]), 0)
    hyd_b = KD.get(str(rb["mut_aa"]), 0) - KD.get(str(rb["wt_aa"]), 0)
    higher = ra if ra["delta"] >= rb["delta"] else rb
    callout(f"CANYA predicts {higher['mutation']} raises aggregation propensity more. "
            f"Hydrophobicity change (Kyte-Doolittle): {ra['mutation']} {hyd_a:+.1f}, {rb['mutation']} {hyd_b:+.1f}.")

    fig = go.Figure(go.Histogram(x=df["delta"], nbinsx=70, marker_color=BLUE, opacity=0.65,
                                 name="all mutations"))
    fig.add_vline(x=float(ra["delta"]), line_color=PURPLE, line_width=3, annotation_text=str(ra["mutation"]))
    fig.add_vline(x=float(rb["delta"]), line_color=PINK, line_width=3, annotation_text=str(rb["mutation"]),
                  annotation_position="top left")
    fig.update_layout(title="Where the two sit among all mutations", xaxis_title="Change in CANYA score",
                      yaxis_title="Count", showlegend=False)
    show_plot(fig, key="cmp_hist")


def explorer_browse(df):
    st.write("Filter all mutations by region, amino-acid type and predicted effect.")
    c1, c2, c3 = st.columns(3)
    with c1:
        regs = st.multiselect("Region", REGION_ORDER, default=REGION_ORDER, key="br_reg")
    with c2:
        classes = st.multiselect("Introduced amino-acid type", list(AA_CLASSES), default=list(AA_CLASSES),
                                 key="br_cls")
    with c3:
        effects = st.multiselect("Predicted effect", EFFECT_ORDER, default=EFFECT_ORDER, key="br_eff")
    lo, hi = st.slider("Percentile range (100 = largest predicted increase)", 0, 100, (0, 100), key="br_pct")
    f = df[df["region"].isin(regs) & df["aa_class"].isin(classes) & df["effect"].isin(effects)
           & df["delta_rank_pct"].between(lo, hi)]
    if f.empty:
        st.info("No mutations match these filters. Widen a filter and try again.")
        return
    card_grid([
        card_html("Matches", f"{len(f):,}", f"of {len(df):,} mutations", PURPLE),
        card_html("Mean change", f"{f['delta'].mean():+.4f}", "in CANYA score", PINK),
        card_html("Most increasing", str(f.loc[f['delta'].idxmax(), 'mutation']),
                  f"{f['delta'].max():+.4f}", GREEN),
        card_html("Most decreasing", str(f.loc[f['delta'].idxmin(), 'mutation']),
                  f"{f['delta'].min():+.4f}", BLUE),
    ], 190)
    cols = [c for c in ["mutation", "position", "region", "aa_class", "wt_score", "mut_score", "delta",
                        "delta_rank_pct", "rank", "effect"] if c in f.columns]
    out = f.sort_values("delta", ascending=False)[cols].round(4)
    show_df(out, height=420)
    st.download_button("Download filtered mutations (CSV)", data=out.to_csv(index=False).encode("utf-8"),
                       file_name="snca_filtered_mutations.csv", mime="text/csv", key="dl_browse")


def page_explorer():
    page_header("Mutation Explorer", "Build a mutation, compare two, or filter all of them.",
                tags=(phase_tag(2),))
    df = get_scores()
    if df is None:
        warn_missing("results/step11_scores_with_ranks.csv")
        return
    t1, t2, t3 = st.tabs(["Single mutation", "Compare two", "Browse and filter"])
    with t1:
        explorer_single(df)
    with t2:
        explorer_compare(df)
    with t3:
        explorer_browse(df)


# ------------------------------- Oracle Challenge ------------------------------- #
def page_game():
    page_header("Oracle Challenge", "Two mutations appear. Pick the one you think raises predicted aggregation more.",
                tags=(phase_tag(3),))
    df = get_scores()
    if df is None:
        warn_missing("results/step11_scores_with_ranks.csv")
        return
    game_init()
    ss = st.session_state

    c1, c2 = st.columns([1, 3])
    with c1:
        st.selectbox("Difficulty", list(GAME_GAPS), key="g_diff", on_change=game_new_round,
                     help="Easy pairs differ a lot in CANYA score; Hard pairs are close.")
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
        b1, b2 = st.columns(2)
        with b1:
            st.button(f"A: {pair[0]}", key="guess_a", on_click=game_guess, args=(0,), use_container_width=True)
        with b2:
            st.button(f"B: {pair[1]}", key="guess_b", on_click=game_guess, args=(1,), use_container_width=True)
        st.caption("Hint: aggregation-prone stretches like greasy (hydrophobic) residues, and prolines tend to break them up.")
    else:
        if result[2]:
            st.success("Right! You and the Oracle agree." + (" Hot streak!" if ss["g_streak"] >= 3 else ""))
        else:
            st.info("Not this time. The Oracle ranks the other mutation higher.")
        fig = go.Figure(go.Bar(x=[str(r["mutation"]) for r in rows], y=[float(r["delta"]) for r in rows],
                               marker_color=[PURPLE, PINK], text=[f"{float(r['delta']):+.4f}" for r in rows],
                               textposition="outside"))
        fig.add_hline(y=0, line_color="gray")
        fig.update_layout(title="What CANYA predicted", yaxis_title="Change in CANYA score", height=320)
        show_plot(fig, key="game_result")
        st.button("Next round", key="game_next", on_click=game_new_round)

    st.button("Reset score", key="game_reset", on_click=game_reset)
    st.caption("This is a game about CANYA's predictions, not about real laboratory aggregation.")


# ------------------------------- Hotspot Mapping ------------------------------- #
def page_hotspots():
    page_header("Hotspot Mapping", "Where in the protein do mutations change the predicted score most?",
                tags=(phase_tag(4), option_tag(1)))
    df = get_scores()
    if df is None:
        show_image("step19_position_sensitivity.png", "Mean |change in score| by position (shaded: NAC region)")
        st.info("Interactive charts need results/step11_scores_with_ranks.csv.")
        return

    pos = df.groupby("position")["delta"].agg(mean_abs=lambda s: s.abs().mean(), mean_delta="mean").reset_index()
    inside = pos[pos["position"].between(61, 95)]["mean_abs"].mean()
    outside = pos[~pos["position"].between(61, 95)]["mean_abs"].mean()
    card_grid([
        card_html("Mean |delta| inside NAC (61-95)", f"{inside:.4f}", "", PINK),
        card_html("Mean |delta| outside NAC", f"{outside:.4f}", "", BLUE),
        card_html("Ratio", f"{inside / outside:.1f}x" if outside else "n/a", "NAC vs rest", GREEN),
    ], 220)

    t1, t2, t3, t4, t5 = st.tabs(["Residue strip", "Position profile", "Heatmap",
                                  "Substitution types", "Regions and leaderboards"])

    with t1:
        lo, hi = st.slider("Zoom into a region", 1, 140, (61, 82), key="hot_zoom")
        html(ribbon_html(df, window=(lo, hi)))
        st.caption("Each tile is one residue, pink strength is the mean size of the change when that residue "
                   "is mutated, and the purple underline is the NAC region. Hover for the exact value.")
        win = df[df["position"].between(lo, hi)]
        if len(win):
            up, down = win.loc[win["delta"].idxmax()], win.loc[win["delta"].idxmin()]
            card_grid([
                card_html(f"Mean |change|, residues {lo}-{hi}", f"{win['delta'].abs().mean():.4f}",
                          f"{len(win):,} mutations in this window", PURPLE),
                card_html("Biggest predicted increase", str(up["mutation"]), f"{up['delta']:+.4f}", PINK),
                card_html("Biggest predicted decrease", str(down["mutation"]), f"{down['delta']:+.4f}", BLUE),
            ], 220)

    with t2:
        pos["smooth"] = pos["mean_abs"].rolling(5, center=True, min_periods=1).mean()
        fig = go.Figure(go.Bar(x=pos["position"], y=pos["mean_abs"], marker_color=PURPLE, name="per position",
                               opacity=0.8))
        fig.add_trace(go.Scatter(x=pos["position"], y=pos["smooth"], mode="lines",
                                 line=dict(color=PINK, width=3), name="5-residue average"))
        fig.add_vrect(x0=61, x1=95, fillcolor=PINK, opacity=0.12, line_width=0, annotation_text="NAC")
        fig.update_layout(title="Interactive: mutational sensitivity by position",
                          xaxis_title="Position", yaxis_title="Mean |change in CANYA score|",
                          legend=dict(orientation="h", y=1.1))
        show_plot(fig, key="hot_bar")
        st.markdown("#### Ten most sensitive positions")
        show_df(pos.nlargest(10, "mean_abs")[["position", "mean_abs", "mean_delta"]].round(4))
        with st.expander("Show the saved figure from the pipeline"):
            show_image("step19_position_sensitivity.png", "Mean |change in score| by position (shaded: NAC region)")

    with t3:
        piv = df.pivot_table(index="mut_aa", columns="position", values="delta").reindex(list(AA))
        lim = float(np.nanmax(np.abs(piv.values)))
        heat = go.Figure(go.Heatmap(z=piv.values, x=piv.columns, y=piv.index, zmin=-lim, zmax=lim,
                                    colorscale=heat_colorscale(), colorbar=dict(title="Δ score")))
        heat.update_layout(title="Every mutation: pink = predicted increase, blue = decrease",
                           xaxis_title="Position", yaxis_title="Amino acid substituted in", height=520)
        show_plot(heat, key="hot_heat")

    with t4:
        core = df[df["position"].between(61, 82)]
        rest = df[~df["position"].between(61, 82)]
        classes = list(AA_CLASSES)
        g1 = core.groupby("aa_class")["delta"].mean().reindex(classes)
        g2 = rest.groupby("aa_class")["delta"].mean().reindex(classes)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=classes, y=g1.values, name="Residues 61-82", marker_color=PINK))
        fig.add_trace(go.Bar(x=classes, y=g2.values, name="Rest of protein", marker_color=BLUE))
        fig.add_hline(y=0, line_color="gray")
        fig.update_layout(barmode="group", title="Mean change in score by type of amino acid introduced",
                          yaxis_title="Mean change in CANYA score", legend=dict(orientation="h", y=1.1))
        show_plot(fig, key="hot_class")
        st.caption("Groups: hydrophobic (A V I L M C), aromatic (F W Y), polar (S T N Q), positive (K R H), "
                   "negative (D E), special (G P).")

        mat = df.pivot_table(index="wt_aa", columns="mut_aa", values="delta", aggfunc="mean").reindex(
            index=list(AA), columns=list(AA))
        lim = float(np.nanmax(np.abs(mat.values))) if np.isfinite(mat.values).any() else 1.0
        heat = go.Figure(go.Heatmap(z=mat.values, x=mat.columns, y=mat.index, zmin=-lim, zmax=lim,
                                    colorscale=heat_colorscale(), colorbar=dict(title="Mean Δ")))
        heat.update_layout(title="Mean change for every wild-type to mutant swap",
                           xaxis_title="Amino acid substituted in", yaxis_title="Original amino acid",
                           height=520)
        show_plot(heat, key="hot_matrix")
        st.caption("Blank cells are swaps that never occur because that wild-type amino acid is absent "
                   "from alpha-synuclein.")

    with t5:
        fig = px.box(df, x="region", y="delta", color="region", category_orders={"region": REGION_ORDER},
                     color_discrete_sequence=[BLUE, PINK, PURPLE])
        fig.add_hline(y=0, line_color="gray")
        fig.update_layout(title="Spread of predicted changes by region", xaxis_title="",
                          yaxis_title="Change in CANYA score", showlegend=False)
        show_plot(fig, key="hot_region")

        n = st.slider("How many to show", 5, 30, 12, key="hot_n")
        left, right = st.columns(2)
        for col, kind, color, key in [(left, "increase", PINK, "hot_up"), (right, "decrease", BLUE, "hot_down")]:
            with col:
                sub = df.nlargest(n, "delta") if kind == "increase" else df.nsmallest(n, "delta")
                f = go.Figure(go.Bar(y=sub["mutation"], x=sub["delta"], orientation="h", marker_color=color,
                                     hovertext=sub["region"]))
                f.update_layout(title=f"Top {n} predicted {kind}s", xaxis_title="Change in CANYA score",
                                yaxis=dict(autorange="reversed"), height=max(320, 22 * n + 120))
                show_plot(f, key=key)


# ------------------------------- Explainable AI ------------------------------- #
def page_xai():
    page_header("Explainable AI", "Which amino-acid properties drive the predicted effects?",
                tags=(phase_tag(4), option_tag(2)))
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

    feats = read_csv_safe(RESULTS / "step15_features.csv")
    if feats is not None and "delta" in feats.columns:
        try:
            num = feats.select_dtypes(include="number").drop(columns=["delta", "position", "rank",
                                                                       "delta_rank_pct"], errors="ignore")
            num = num.loc[:, num.nunique() > 1]
            if num.shape[1]:
                corr = num.apply(lambda c: c.corr(feats["delta"], method="spearman")).dropna()
                corr = corr.reindex(corr.abs().sort_values(ascending=False).index).head(12)
                st.markdown("### Which features track the effect?")
                f2 = go.Figure(go.Bar(y=corr.index, x=corr.values, orientation="h",
                                      marker_color=[PINK if v > 0 else BLUE for v in corr.values]))
                f2.update_layout(title="Spearman correlation of each feature with the change in score",
                                 xaxis_title="Spearman correlation", yaxis=dict(autorange="reversed"),
                                 height=max(320, 28 * len(corr) + 120))
                show_plot(f2, key="xai_feat")
                st.caption("A simple, model-free check that complements SHAP. Correlation is not causation, and "
                           "related features move together.")
        except Exception:
            pass

    pred = read_csv_safe(RESULTS / "step16_predictions.csv")
    if pred is not None and {"delta", "predicted_delta"}.issubset(pred.columns):
        st.markdown("### Random forest: predicted vs actual (held-out blocks)")
        fig2 = px.scatter(pred, x="delta", y="predicted_delta", opacity=0.5,
                          color_discrete_sequence=[PINK], hover_name="mutation" if "mutation" in pred else None)
        fig2.update_layout(xaxis_title="CANYA change in score", yaxis_title="Model-predicted change")
        show_plot(fig2, key="xai_pred")
        st.caption("Ranking is moderately captured (Spearman about 0.45 in this project) "
                   "but magnitudes are not (R-squared about 0).")


# ------------------------------- Parkinson Mutations ------------------------------- #
def page_parkinson():
    page_header("Parkinson Mutations", "Familial alpha-synuclein mutations in the CANYA predictions.",
                tags=(phase_tag(4), option_tag(4)))
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

    show_plot(architecture_fig(), key="pd_arch")

    if df is not None:
        sub = df[df["mutation"].isin(FAMILIAL)].set_index("mutation").reindex(
            [m for m in FAMILIAL if m in set(df["mutation"])]).reset_index()
        fig = go.Figure(go.Bar(y=sub["mutation"], x=sub["delta_rank_pct"], orientation="h",
                               marker_color=PINK))
        fig.add_vline(x=50, line_dash="dash", line_color="black")
        fig.update_layout(title="Rank percentile of each mutation (100 = largest predicted increase)",
                          xaxis=dict(range=[0, 100], title="Percentile"), yaxis=dict(autorange="reversed"))
        show_plot(fig, key="pd_bar")

        hist = go.Figure(go.Histogram(x=df["delta"], nbinsx=70, marker_color=BLUE, opacity=0.6))
        for _, r in sub.iterrows():
            hist.add_vline(x=float(r["delta"]), line_color=PINK, line_width=2)
            hist.add_annotation(x=float(r["delta"]), y=1, yref="paper", text=str(r["mutation"]),
                                textangle=-90, showarrow=False, yanchor="top", xanchor="right",
                                font=dict(size=11, color=INK))
        hist.update_layout(title="Known mutations (pink lines) against all 2,660 predicted changes",
                           xaxis_title="Change in CANYA score", yaxis_title="Count")
        show_plot(hist, key="pd_hist")

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


# ------------------------------- Cross-Disease ------------------------------- #
def page_cross():
    page_header("Cross-Disease Analysis", "Alpha-synuclein next to tau, amyloid-beta, huntingtin and TDP-43.",
                tags=(option_tag(3),))
    proteins = ["Alpha-Synuclein", "Tau", "Amyloid-Beta", "Huntingtin", "TDP-43"]
    cross = find_cross_file()
    cdf = read_csv_safe(cross) if cross else None

    if cdf is not None and len(cdf) and cdf.shape[1] >= 2:
        st.success(f"Loaded {cross.name}")
        name_col = next((c for c in cdf.columns if "protein" in c.lower()), cdf.columns[0])
        nums = [c for c in cdf.columns if c != name_col and pd.api.types.is_numeric_dtype(cdf[c])]
        show_df(cdf)
        if nums:
            metric = st.selectbox("Metric to chart", nums, key="cross_metric") if len(nums) > 1 else nums[0]
            fig = px.bar(cdf, x=name_col, y=metric, color_discrete_sequence=[PURPLE])
            fig.update_layout(title=f"{metric} by protein")
            show_plot(fig, key="cross_bar")
            if len(nums) >= 2:
                heat = go.Figure(go.Heatmap(z=cdf[nums].values, x=nums, y=cdf[name_col],
                                            colorscale=heat_colorscale()))
                heat.update_layout(title="Protein comparison heatmap")
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
    fig.update_layout(title="Demo: protein comparison (not real data)")
    show_plot(fig, key="cross_demo_bar")
    regions = ["N-term", "Region 2", "Region 3", "Region 4", "C-term"]
    z = np.round(rng.uniform(0, 1, (5, 5)), 2)
    heat = go.Figure(go.Heatmap(z=z, x=regions, y=proteins, colorscale=heat_colorscale()))
    heat.update_layout(title="Demo: region sensitivity heatmap (not real data)")
    show_plot(heat, key="cross_demo_heat")
    st.caption("To fill this page, save a CSV such as results/cross_disease_summary.csv with one row "
               "per protein and numeric columns.")


# ------------------------------- Statistical Validation ------------------------------- #
def page_stats():
    page_header("Statistical Validation", "Do the known mutations stand out from the rest?",
                tags=(option_tag(4),))
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
    sd = float(means.std())
    z = (obs - float(means.mean())) / sd if sd else float("nan")
    med_pct = float(df[df["mutation"].isin(chosen)]["delta_rank_pct"].median())

    card_grid([
        card_html("Mean change, known set", f"{obs:+.4f}", f"n = {len(chosen)}", PURPLE),
        card_html("Mean change, all mutations", f"{all_vals.mean():+.4f}", f"n = {len(all_vals):,}", BLUE),
        card_html("P(random set ≥ known)", f"{p_high:.3f}", "10,000 permutations", PINK),
        card_html("P(random set ≤ known)", f"{p_low:.3f}", "10,000 permutations", GREEN),
    ], 210)
    card_grid([
        card_html("Effect size (z)", f"{z:+.2f}", "known mean vs random sets of the same size", PURPLE),
        card_html("Median percentile", f"{med_pct:.0f}th", "of the known set among all mutations", PINK),
    ], 210)

    if p_high < 0.05:
        callout("The known set scores higher than most random sets of the same size. With this few mutations, "
                "treat it as a hint, not proof.", "good")
    elif p_low < 0.05:
        callout("The known set scores lower than most random sets of the same size. CANYA does not flag these "
                "mutations as raising aggregation, which says something about the method's scope, not about "
                "the mutations.", "warn")
    else:
        callout("The known set cannot be told apart from random sets of the same size. With only a handful of "
                "mutations the test has little power, so this is not evidence of no effect.")

    fig = go.Figure(go.Histogram(x=means, nbinsx=60, marker_color=PURPLE, opacity=0.8))
    fig.add_vline(x=obs, line_color=PINK, line_width=3, annotation_text="known set")
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


# ------------------------------- Downloads ------------------------------- #
def page_downloads():
    page_header("Downloads", "Everything in the results folder, picked up automatically.",
                tags=(phase_tag(4),))
    files = results_files()
    if not files:
        warn_missing("results/ is empty or missing")
        return

    sig = tuple((p.relative_to(RESULTS).as_posix(), p.stat().st_mtime) for p in files)
    total_kb = sum(p.stat().st_size for p in files) / 1024
    c1, c2 = st.columns([3, 1])
    with c1:
        st.write(f"**{len(files)} files**, about {total_kb:,.0f} KB in total.")
    with c2:
        st.download_button("Download all (ZIP)", data=zip_results(sig), file_name="snca_results.zip",
                           mime="application/zip", key="dl_zip", use_container_width=True)

    mimes = {".csv": "text/csv", ".png": "image/png", ".txt": "text/plain", ".tsv": "text/tab-separated-values"}
    tabs = st.tabs(["CSV files", "PNG files", "TXT and TSV files"])
    groups = [{".csv"}, {".png"}, {".txt", ".tsv"}]
    for tab, exts in zip(tabs, groups):
        with tab:
            group = [p for p in files if p.suffix.lower() in exts]
            if not group:
                st.info(f"No {'/'.join(sorted(exts))} files found in results/.")
                continue
            for p in group:
                rel = p.relative_to(RESULTS).as_posix()
                try:
                    data = p.read_bytes()
                except Exception:
                    continue
                a, b = st.columns([4, 1])
                with a:
                    st.write(f"**{rel}**  ({len(data) / 1024:,.0f} KB)")
                with b:
                    st.download_button("Download", data=data, file_name=p.name,
                                       mime=mimes[p.suffix.lower()], key=f"dl_{rel}")
                if p.suffix.lower() == ".png":
                    with st.expander("Preview"):
                        try:
                            st.image(data, use_container_width=True)
                        except Exception:
                            st.image(data)


# ------------------------------- About ------------------------------- #
def page_about():
    page_header("About", "Research question, methods and honest limits.", tags=("Phases 1-4", "Options 1-5"))
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
    st.code("Phase 1  Meet the Protein   get the sequence\n"
            "Phase 2  Mutation Forge     generate 2,660 mutations\n"
            "Phase 3  Ask the Oracle     score 20-residue windows with CANYA\n"
            "Phase 4  Data Vault         build the dataset\n"
            "Option 1 Hotspot Atlas      landscape mapping\n"
            "Option 2 Decode the Oracle  explainable machine learning\n"
            "Option 3 Protein Crossover  cross-disease comparison\n"
            "Option 4 Reality Check      validation study\n"
            "Option 5 Mutation Radar     scanner tool",
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
             "experimental mutational scans; test other aggregation predictors; run CANYA inside the dashboard "
             "so scanning needs no upload step.")
    st.markdown("### Applications")
    st.write("Prioritising mutations for lab testing, teaching how sequence shapes aggregation, and "
             "benchmarking aggregation predictors.")
    st.markdown("### Credits")
    st.write("CANYA: Thompson et al., Science Advances 11(18):eadt5111 (2025). Sequence: UniProt P37840. "
             "Built by Sana Manchikanti.")


# ------------------------------- Mutation Radar (Option 5) ------------------------------- #
WINDOW = 20
MUT_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")
EXISTING_SCORES = "step6_canya_output.tsv"


def clean_sequence(text):
    """Accept plain letters or FASTA; return an uppercase letters-only sequence."""
    lines = [ln.strip() for ln in str(text).splitlines() if ln.strip() and not ln.strip().startswith(">")]
    return re.sub(r"[^A-Za-z]", "", "".join(lines)).upper()


def window_starts(length, positions, window=WINDOW):
    """1-based start of every window that covers at least one of the given positions."""
    starts = set()
    for p in positions:
        for s in range(max(1, p - window + 1), min(p, length - window + 1) + 1):
            starts.add(s)
    return sorted(starts)


def apply_subs(seq, subs):
    chars = list(seq)
    for _, p, m in subs:
        chars[p - 1] = m
    return "".join(chars)


def parse_mutation_specs(text, seq):
    """Turn 'A53T, A30P+A53T' into validated specs. Returns (specs, error messages)."""
    specs, errors, seen = [], [], set()
    for token in re.split(r"[,\s;]+", str(text).upper().strip()):
        if not token:
            continue
        subs, ok = [], True
        for part in re.split(r"[+/]", token):
            part = re.sub(r"^P\.", "", part.strip())
            m = MUT_RE.match(part)
            if not m:
                found = re.findall(r"[A-Z]\d+[A-Z]", part)
                if len(found) > 1:
                    errors.append(f"'{part}' looks like several mutations stuck together ({', '.join(found)}). "
                                  "Separate them with commas, or join a double mutant with +, like A30P+A53T.")
                elif found:
                    errors.append(f"'{part}' has extra characters. Did you mean {found[0]}?")
                elif part:
                    errors.append(f"Could not read '{part}'. Use old amino acid, position, new amino acid, "
                                  "like A53T.")
                ok = False
                continue
            wt, pos, mt = m.group(1), int(m.group(2)), m.group(3)
            if not 1 <= pos <= len(seq):
                errors.append(f"{part}: position {pos} is outside the sequence (1-{len(seq)}).")
                ok = False
            elif seq[pos - 1] != wt:
                errors.append(f"{part}: position {pos} is {seq[pos - 1]} in the sequence, not {wt}.")
                ok = False
            elif mt == wt or mt not in AA:
                errors.append(f"{part}: the new amino acid must be one of the other 19 standard ones.")
                ok = False
            else:
                subs.append((wt, pos, mt))
        if not ok or not subs:
            continue
        if len({p for _, p, _ in subs}) < len(subs):
            errors.append(f"{token}: the same position appears twice.")
            continue
        subs.sort(key=lambda x: x[1])
        label = "+".join(f"{w}{p}{m}" for w, p, m in subs)
        if label in seen:
            continue
        seen.add(label)
        specs.append({"label": label, "subs": subs, "positions": [p for _, p, _ in subs]})
    return specs, errors


def spec_windows(seq, spec):
    mseq = apply_subs(seq, spec["subs"])
    starts = window_starts(len(seq), spec["positions"])
    return (starts, [seq[s - 1:s - 1 + WINDOW] for s in starts],
            [mseq[s - 1:s - 1 + WINDOW] for s in starts])


def build_scan_input(seq, specs):
    """Unique window sequences to score: {sequence: id} (wild-type windows plus mutant windows)."""
    rows = {}
    for sp in specs:
        starts, wts, muts = spec_windows(seq, sp)
        for s, w, m in zip(starts, wts, muts):
            rows.setdefault(w, f"WT_{s}")
            rows.setdefault(m, f"{sp['label']}_{s}")
    return rows


def tsv_from_rows(rows):
    return "id\tsequence\n" + "\n".join(f"{i}\t{s}" for s, i in rows.items()) + "\n"


def read_table(src):
    """Read a tab- or comma-separated table that has a header row."""
    for sep in ("\t", ","):
        try:
            if hasattr(src, "seek"):
                src.seek(0)
            df = pd.read_csv(src, sep=sep)
            if df.shape[1] >= 2:
                return df
        except Exception:
            continue
    raise ValueError("Needs a table with a header row and at least two columns (sequence and score).")


@st.cache_data(show_spinner=False)
def read_existing_table(path_str, mtime):
    return read_table(path_str)


def get_scored_table(uploaded, use_existing):
    try:
        if uploaded is not None:
            return read_table(uploaded)
        if use_existing:
            p = RESULTS / EXISTING_SCORES
            if p.exists():
                return read_existing_table(str(p), p.stat().st_mtime)
    except Exception as exc:
        st.error(f"Could not read that file. {exc}")
    return None


def pick_columns(df, known, key):
    """Guess the sequence and score columns, then let the user correct them."""
    cols = list(df.columns)
    seq_guess, best = cols[0], 0
    for c in cols:
        n = int(df[c].astype(str).str.strip().str.upper().isin(known).sum())
        if n > best:
            seq_guess, best = c, n
    nums = [c for c in cols if c != seq_guess and pd.api.types.is_numeric_dtype(df[c])]
    pref = [c for c in nums if any(k in str(c).lower() for k in ("score", "nucle", "prob", "canya", "pred"))]
    score_guess = pref[0] if pref else (nums[-1] if nums else None)
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        return None, None
    c1, c2 = st.columns(2)
    with c1:
        seq_col = st.selectbox("Sequence column", cols, index=cols.index(seq_guess), key=f"{key}_seqcol")
    with c2:
        idx = num_cols.index(score_guess) if score_guess in num_cols else 0
        score_col = st.selectbox("Score column", num_cols, index=idx, key=f"{key}_scorecol")
    return seq_col, score_col


def build_score_map(df, seq_col, score_col):
    m = pd.DataFrame({"s": df[seq_col].astype(str).str.strip().str.upper(),
                      "v": pd.to_numeric(df[score_col], errors="coerce")}).dropna()
    return dict(zip(m["s"], m["v"]))


def find_scoring_scripts():
    """Python files next to app.py that look like a scoring script."""
    out = []
    for p in sorted(BASE.glob("*.py")):
        n = p.name.lower()
        if n in ("app.py", "run_all.py") or p.resolve() == Path(__file__).resolve():
            continue
        if "score" in n or "canya" in n:
            out.append(p.name)
    return out


def run_script(script, arg_tokens, timeout=1800):
    """Run a project script with the same Python as the app. Returns (returncode, stdout, stderr, seconds)."""
    cmd = [sys.executable, str(BASE / script)] + list(arg_tokens)
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=str(BASE), capture_output=True, text=True, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr, time.time() - t0
    except subprocess.TimeoutExpired:
        return -1, "", f"Stopped after {timeout // 60} minutes without finishing.", time.time() - t0
    except Exception as exc:
        return -1, "", str(exc), time.time() - t0


def scoring_runner(rows, key):
    """Optional: score the windows right here with the project's own CANYA script. Returns a table or None."""
    ss = st.session_state
    with st.expander("Or score here with your project's CANYA script (run the app on your own computer)"):
        scripts = find_scoring_scripts()
        if not scripts:
            st.info("No scoring script found next to app.py. Looking for a .py file with 'score' or 'canya' "
                    "in its name, such as score_peptides.py. You can still use the download and upload steps.")
            return None
        st.caption("The windows are saved to results/radar_input.tsv, your script runs, and its output file is "
                   "read back automatically. Use {input} and {output} in the arguments for the two file paths.")
        c1, c2 = st.columns(2)
        with c1:
            script = st.selectbox("Script", scripts, key=f"{key}_script")
            args = st.text_input("Arguments", value="{input} {output}", key=f"{key}_args")
        with c2:
            out_rel = st.text_input("Output file the script writes (inside your project folder)",
                                    value="results/radar_scored.tsv", key=f"{key}_out")
        b1, b2 = st.columns(2)
        with b1:
            show_help = st.button("Show how this script is used", key=f"{key}_help", use_container_width=True)
        with b2:
            go_run = st.button("Run scoring", key=f"{key}_run", use_container_width=True)

        out_path = (BASE / out_rel).resolve()
        if BASE.resolve() not in out_path.parents:
            st.error("The output file must be inside your project folder.")
            return None
        if show_help:
            rc, so, se, _ = run_script(script, ["--help"], timeout=60)
            st.code((so or se or "The script printed no help text.").strip()[:3000], language="text")
        if go_run:
            RESULTS.mkdir(exist_ok=True)
            in_path = RESULTS / "radar_input.tsv"
            in_path.write_text(tsv_from_rows(rows), encoding="utf-8")
            if out_path.exists():
                try:
                    out_path.unlink()
                except Exception:
                    pass
            tokens = [t.replace("{input}", str(in_path)).replace("{output}", str(out_path))
                      for t in args.split()]
            with st.spinner(f"Running {script} on {len(rows):,} windows..."):
                rc, so, se, secs = run_script(script, tokens)
            if rc == 0 and out_path.exists():
                ss[f"{key}_scored_path"] = str(out_path)
                st.success(f"Scored in {secs:.0f} seconds. Output read from {out_rel}.")
            else:
                ss.pop(f"{key}_scored_path", None)
                st.error("The script did not produce the output file." if rc == 0
                         else f"The script stopped with an error (code {rc}).")
                st.code(((se or "") + "\n" + (so or "")).strip()[-3000:] or "No output.", language="text")
        saved = ss.get(f"{key}_scored_path")
        if saved and Path(saved).exists():
            try:
                return read_table(saved)
            except Exception as exc:
                st.error(f"Could not read the output file. {exc}")
    return None


def set_example_sequence(seq):
    st.session_state["rad_seq"] = seq


def show_radar_results(df, results):
    """results: list of dicts with label, positions, wt_score, mut_score, delta, source."""
    out = pd.DataFrame(results)
    out["percentile"] = out["delta"].map(lambda d: float((df["delta"] <= d).mean() * 100))
    out["effect"] = out["percentile"].map(effect_category)
    accents = [PURPLE, PINK, BLUE, GREEN]
    card_grid([card_html(r["label"], f"{r['delta']:+.4f}", f"{r['percentile']:.0f}th percentile, {r['effect']}",
                         accents[i % 4]) for i, r in out.head(8).iterrows()], 230)

    hist = go.Figure(go.Histogram(x=df["delta"], nbinsx=70, marker_color=BLUE, opacity=0.6))
    for i, r in out.head(10).iterrows():
        hist.add_vline(x=float(r["delta"]), line_color=PINK, line_width=2)
        hist.add_annotation(x=float(r["delta"]), y=1, yref="paper", text=str(r["label"]), textangle=-90,
                            showarrow=False, yanchor="top", xanchor="right", font=dict(size=11, color=INK))
    hist.update_layout(title="Your mutations (pink lines) against all 2,660 single substitutions",
                       xaxis_title="Change in CANYA score", yaxis_title="Count")
    show_plot(hist, key="radar_hist")

    table = out[["label", "source", "wt_score", "mut_score", "delta", "percentile", "effect"]].rename(
        columns={"label": "mutation"}).round(4)
    show_df(table)
    st.download_button("Download radar results (CSV)", data=table.to_csv(index=False).encode("utf-8"),
                       file_name="snca_radar_results.csv", mime="text/csv", key="dl_radar")
    if any("+" in str(r["label"]) for r in results):
        st.caption("Percentiles compare each change with the 2,660 single substitutions. Combined mutations "
                   "are outside that set, so read their percentile as a rough guide only.")


def radar_mutations(df, seq):
    st.write("Type any mutations, one or several, and combine them with a plus sign "
             "(for example A30P+A53T for a double mutant).")
    text = st.text_area("Mutations to scan", value="A53T, A30P+A53T, E46K+A53T", height=90, key="rad_muts",
                        help="Separate with commas, spaces or new lines.")
    specs, errors = parse_mutation_specs(text, seq)
    for e in errors:
        st.warning(e)
    if not specs:
        st.info("Enter at least one valid mutation, such as A53T.")
        return
    st.caption("Understood: " + ", ".join(sp["label"] for sp in specs))

    by_label, pending = {}, []
    for sp in specs:
        hit = df[df["mutation"] == sp["label"]] if len(sp["subs"]) == 1 else pd.DataFrame()
        if len(hit):
            r = hit.iloc[0]
            by_label[sp["label"]] = dict(
                label=sp["label"], source="pipeline results", delta=float(r["delta"]),
                wt_score=float(r["wt_score"]) if "wt_score" in r.index and pd.notna(r["wt_score"]) else np.nan,
                mut_score=float(r["mut_score"]) if "mut_score" in r.index and pd.notna(r["mut_score"]) else np.nan)
        else:
            pending.append(sp)

    if pending:
        rows = build_scan_input(seq, pending)
        st.markdown("#### Score the new mutations")
        card_grid([
            card_html("Ready now", str(len(by_label)), "single mutations already scored by your pipeline", GREEN),
            card_html("Need scoring", str(len(pending)), ", ".join(p["label"] for p in pending[:4])
                      + (" ..." if len(pending) > 4 else ""), PINK),
            card_html("Windows to score", f"{len(rows):,}", f"{WINDOW}-residue peptides, wild type and mutant", PURPLE),
        ], 220)
        st.write("**Fastest:** open the scoring box below and press Run. **Or:** download the windows, run them "
                 f"through the same CANYA scoring step your pipeline uses (the one that makes results/{EXISTING_SCORES}), "
                 "and upload the scored file.")
        st.download_button("1. Download windows to score (TSV)", data=tsv_from_rows(rows).encode("utf-8"),
                           file_name="radar_windows_to_score.tsv", mime="text/tab-separated-values",
                           key="dl_radar_in")
        up = st.file_uploader("3. Upload the scored file (TSV or CSV with a header row)",
                              type=["tsv", "csv", "txt"], key="rad_up_mut")
        use_existing = False
        if (RESULTS / EXISTING_SCORES).exists():
            use_existing = st.checkbox(f"Use results/{EXISTING_SCORES} from your project instead", key="rad_use_ex_mut")
        scored = get_scored_table(up, use_existing)
        ran = scoring_runner(rows, "run_mut")
        if scored is None:
            scored = ran
        if scored is None:
            st.info("Waiting for scores. Until then, only the mutations already in your results are shown.")
        else:
            seq_col, score_col = pick_columns(scored, set(rows), "rad_mut")
            if seq_col is None:
                st.error("No numeric score column found in that file.")
            else:
                score_map = build_score_map(scored, seq_col, score_col)
                matched = sum(1 for s in rows if s in score_map)
                st.caption(f"Matched {matched:,} of {len(rows):,} windows in your file.")
                for sp in pending:
                    starts, wts, muts = spec_windows(seq, sp)
                    w = [score_map.get(x) for x in wts]
                    m = [score_map.get(x) for x in muts]
                    if any(v is None for v in w + m):
                        st.warning(f"{sp['label']}: some of its windows were not found in your file, so it was skipped.")
                        continue
                    by_label[sp["label"]] = dict(label=sp["label"], source="uploaded scores",
                                                 wt_score=max(w), mut_score=max(m), delta=max(m) - max(w))

    results = [by_label[sp["label"]] for sp in specs if sp["label"] in by_label]
    if results:
        st.markdown("#### Radar results")
        show_radar_results(df, results)
    st.caption("Mutation score = the highest-scoring window that covers a mutated residue; the effect is mutant "
               "minus wild type, the same rule the project uses everywhere else.")


def radar_sequence(seq_ref):
    st.write("Paste any protein sequence to see which stretches CANYA scores as most aggregation-prone.")
    st.button("Load alpha-synuclein as an example", key="rad_example", on_click=set_example_sequence,
              args=(seq_ref,), disabled=not seq_ref)
    raw = st.text_area("Protein sequence (plain letters or FASTA)", key="rad_seq", height=140,
                       placeholder="MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSK...")
    s = clean_sequence(raw)
    if not s:
        st.info("Paste a sequence of at least 20 amino acids.")
        return
    bad = sorted(set(s) - set(AA))
    if bad:
        st.warning(f"Unsupported letters: {', '.join(bad)}. Use the 20 standard amino-acid letters.")
        return
    if len(s) < WINDOW:
        st.error(f"That sequence has {len(s)} residues. It needs at least {WINDOW}.")
        return
    if len(s) > 3000:
        st.error("Please keep the sequence under 3,000 residues.")
        return

    rows = {}
    for i in range(1, len(s) - WINDOW + 2):
        rows.setdefault(s[i - 1:i - 1 + WINDOW], f"win_{i}")
    card_grid([card_html("Residues", f"{len(s):,}", "in your sequence", PURPLE),
               card_html("Windows to score", f"{len(rows):,}", f"every {WINDOW}-residue stretch", PINK)], 220)
    st.download_button("1. Download windows to score (TSV)", data=tsv_from_rows(rows).encode("utf-8"),
                       file_name="radar_sequence_windows.tsv", mime="text/tab-separated-values", key="dl_radar_seq")
    up = st.file_uploader("3. Upload the scored file (TSV or CSV with a header row)",
                          type=["tsv", "csv", "txt"], key="rad_up_seq")
    use_existing = False
    if (RESULTS / EXISTING_SCORES).exists():
        use_existing = st.checkbox(f"Use results/{EXISTING_SCORES} from your project instead", key="rad_use_ex_seq",
                                   help="Works when your sequence is alpha-synuclein, whose windows are already scored.")
    scored = get_scored_table(up, use_existing)
    ran = scoring_runner(rows, "run_seq")
    if scored is None:
        scored = ran
    if scored is None:
        st.info("Waiting for scores.")
        return
    seq_col, score_col = pick_columns(scored, set(rows), "rad_seq")
    if seq_col is None:
        st.error("No numeric score column found in that file.")
        return
    score_map = build_score_map(scored, seq_col, score_col)
    recs = []
    for i in range(1, len(s) - WINDOW + 2):
        w = s[i - 1:i - 1 + WINDOW]
        if w in score_map:
            recs.append({"start": i, "end": i + WINDOW - 1, "sequence": w, "score": float(score_map[w])})
    st.caption(f"Matched {len(recs):,} of {len(s) - WINDOW + 1:,} windows in your file.")
    if not recs:
        st.warning("None of the windows matched. Check that you scored the file you downloaded and picked the "
                   "right sequence column.")
        return
    wdf = pd.DataFrame(recs)
    prof = np.full(len(s), np.nan)
    for r in recs:
        a = r["start"] - 1
        prof[a:a + WINDOW] = np.fmax(prof[a:a + WINDOW], r["score"])
    top = wdf.loc[wdf["score"].idxmax()]
    card_grid([
        card_html("Highest window", f"{int(top['start'])}-{int(top['end'])}", f"score {top['score']:.3f}", PINK),
        card_html("Mean window score", f"{wdf['score'].mean():.3f}", f"{len(wdf):,} windows scored", PURPLE),
        card_html("Windows in top 10%", f"{int((wdf['score'] >= wdf['score'].quantile(0.9)).sum()):,}",
                  "relative to this sequence", BLUE),
    ], 220)

    fig = go.Figure(go.Scatter(x=np.arange(1, len(s) + 1), y=prof, mode="lines", fill="tozeroy",
                               line=dict(color=PURPLE, width=2), fillcolor="rgba(139,92,246,.22)",
                               name="best window covering each residue"))
    if s == get_sequence(get_scores()):
        fig.add_vrect(x0=61, x1=95, fillcolor=PINK, opacity=0.12, line_width=0, annotation_text="NAC")
    fig.update_layout(title="Aggregation propensity along the sequence (CANYA window scores)",
                      xaxis_title="Residue position", yaxis_title="Best window score covering the residue")
    show_plot(fig, key="radar_profile")
    st.markdown("#### Highest-scoring windows")
    show_df(wdf.nlargest(10, "score").round(4))
    st.download_button("Download all window scores (CSV)", data=wdf.to_csv(index=False).encode("utf-8"),
                       file_name="radar_sequence_scores.csv", mime="text/csv", key="dl_radar_seqres")
    st.caption("Scores are CANYA predictions for 20-residue peptides, not laboratory measurements. Only compare "
               "scores from the same scoring run.")


def page_radar():
    page_header("Mutation Radar", "Scan any mutation, a combination of mutations, or a brand-new protein sequence.",
                tags=(option_tag(5),))
    df = get_scores()
    seq = get_sequence(df)
    steps = ["Prepare 20-residue windows", "Score them with CANYA", "Upload the scores", "Read the radar"]
    html('<div class="flow">' + '<span class="arrow">&rarr;</span>'.join(
        f'<span class="node">{i + 1}. {esc(t)}</span>' for i, t in enumerate(steps)) + '</div>')
    callout("The radar builds exactly the windows CANYA needs, scores them with your project's CANYA script "
            "(or reads a scored file you upload), and does the rest: scores, percentiles and charts.")
    t1, t2 = st.tabs(["Scan mutations", "Scan a new sequence"])
    with t1:
        if df is None or not seq:
            warn_missing("results/step11_scores_with_ranks.csv")
        else:
            radar_mutations(df, seq)
    with t2:
        radar_sequence(seq)


PAGES = {
    P_HOME: page_home,
    P_SUM: page_summary,
    P_MAP: page_map,
    P_EXPLORE: page_explorer,
    P_RADAR: page_radar,
    P_GAME: page_game,
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
         '<div class="t2">by Sana Manchikanti</div></div></div>')
    current = st.session_state.get("nav", P_HOME)
    html(f'<div class="bubble">{esc(MASCOT.get(current, MASCOT[P_HOME]))}</div>')

    st.radio("Navigate", NAV, key="nav", label_visibility="collapsed")

    st.text_input("Jump to a mutation", key="jump_q", placeholder="A53T, E46K or 73",
                  on_change=jump_to_mutation)

    try:
        phases, options = phase_info(), option_info()
        score, total = progress_value()
        steps = "".join(
            f'<div class="step"><span class="dot {p["status"]}"></span>{p["n"]}. {esc(p["title"])}</div>'
            for p in phases)
        opts = "".join(
            f'<div class="opt"><span class="dot {o["status"]}"></span>{o["n"]}. {esc(o["title"])}</div>'
            for o in options)
        html(f'<div class="sidebox"><div class="h">Research path</div><div class="steps">{steps}</div></div>')
        html(f'<div class="sidebox"><div class="h">Options explored</div>{opts}</div>')
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
    if "_goto" in st.session_state:
        target = st.session_state.pop("_goto")
        if target in NAV:
            st.session_state["nav"] = target
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