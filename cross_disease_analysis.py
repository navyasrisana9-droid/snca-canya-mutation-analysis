# %% [markdown]
# # Component 5: Cross-Disease Aggregation Analysis
# Alpha-synuclein, tau, amyloid-beta, huntingtin, TDP-43.
#
# Run in VS Code / Jupyter (cells are `# %%` delimited) or as a script:
#     python cross_disease_analysis.py
# Pipeline smoke test (synthetic sequences + toy scorer, NOT real results):
#     SMOKE_TEST=1 python cross_disease_analysis.py
#
# IMPORTANT: scoring is delegated to your Component 1 Canya wrapper.
# Create `canya_wrapper.py` next to this file exposing
#     score_sequences(list_of_str) -> list_of_float
# so the exact same scorer is used for every protein.

# %% [markdown]
# ## Step 0. Setup and configuration
# Purpose: one place for every analytic choice, so choices are explicit and reportable.

# %%
import os, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score

warnings.filterwarnings("ignore")
SMOKE = os.environ.get("SMOKE_TEST") == "1"
ROOT = Path("cross_disease_smoketest" if SMOKE else "cross_disease")
DATA, RES, FIG = ROOT / "data", ROOT / "results", ROOT / "figures"
for d in (DATA, RES, FIG):
    d.mkdir(parents=True, exist_ok=True)

SEED, WINDOW, BLOCK = 42, 20, 10          # BLOCK = bootstrap block size (residues)
HOT_QUANTILE, HIGH_RISK_QUANTILE = 0.90, 0.95   # pooled thresholds (see Step 3)
MIN_RUN, MOTIF_K = 5, 3
AA = "ACDEFGHIKLMNPQRSTVWY"
rng = np.random.default_rng(SEED)

# region = (start, end) 1-based inclusive within the UniProt sequence, or None = full length.
PROTEINS = {
    "alpha-synuclein": dict(acc="P37840",   full_len=140,  region=None,       disease="Parkinson's"),
    "tau":             dict(acc="P10636-8", full_len=441,  region=None,       disease="Alzheimer's"),   # 2N4R isoform
    "amyloid-beta":    dict(acc="P05067",   full_len=770,  region=(672, 713), disease="Alzheimer's"),   # Abeta42 inside APP
    "huntingtin":      dict(acc="P42858",   full_len=3142, region=(1, 400),   disease="Huntington's"),  # N-terminal window
    "TDP-43":          dict(acc="Q13148",   full_len=414,  region=None,       disease="ALS/FTD"),
}
COLORS = dict(zip(PROTEINS, sns.color_palette("colorblind", 5)))
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 120, "savefig.dpi": 300, "font.family": "DejaVu Sans"})

def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close()

def holm(p):
    p = np.asarray(p, float); n = len(p); order = np.argsort(p)
    adj = np.maximum.accumulate((n - np.arange(n)) * p[order])
    out = np.empty(n); out[order] = np.minimum(adj, 1); return out

def bh(p):
    p = np.asarray(p, float); n = len(p); order = np.argsort(p)
    adj = np.minimum.accumulate((p[order] * n / (np.arange(n) + 1))[::-1])[::-1]
    out = np.empty(n); out[order] = np.minimum(adj, 1); return out

# %% [markdown]
# ## Step 1. Obtain canonical sequences
# Purpose: reproducible, versioned inputs. Design decisions to state in the Methods:
# * **Tau**: UniProt's canonical P10636-1 is the 758-aa PNS-tau; the 441-aa 2N4R brain isoform (P10636-8)
#   is the standard in aggregation literature, so it is used here.
# * **Amyloid-beta**: Abeta42 = APP residues 672-713 (the peptide that actually aggregates, not the 770-aa precursor).
# * **Huntingtin**: 3,142 aa is ~22x alpha-synuclein, which would dominate any pooled statistic, so a 400-aa
#   N-terminal window (exon 1 + flank, containing the polyQ tract) is the default. Change `region` to widen.
# * All metrics below are **per-residue / per-variant** so that length differences do not bias comparisons.

# %%
def load_sequence(name, cfg):
    fa = DATA / f"{cfg['acc']}.fasta"
    if not fa.exists():
        if SMOKE:
            r = np.random.default_rng(abs(hash(name)) % 2**32)
            fa.write_text(f">{name}\n" + "".join(r.choice(list(AA), cfg["full_len"])) + "\n")
        else:
            import requests
            resp = requests.get(f"https://rest.uniprot.org/uniprotkb/{cfg['acc']}.fasta", timeout=30)
            resp.raise_for_status(); fa.write_text(resp.text)
    seq = "".join(l.strip() for l in fa.read_text().splitlines() if not l.startswith(">"))
    assert len(seq) == cfg["full_len"], f"{name}: got {len(seq)} aa, expected {cfg['full_len']} - check accession/isoform"
    if cfg["region"]:
        s, e = cfg["region"]; seq = seq[s - 1:e]; offset = s - 1
    else:
        offset = 0
    if name == "amyloid-beta" and not SMOKE:
        assert seq.startswith("DAEFRHDS") and len(seq) == 42, "Abeta42 slice is wrong"
    return seq, offset

SEQS, OFFSETS = {}, {}
for n, c in PROTEINS.items():
    SEQS[n], OFFSETS[n] = load_sequence(n, c)
pd.DataFrame([dict(protein=n, uniprot=PROTEINS[n]["acc"], disease=PROTEINS[n]["disease"],
                   analysed_length=len(s), offset=OFFSETS[n], sequence=s) for n, s in SEQS.items()]
             ).to_csv(RES / "sequences_used.csv", index=False)
print({n: len(s) for n, s in SEQS.items()})

# %% [markdown]
# ## Step 2. Mutation generation and scoring (same logic as the alpha-synuclein pipeline)
# Purpose: every position x 19 substitutions. Because aggregation predictors are typically trained on short
# peptides, each variant is scored as a fixed 20-residue window around the mutated site, and
# `delta = mutant window score - wild-type window score`. **If Component 1 scored whole sequences or used a
# different window, change `WINDOW` / `windows()` to match it exactly**, otherwise the cross-protein comparison
# is not comparable to your alpha-synuclein results.

# %%
class ProxyScorer:
    """Toy scorer for SMOKE TESTS ONLY (hydrophobicity + charge penalty). Never use for results."""
    H = dict(zip("ARNDCQEGHILKMFPSTWYV", [1.8,-4.5,-3.5,-3.5,2.5,-3.5,-3.5,-.4,-3.2,4.5,3.8,-3.9,1.9,2.8,-1.6,-.8,-.7,-.9,-1.3,4.2]))
    def score_batch(self, seqs):
        out = []
        for s in seqs:
            h = np.mean([self.H[a] for a in s]); q = s.count("K") + s.count("R") - s.count("D") - s.count("E")
            out.append(1 / (1 + np.exp(-(0.6 * h - 0.15 * abs(q)))))
        return np.array(out)

class CanyaScorer:
    def __init__(self):
        from canya_wrapper import score_sequences  # your Component 1 function
        self.fn = score_sequences
    def score_batch(self, seqs):
        return np.asarray(self.fn(list(seqs)), float)

scorer = ProxyScorer() if SMOKE else CanyaScorer()
CACHE = RES / "score_cache.csv"
_cache = dict(pd.read_csv(CACHE).values) if CACHE.exists() else {}

def score_unique(seqs, batch=512):
    todo = [s for s in set(seqs) if s not in _cache]
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        _cache.update(zip(chunk, scorer.score_batch(chunk)))
    pd.DataFrame(list(_cache.items()), columns=["seq", "score"]).to_csv(CACHE, index=False)
    return {s: _cache[s] for s in set(seqs)}

def window_at(seq, i):
    L = len(seq); s = min(max(0, i - WINDOW // 2), max(0, L - WINDOW))
    return seq[s:s + WINDOW], i - s

def build_variants(name, seq):
    rows = []
    for i, wt in enumerate(seq):
        w, j = window_at(seq, i)
        for m in AA:
            if m != wt:
                rows.append((name, i + 1, wt, m, w, w[:j] + m + w[j + 1:]))
    df = pd.DataFrame(rows, columns=["protein", "pos", "wt", "mut", "wt_win", "mut_win"])
    sc = score_unique(list(df.wt_win) + list(df.mut_win))
    df["wt_score"], df["mut_score"] = df.wt_win.map(sc), df.mut_win.map(sc)
    df["delta"] = df.mut_score - df.wt_score
    df["pos_orig"] = df.pos + OFFSETS[name]
    return df.drop(columns=["wt_win", "mut_win"])

VAR = pd.concat([build_variants(n, s) for n, s in SEQS.items()], ignore_index=True)
VAR.to_csv(RES / "all_variants_scored.csv", index=False)
print(VAR.groupby("protein").size())

# %% [markdown]
# ## Step 3. Residue-level table, thresholds and the four requested metrics
# Purpose: define comparable metrics.
# * **Average aggregation propensity**: mean wild-type window score across residues.
# * **Hotspot density**: fraction of residues in hotspots, and contiguous hotspot regions (>= MIN_RUN residues) per 100 aa.
# * **High-risk mutation frequency**: fraction of variants with delta >= the pooled 95th percentile of positive deltas.
# * **Mutation sensitivity index (MSI)**: mean |delta| over the 19 substitutions at a position (protein MSI = mean over positions).
# Thresholds are pooled across proteins (not set from alpha-synuclein) so no protein is privileged; a
# sensitivity analysis over alternative thresholds is saved. If Component 2 used a fixed absolute hotspot cutoff,
# set `HOT_THR` / `HIGH_RISK_THR` manually for consistency.

# %%
KD = dict(zip("ARNDCQEGHILKMFPSTWYV", [1.8,-4.5,-3.5,-3.5,2.5,-3.5,-3.5,-.4,-3.2,4.5,3.8,-3.9,1.9,2.8,-1.6,-.8,-.7,-.9,-1.3,4.2]))
CHG = {"K": 1, "R": 1, "H": .1, "D": -1, "E": -1}

def pos_table(var, hi_thr):
    pt = var.groupby(["protein", "pos", "wt"], sort=False).agg(
        wt_score=("wt_score", "first"), pos_orig=("pos_orig", "first"),
        msi=("delta", lambda x: np.abs(x).mean()), mean_delta=("delta", "mean"),
        frac_high=("delta", lambda x: (x >= hi_thr).mean()),
        frac_protective=("delta", lambda x: (x <= -hi_thr).mean())).reset_index()
    kd, ch = [], []
    for n, s in SEQS.items():
        k = pd.Series([KD[a] for a in s]).rolling(7, center=True, min_periods=1).mean()
        c = pd.Series([CHG.get(a, 0) for a in s]).rolling(7, center=True, min_periods=1).mean()
        kd += list(k); ch += list(c)
    pt["kd"], pt["charge"] = kd, ch
    return pt

POS0 = pos_table(VAR, hi_thr=np.inf)  # placeholder to get pooled wt_score
HOT_THR = POS0.wt_score.quantile(HOT_QUANTILE)
HIGH_RISK_THR = VAR.delta[VAR.delta > 0].quantile(HIGH_RISK_QUANTILE)
POS = pos_table(VAR, HIGH_RISK_THR)
POS["is_hot"] = POS.wt_score >= HOT_THR
print(f"hotspot threshold={HOT_THR:.3f}  high-risk delta threshold={HIGH_RISK_THR:.3f}")

def runs(flag, min_run=MIN_RUN):
    out, start = [], None
    for i, f in enumerate(list(flag) + [False]):
        if f and start is None: start = i
        if not f and start is not None:
            if i - start >= min_run: out.append((start, i - 1))
            start = None
    return out

def block_ci(x, block=BLOCK, n=2000):
    x = np.asarray(x); L = len(x); block = min(block, L); nb = int(np.ceil(L / block))
    st = rng.integers(0, L - block + 1, (n, nb))
    idx = (st[:, :, None] + np.arange(block)).reshape(n, -1)[:, :L]
    m = x[idx].mean(1); return np.percentile(m, [2.5, 97.5])

def summarize(pt, var, hot_thr=HOT_THR, hi_thr=HIGH_RISK_THR):
    rows = []
    for n in PROTEINS:
        p, v = pt[pt.protein == n], var[var.protein == n]
        hot = (p.wt_score >= hot_thr).values; r = runs(hot)
        lo, hi = block_ci(p.msi.values)
        rows.append(dict(protein=n, disease=PROTEINS[n]["disease"], length=len(p),
            mean_wt_propensity=p.wt_score.mean(), hotspot_residue_fraction=hot.mean(),
            hotspot_regions_per_100aa=len(r) / len(p) * 100,
            high_risk_mutation_freq=(v.delta >= hi_thr).mean(),
            protective_mutation_freq=(v.delta <= -hi_thr).mean(),
            MSI=p.msi.mean(), MSI_ci_lo=lo, MSI_ci_hi=hi, mean_delta=v.delta.mean()))
    return pd.DataFrame(rows)

SUMM = summarize(POS, VAR); SUMM.to_csv(RES / "protein_summary_metrics.csv", index=False)
print(SUMM.round(3).to_string())

sens = []
for q in (0.80, 0.90, 0.95):
    t = POS.wt_score.quantile(q); s = summarize(POS, VAR, hot_thr=t)
    sens += [dict(hot_quantile=q, protein=r.protein, hotspot_residue_fraction=r.hotspot_residue_fraction,
                  hotspot_regions_per_100aa=r.hotspot_regions_per_100aa) for r in s.itertuples()]
pd.DataFrame(sens).to_csv(RES / "hotspot_threshold_sensitivity.csv", index=False)

hs = []
for n in PROTEINS:
    p = POS[POS.protein == n].reset_index(drop=True)
    for a, b in runs(p.is_hot):
        hs.append(dict(protein=n, start=p.pos_orig[a], end=p.pos_orig[b], length=b - a + 1,
                       sequence="".join(p.wt[a:b + 1]), mean_score=p.wt_score[a:b + 1].mean()))
pd.DataFrame(hs).to_csv(RES / "hotspot_regions.csv", index=False)

# %% [markdown]
# ## Step 4. Statistical testing (RQ1)
# Purpose: are proteins different in mutation sensitivity? **Unit of analysis = residue position**, not variant:
# the 19 variants at one site are not independent, so testing 19*L variants would be pseudoreplication.
# Residues along a chain are still autocorrelated, so p-values are optimistic; block-bootstrap CIs (Step 3) are
# the safer uncertainty statement. Effect sizes (epsilon^2, Cliff's delta) are reported beside every p-value.
# Pairwise: Mann-Whitney U, Holm-corrected within each metric; omnibus tests BH-corrected across metrics.

# %%
METRICS = ["wt_score", "msi", "mean_delta", "frac_high"]
names = list(PROTEINS)
omni, pair = [], []
for m in METRICS:
    groups = [POS.loc[POS.protein == n, m].values for n in names]
    F, pF = stats.f_oneway(*groups); H, pH = stats.kruskal(*groups)
    N, k = sum(map(len, groups)), len(groups)
    omni.append(dict(metric=m, ANOVA_F=F, ANOVA_p=pF, KW_H=H, KW_p=pH, epsilon_sq=(H - k + 1) / (N - k)))
    rows = []
    for i in range(k):
        for j in range(i + 1, k):
            U, p = stats.mannwhitneyu(groups[i], groups[j], alternative="two-sided")
            rows.append(dict(metric=m, A=names[i], B=names[j], U=U, p=p,
                             cliffs_delta=2 * U / (len(groups[i]) * len(groups[j])) - 1))
    pdf = pd.DataFrame(rows); pdf["p_holm"] = holm(pdf.p); pair.append(pdf)
OMNI = pd.DataFrame(omni); OMNI["ANOVA_q_BH"], OMNI["KW_q_BH"] = bh(OMNI.ANOVA_p), bh(OMNI.KW_p)
PAIR = pd.concat(pair, ignore_index=True)
OMNI.to_csv(RES / "stats_omnibus.csv", index=False); PAIR.to_csv(RES / "stats_pairwise.csv", index=False)
print(OMNI.round(4).to_string())

# %% [markdown]
# ## Step 5. Grouping proteins (RQ4): hierarchical clustering, PCA, UMAP, correlations
# Purpose and caveat: with only 5 proteins a protein-level PCA/UMAP is meaningless, so
# (a) residue-level features are embedded (PCA/UMAP), asking whether proteins occupy distinct regions of
#     "aggregation-behaviour space";
# (b) protein-level dendrograms use summary metrics and Wasserstein distances between delta distributions,
#     with block-bootstrap co-clustering frequencies as a stability check;
# (c) a block-cross-validated random forest asks whether protein identity is predictable from residue features
#     (high accuracy = proteins are distinct; near chance (0.2) = shared behaviour).

# %%
FEATS = ["wt_score", "msi", "mean_delta", "frac_high", "kd", "charge"]
X = StandardScaler().fit_transform(POS[FEATS])
pca = PCA(n_components=3, random_state=SEED).fit(X); PC = pca.transform(X)
POS[["PC1", "PC2", "PC3"]] = PC
pd.DataFrame(pca.components_.T, index=FEATS, columns=["PC1", "PC2", "PC3"]).to_csv(RES / "pca_loadings.csv")
try:
    import umap
    UM = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=SEED).fit_transform(X)
    POS["UMAP1"], POS["UMAP2"] = UM[:, 0], UM[:, 1]
except ImportError:
    print("umap-learn not installed (pip install umap-learn); skipping UMAP")

groups_blk = (POS.protein + "_" + (POS.pos // 20).astype(str)).values
y = POS.protein.values
pred = cross_val_predict(RandomForestClassifier(300, random_state=SEED, class_weight="balanced"), X, y,
                         groups=groups_blk, cv=GroupKFold(5))
BACC = balanced_accuracy_score(y, pred)
pd.crosstab(pd.Series(y, name="true"), pd.Series(pred, name="predicted")).to_csv(RES / "protein_identity_confusion.csv")
print(f"protein-identity balanced accuracy (chance=0.20): {BACC:.3f}")

CLUST_COLS = ["mean_wt_propensity", "hotspot_residue_fraction", "high_risk_mutation_freq", "MSI"]
Z = StandardScaler().fit_transform(SUMM[CLUST_COLS]); L_metric = linkage(Z, "ward")
D = np.zeros((5, 5))
for i, a in enumerate(names):
    for j, b in enumerate(names):
        da, db = VAR.delta[VAR.protein == a], VAR.delta[VAR.protein == b]
        pa, pb = POS.msi[POS.protein == a], POS.msi[POS.protein == b]
        D[i, j] = stats.wasserstein_distance(da, db) / VAR.delta.std() + stats.wasserstein_distance(pa, pb) / POS.msi.std()
L_dist = linkage(squareform(D, checks=False), "average")

co = np.zeros((5, 5)); B = 500
for _ in range(B):
    boot = []
    for n in names:
        p = POS[POS.protein == n]; L = len(p); bl = min(BLOCK, L)
        st = rng.integers(0, L - bl + 1, int(np.ceil(L / bl)))
        boot.append(p.iloc[(st[:, None] + np.arange(bl)).ravel()[:L]].assign(protein=n))
    bp = pd.concat(boot); vsub = VAR[["protein", "delta"]]
    rows = [[bp[bp.protein == n].wt_score.mean(), (bp[bp.protein == n].wt_score >= HOT_THR).mean(),
             bp[bp.protein == n].frac_high.mean(), bp[bp.protein == n].msi.mean()] for n in names]
    lab = fcluster(linkage(StandardScaler().fit_transform(rows), "ward"), 2, "maxclust")
    co += lab[:, None] == lab[None, :]
CO = pd.DataFrame(co / B, index=names, columns=names); CO.to_csv(RES / "cocluster_stability.csv")

def subst_matrix(n):
    return VAR[VAR.protein == n].pivot_table(index="wt", columns="mut", values="delta", aggfunc="mean"
                                             ).reindex(index=list(AA), columns=list(AA))
SM = {n: subst_matrix(n) for n in names}
SMC = pd.DataFrame(index=names, columns=names, dtype=float)
for a in names:
    for b in names:
        va, vb = SM[a].values.ravel(), SM[b].values.ravel(); ok = np.isfinite(va) & np.isfinite(vb)
        SMC.loc[a, b] = stats.spearmanr(va[ok], vb[ok])[0]
SMC.to_csv(RES / "substitution_profile_correlation.csv")
FCORR = POS[FEATS].corr(method="spearman"); FCORR.to_csv(RES / "feature_correlation_pooled.csv")

# %% [markdown]
# ## Step 6. Shared hotspots and motifs (RQ2, RQ3)
# Purpose: proteins are unrelated, so residue-level alignment of hotspots is not meaningful; shared behaviour is
# tested at the **motif** level. Sequences are mapped to a 7-class physicochemical alphabet
# (h aliphatic, a aromatic, p polar, + basic, - acidic, g Gly, x Pro); for each protein, reduced 3-mers
# (hot if >= 2 of 3 residues are in a hotspot) are tested for enrichment (Fisher's exact, BH-corrected).
# Motifs enriched in >= 2 proteins are "shared". Overlapping k-mers are not independent; treat q-values as
# exploratory and confirm with a permutation/shuffled-sequence control before publishing.

# %%
RED = {**{a: "h" for a in "AVILMC"}, **{a: "a" for a in "FWY"}, **{a: "p" for a in "STNQ"},
       **{a: "+" for a in "KRH"}, **{a: "-" for a in "DE"}, "G": "g", "P": "x"}
enriched, comp = {}, {}
for n in names:
    s = SEQS[n]; red = "".join(RED[a] for a in s); hot = POS[POS.protein == n].is_hot.values
    kms = [(red[i:i + MOTIF_K], hot[i:i + MOTIF_K].sum() >= 2) for i in range(len(s) - MOTIF_K + 1)]
    df = pd.DataFrame(kms, columns=["motif", "hot"]); res = []
    for mo, g in df.groupby("motif"):
        a, b = int(g.hot.sum()), int((~g.hot).sum())
        if a < 3: continue
        c, d = int(df.hot.sum()) - a, int((~df.hot).sum()) - b
        OR, p = stats.fisher_exact([[a, b], [c, d]]); res.append((mo, a, b, OR, p))
    r = pd.DataFrame(res, columns=["motif", "hot_n", "nonhot_n", "odds_ratio", "p"])
    if len(r): r["q"] = bh(r.p)
    r.insert(0, "protein", n); enriched[n] = r
    hs_res = "".join(np.array(list(s))[hot]); bg = "".join(np.array(list(s))[~hot])
    comp[n] = {a: np.log2(((hs_res.count(a) + 1) / (len(hs_res) + 20)) / ((bg.count(a) + 1) / (len(bg) + 20))) for a in AA}
ENR = pd.concat(enriched.values(), ignore_index=True); ENR.to_csv(RES / "motif_enrichment_all.csv", index=False)
sig = ENR[(ENR.get("q", 1) < 0.05) & (ENR.odds_ratio > 1)] if "q" in ENR else ENR.iloc[0:0]
shared = (sig.groupby("motif").protein.agg(["nunique", list]).query("nunique >= 2")
          .sort_values("nunique", ascending=False))
shared.to_csv(RES / "shared_motifs.csv")
COMP = pd.DataFrame(comp)
COMP.to_csv(RES / "hotspot_composition_log2enrichment.csv")
print("shared motifs:\n", shared.head(10))

# %% [markdown]
# ## Step 7. Publication figures

# %%
# Fig 1: mutation-landscape heatmaps (delta score; positions x substituted residue)
vmax = np.percentile(np.abs(VAR.delta), 99)
fig, axes = plt.subplots(5, 1, figsize=(10, 11))
for ax, n in zip(axes, names):
    M = VAR[VAR.protein == n].pivot(index="mut", columns="pos", values="delta").reindex(list(AA))
    im = ax.imshow(M.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   extent=[OFFSETS[n] + .5, OFFSETS[n] + M.shape[1] + .5, 20, 0])
    ax.set_yticks(np.arange(20) + .5); ax.set_yticklabels(list(AA), fontsize=5)
    ax.set_title(f"{n} ({PROTEINS[n]['disease']})", loc="left", fontsize=9); ax.set_xlabel("Residue")
fig.colorbar(im, ax=axes, shrink=.5, label="Δ aggregation score (mutant − WT)")
savefig("fig1_mutation_landscape_heatmaps")

# Fig 2: dendrograms
fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
dendrogram(L_metric, labels=names, ax=ax[0], leaf_rotation=45, color_threshold=0); ax[0].set_title("A. Summary-metric (Ward)")
dendrogram(L_dist, labels=names, ax=ax[1], leaf_rotation=45, color_threshold=0); ax[1].set_title("B. Distribution distance (average)")
savefig("fig2_dendrograms")
fig, ax = plt.subplots(figsize=(4.5, 4)); sns.heatmap(CO, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1, ax=ax)
ax.set_title("Bootstrap co-clustering frequency (k=2)"); savefig("fig2b_cluster_stability")

# Fig 3: PCA
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
for n in names:
    s = POS[POS.protein == n]; ax[0].scatter(s.PC1, s.PC2, s=8, alpha=.6, color=COLORS[n], label=n)
ev = pca.explained_variance_ratio_
ax[0].set_xlabel(f"PC1 ({ev[0]:.0%})"); ax[0].set_ylabel(f"PC2 ({ev[1]:.0%})"); ax[0].legend(markerscale=2, frameon=False)
for f, (x, yv) in zip(FEATS, pca.components_[:2].T):
    ax[1].arrow(0, 0, x, yv, head_width=.02, color="k"); ax[1].text(x * 1.1, yv * 1.1, f)
ax[1].set_xlim(-1, 1); ax[1].set_ylim(-1, 1); ax[1].set_title("Loadings"); savefig("fig3_pca")

# Fig 3b: UMAP
if "UMAP1" in POS:
    fig, ax = plt.subplots(figsize=(5, 4.2))
    for n in names:
        s = POS[POS.protein == n]; ax.scatter(s.UMAP1, s.UMAP2, s=8, alpha=.6, color=COLORS[n], label=n)
    ax.legend(markerscale=2, frameon=False); ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2"); savefig("fig3b_umap")

# Fig 4: hotspot comparison
fig = plt.figure(figsize=(11, 7)); gs = fig.add_gridspec(5, 2, width_ratios=[3, 1.2])
for i, n in enumerate(names):
    ax = fig.add_subplot(gs[i, 0]); p = POS[POS.protein == n]
    ax.plot(p.pos_orig, p.wt_score, color=COLORS[n], lw=1); ax.axhline(HOT_THR, ls="--", c="grey", lw=.7)
    for a, b in runs(p.is_hot): ax.axvspan(p.pos_orig.iloc[a], p.pos_orig.iloc[b], color=COLORS[n], alpha=.25)
    ax.set_ylabel(n, fontsize=7)
ax = fig.add_subplot(gs[:2, 1]); sns.barplot(data=SUMM, y="protein", x="hotspot_residue_fraction", palette=COLORS, ax=ax); ax.set_xlabel("Hotspot residue fraction")
ax = fig.add_subplot(gs[2:4, 1]); sns.barplot(data=SUMM, y="protein", x="hotspot_regions_per_100aa", palette=COLORS, ax=ax); ax.set_xlabel("Hotspot regions / 100 aa")
savefig("fig4_hotspot_comparison")

# Fig 5: risk distributions
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
sns.violinplot(data=VAR, x="protein", y="delta", palette=COLORS, cut=0, ax=ax[0], inner="quartile")
ax[0].axhline(HIGH_RISK_THR, ls="--", c="r", lw=.7); ax[0].axhline(-HIGH_RISK_THR, ls="--", c="b", lw=.7); ax[0].tick_params(axis="x", rotation=40)
sns.boxplot(data=POS, x="protein", y="msi", palette=COLORS, ax=ax[1], showfliers=False); ax[1].tick_params(axis="x", rotation=40)
ax[1].set_ylabel("Position MSI")
SS = SUMM.set_index("protein")[["high_risk_mutation_freq", "protective_mutation_freq"]]; SS.plot.bar(ax=ax[2], color=["#c0392b", "#2980b9"]); ax[2].tick_params(axis="x", rotation=40)
savefig("fig5_risk_distributions")

# Fig 6: correlation matrices
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
sns.heatmap(SMC.astype(float), annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax[0]); ax[0].set_title("Substitution-profile correlation (Spearman)")
sns.heatmap(FCORR, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax[1]); ax[1].set_title("Residue-feature correlation (pooled)")
savefig("fig6_correlation_matrices")

# Fig 7: hotspot composition
fig, ax = plt.subplots(figsize=(8, 3.5)); sns.heatmap(COMP.T, cmap="vlag", center=0, annot=False, ax=ax)
ax.set_title("Hotspot amino-acid enrichment (log2, hotspot vs non-hotspot)"); savefig("fig7_hotspot_composition")

# %% [markdown]
# ## Step 8. Auto-generated numeric results draft
# Writes the real numbers from this run into `results/results_draft.md`. Interpretation is left to you:
# do not claim more than the effect sizes and bootstrap CIs support.

# %%
S = SUMM.sort_values("MSI", ascending=False)
sigpairs = PAIR[PAIR.p_holm < 0.05]
lines = ["# Results draft (auto-generated; verify before use)\n",
         f"Thresholds: hotspot WT window score >= {HOT_THR:.3f} (pooled {HOT_QUANTILE:.0%}); high-risk delta >= {HIGH_RISK_THR:.3f} (pooled {HIGH_RISK_QUANTILE:.0%}).\n",
         "## RQ1 Mutation sensitivity ranking (MSI, mean |Δ|; 95% block-bootstrap CI)"]
lines += [f"- {r.protein}: {r.MSI:.3f} [{r.MSI_ci_lo:.3f}, {r.MSI_ci_hi:.3f}]; high-risk freq {r.high_risk_mutation_freq:.3%}" for r in S.itertuples()]
o = OMNI.set_index("metric")
lines += [f"\nKruskal-Wallis on position MSI: H={o.loc['msi','KW_H']:.2f}, q={o.loc['msi','KW_q_BH']:.2g}, epsilon²={o.loc['msi','epsilon_sq']:.3f}; ANOVA q={o.loc['msi','ANOVA_q_BH']:.2g}.",
          f"{len(sigpairs)} of {len(PAIR)} pairwise comparisons significant after Holm correction (see stats_pairwise.csv).",
          "\n## RQ2/3 Shared hotspots and motifs",
          f"Motifs enriched in >=2 proteins: {len(shared)} (see shared_motifs.csv).",
          "\n## RQ4 Grouping",
          f"Protein identity predicted from residue features with balanced accuracy {BACC:.2f} (chance 0.20).",
          "Dendrogram co-clustering frequencies: see cocluster_stability.csv."]
(RES / "results_draft.md").write_text("\n".join(lines))
print("\n".join(lines))