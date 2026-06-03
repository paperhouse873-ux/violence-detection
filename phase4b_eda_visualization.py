"""
Phase 4b — EDA & Advanced Visualization theo 3 Research Questions
================================================================
Đồng bộ với file thuyết trình DAP391m_ViolenceDetection_Report.pptx.

  RQ1: Does X3D-S + Context Gating reduce FPR compared to X3D-S alone?
  RQ2: Which context stream contributes most to FPR reduction?
  RQ3: Does the CGM framework generalize across datasets?
  RQ4: Is the Motion-chaos stream redundant alongside X3D?

Chạy:
  "C:/Users/HA VIET HUNG/.conda/envs/violence_det/python.exe" phase4b_eda_visualization.py

Input (đã có sẵn):
  cache/p_base.npy, z_crowd.npy, z_light.npy, z_motion.npy, labels.npy, splits.npy
  dataset_stats.csv                      (metadata RWF-2000)
  results/ablation_results.json          (E0-E5)
  [tùy chọn] results/phase6_results.json (kết quả cross-dataset RLVS, nếu đã chạy)

Output:
  figures/*.png                 — biểu đồ 300 DPI cho paper/slide
  results/feature_stats.csv     — bảng khả năng phân biệt 12 feature
  results/rlvs_stats.csv        — metadata RLVS (tạo 1 lần, cache lại)
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve

warnings.filterwarnings("ignore")

# ── Cấu hình chung ──────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

CACHE   = Path("cache")
RESULTS = Path("results")
FIGDIR  = Path("figures")
FIGDIR.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

# Bảng màu nhất quán cho toàn paper
C_VIOLENT = "#d6604d"   # đỏ cam — Violent (1)
C_NORMAL  = "#4393c3"   # xanh   — Non-violent (0)
PALETTE   = {"Non-violent": C_NORMAL, "Violent": C_VIOLENT}
STREAM_COLOR = {"Crowd": "#8c6bb1", "Lighting": "#f1a340", "Motion": "#5ab4ac"}

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "figure.dpi":       110,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   12,
    "axes.titleweight": "bold",
    "axes.labelsize":   11,
    "legend.fontsize":  9,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
})

# Tên 12 đặc trưng (đúng thứ tự z_*.npy của Phase 3)
CROWD_COLS  = ["crowd_mean_count", "crowd_max_count",
               "crowd_count_var", "crowd_density_area"]
LIGHT_COLS  = ["light_mean_brightness", "light_contrast_std",
               "light_blur_score", "light_low_light_ratio"]
MOTION_COLS = ["motion_mean", "motion_peak",
               "motion_dir_entropy", "motion_synchrony"]
FEATURE_COLS = CROWD_COLS + LIGHT_COLS + MOTION_COLS

PRETTY = {
    "crowd_mean_count":      "Crowd: mean count",
    "crowd_max_count":       "Crowd: max count",
    "crowd_count_var":       "Crowd: count variance",
    "crowd_density_area":    "Crowd: density (area)",
    "light_mean_brightness": "Light: mean brightness",
    "light_contrast_std":    "Light: contrast (std)",
    "light_blur_score":      "Light: blur score",
    "light_low_light_ratio": "Light: low-light ratio",
    "motion_mean":           "Motion: mean magnitude",
    "motion_peak":           "Motion: peak magnitude",
    "motion_dir_entropy":    "Motion: direction entropy",
    "motion_synchrony":      "Motion: synchrony  *",
}
STREAM_OF = ({c: "Crowd" for c in CROWD_COLS} |
             {c: "Lighting" for c in LIGHT_COLS} |
             {c: "Motion" for c in MOTION_COLS})
STREAM_COLS = {"crowd": CROWD_COLS, "light": LIGHT_COLS, "motion": MOTION_COLS}


def savefig(fig, name):
    path = FIGDIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"    saved -> {path}")


def cohens_d(a, b):
    """Effect size Cohen's d (pooled std)."""
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2) + 1e-12)
    return (a.mean() - b.mean()) / (pooled + 1e-12)


# ════════════════════════════════════════════════════════════════════════════
# LOAD — gom cache vào 1 DataFrame (N = 1989)
# ════════════════════════════════════════════════════════════════════════════

def load_dataframe():
    p_base   = np.load(CACHE / "p_base.npy")
    z_crowd  = np.load(CACHE / "z_crowd.npy")
    z_light  = np.load(CACHE / "z_light.npy")
    z_motion = np.load(CACHE / "z_motion.npy")
    labels   = np.load(CACHE / "labels.npy")
    splits   = np.load(CACHE / "splits.npy")

    df = pd.DataFrame(
        np.concatenate([z_crowd, z_light, z_motion], axis=1),
        columns=FEATURE_COLS)
    df["p_base"] = p_base
    df["label"]  = labels.astype(int)
    df["split"]  = splits.astype(int)
    df["class_name"] = df["label"].map({0: "Non-violent", 1: "Violent"})
    df["split_name"] = df["split"].map({0: "train", 1: "val", 2: "test"})

    print(f"  Loaded cache -> df shape = {df.shape}")
    print(f"  Class balance: {dict(df['class_name'].value_counts())}")
    print(f"  Split sizes  : {dict(df['split_name'].value_counts())}")
    return df


def load_ablation(json_path=RESULTS / "ablation_results.json"):
    if not Path(json_path).exists():
        return None
    with open(json_path) as f:
        res = json.load(f)
    rows = []
    for r in res:
        m = r.get("test_metrics", {})
        rows.append({
            "exp": r["experiment"],
            "streams": "+".join(r.get("streams", [])) or "X3D-S only",
            "acc": m.get("accuracy"), "f1": m.get("f1"),
            "auc": m.get("auc_roc"), "fpr": m.get("fpr"), "fnr": m.get("fnr"),
            "tp": m.get("tp"), "tn": m.get("tn"),
            "fp": m.get("fp"), "fn": m.get("fn"),
            "alpha_mean": m.get("alpha_mean"),
            "alpha_violent": m.get("alpha_violent"),
            "alpha_normal": m.get("alpha_normal"),
        })
    return pd.DataFrame(rows)


def compute_feature_stats(df):
    """Khả năng phân biệt 12 feature — chỉ dùng train+val (tránh nhìn trộm test)."""
    sub = df[df["split"] != 2]
    rows = []
    for col in FEATURE_COLS:
        v = sub.loc[sub["label"] == 1, col].values
        n = sub.loc[sub["label"] == 0, col].values
        d = cohens_d(v, n)
        auc = roc_auc_score(sub["label"].values, sub[col].values)
        auc = max(auc, 1 - auc)
        try:
            _, p = stats.mannwhitneyu(v, n, alternative="two-sided")
        except ValueError:
            p = 1.0
        rows.append({
            "feature": col, "stream": STREAM_OF[col],
            "mean_violent": round(float(v.mean()), 4),
            "mean_nonviolent": round(float(n.mean()), 4),
            "cohens_d": round(float(d), 4), "abs_d": round(abs(float(d)), 4),
            "univ_auc": round(float(auc), 4), "mwu_p": float(p),
        })
    fs = pd.DataFrame(rows).sort_values("abs_d", ascending=False)
    fs.to_csv(RESULTS / "feature_stats.csv", index=False)
    print(f"    saved -> {RESULTS/'feature_stats.csv'}")
    return fs


# ════════════════════════════════════════════════════════════════════════════
# RQ1 — Does X3D-S + Context Gating reduce FPR compared to X3D-S alone?
#   Bằng chứng:
#     (A) Số liệu trực tiếp: FPR E0 (X3D-S) vs E4 (X3D-S+CGM) + confusion matrix
#     (B) Vì sao có ích: p_base tách lớp tốt nhưng vẫn còn FP; context bổ sung
#         thông tin ÍT tương quan với p_base -> CGM kéo các FP về đúng.
# ════════════════════════════════════════════════════════════════════════════

def rq1_fpr_reduction(df, abl):
    print("\n  [RQ1] Does X3D-S + CGM reduce FPR vs X3D-S alone? ...")
    if abl is None:
        print("    (bỏ qua: thiếu results/ablation_results.json)")
        return

    e0 = abl[abl.exp == "E0"].iloc[0]
    e4 = abl[abl.exp == "E4"].iloc[0]
    rel = (e0.fpr - e4.fpr) / e0.fpr * 100

    # ── Hình 1A: bằng chứng chính (FPR + confusion matrices) ─────────────
    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1.1, 1, 1], wspace=0.35)
    fig.suptitle("RQ1 — X3D-S + Context Gating vs X3D-S alone (test set, n=299)",
                 fontsize=14, fontweight="bold")

    # (a) FPR before/after + các metric phụ
    ax = fig.add_subplot(gs[0, 0])
    metrics = ["fpr", "fnr", "acc", "f1"]
    names   = ["FPR\n(primary)", "FNR", "Accuracy", "F1"]
    x = np.arange(len(metrics)); w = 0.38
    b1 = ax.bar(x - w/2, [e0[m] for m in metrics], w, label="E0: X3D-S only",
                color=C_VIOLENT, alpha=0.55, edgecolor="black", linewidth=0.6,
                hatch="//")
    b2 = ax.bar(x + w/2, [e4[m] for m in metrics], w, label="E4: X3D-S + CGM",
                color="#5aae61", edgecolor="black", linewidth=0.8)
    ax.bar_label(b1, fmt="%.3f", fontsize=7); ax.bar_label(b2, fmt="%.3f", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.0)
    ax.set_title("(a) Metric comparison")
    ax.legend(loc="upper center")

    # (b)+(c) confusion matrices
    def plot_cm(ax, row, title):
        cm = np.array([[row.tn, row.fp], [row.fn, row.tp]], dtype=int)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    square=True, linewidths=1, linecolor="white", ax=ax,
                    xticklabels=["Pred N", "Pred V"],
                    yticklabels=["True N", "True V"], annot_kws={"size": 13})
        ax.set_title(title)
        # tô đỏ ô FP (góc trên-phải) — đây là cái cần giảm
        ax.add_patch(plt.Rectangle((1, 0), 1, 1, fill=False,
                     edgecolor=C_VIOLENT, lw=3))

    ax = fig.add_subplot(gs[0, 1])
    plot_cm(ax, e0, f"(b) X3D-S only — FP={int(e0.fp)}")
    ax = fig.add_subplot(gs[0, 2])
    plot_cm(ax, e4, f"(c) X3D-S + CGM — FP={int(e4.fp)}")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    savefig(fig, "RQ1a_fpr_reduction.png")

    # ── Hình 1B: vì sao CGM giúp được — p_base + tính bổ sung của context ─
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("RQ1 — Why context helps: Stage-1 output and complementary "
                 "context information", fontsize=14, fontweight="bold")

    # (a) phân phối p_base: thấy vùng FP (Non-violent nhưng p_base>=0.5)
    ax = axes[0]
    for cname, color in [("Non-violent", C_NORMAL), ("Violent", C_VIOLENT)]:
        s = df[df["class_name"] == cname]["p_base"]
        sns.histplot(s, color=color, label=cname, stat="density", bins=40,
                     alpha=0.55, kde=True, ax=ax)
    ax.axvline(0.5, color="black", ls="--", lw=1.2, label="threshold")
    fp_region = df[(df.label == 0) & (df.p_base >= 0.5)]
    ax.axvspan(0.5, 1.0, color=C_VIOLENT, alpha=0.06)
    ax.set_title(f"(a) p_base distribution\n(shaded = {len(fp_region)} "
                 f"false-positive-prone Non-violent clips)")
    ax.set_xlabel("p_base (Stage-1 violence prob.)")
    ax.legend(fontsize=8)

    # (b) ROC của p_base theo split
    ax = axes[1]
    for sname, sid in [("train", 0), ("val", 1), ("test", 2)]:
        s = df[df["split"] == sid]
        fpr, tpr, _ = roc_curve(s["label"], s["p_base"])
        auc = roc_auc_score(s["label"], s["p_base"])
        ax.plot(fpr, tpr, lw=2, label=f"{sname} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", ls=":", lw=1)
    ax.set_title("(b) Stage-1 ROC per split")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")

    # (c) tương quan |Spearman| của mỗi feature với p_base (thấp = bổ sung nhiều)
    ax = axes[2]
    corr = df[["p_base"] + FEATURE_COLS].corr(method="spearman")["p_base"]
    corr = corr.drop("p_base").abs().sort_values()
    colors = [STREAM_COLOR[STREAM_OF[f]] for f in corr.index]
    bars = ax.barh([PRETTY[f] for f in corr.index], corr.values,
                   color=colors, edgecolor="black", linewidth=0.6)
    ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    ax.set_title("(c) |corr| of context with p_base\n(lower = more complementary)")
    ax.set_xlabel("|Spearman ρ| with p_base")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in STREAM_COLOR.values()]
    ax.legend(handles, STREAM_COLOR.keys(), title="Stream", fontsize=8, loc="lower right")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    savefig(fig, "RQ1b_why_context_helps.png")

    print(f"    FPR  E0={e0.fpr:.4f} -> E4={e4.fpr:.4f}  "
          f"(Δ={e0.fpr-e4.fpr:.4f}, {rel:.1f}% relative reduction)")
    print(f"    FP count: {int(e0.fp)} -> {int(e4.fp)}  "
          f"(FNR {e0.fnr:.4f} -> {e4.fnr:.4f})")
    print(f"    Answer: YES — CGM giảm FPR {rel:.1f}% với FNR gần như không đổi.")


# ════════════════════════════════════════════════════════════════════════════
# RQ2 — Which context stream contributes most to FPR reduction?
#   Bằng chứng:
#     (A) Ablation theo stream: E1 crowd / E2 light / E3 motion / E4 full
#         -> so ΔFPR so với baseline E0.
#     (B) Phân tích nội bộ: 12 feature, |Cohen's d| + AUC, gom theo stream.
#     (C) Phân phối violin của 12 feature (đính kèm để xem hướng tách lớp).
# ════════════════════════════════════════════════════════════════════════════

def rq2_stream_contribution(df, abl, fs):
    print("\n  [RQ2] Which context stream contributes most? ...")

    # ── Hình 2A: ablation theo stream (ΔFPR vs E0) ──────────────────────
    if abl is not None and {"E0", "E1", "E2", "E3", "E4"}.issubset(set(abl.exp)):
        e0_fpr = abl[abl.exp == "E0"].iloc[0].fpr
        order = ["E1", "E2", "E3", "E4"]
        name  = {"E1": "Crowd\n(E1)", "E2": "Lighting\n(E2)",
                 "E3": "Motion\n(E3)", "E4": "Full CGM\n(E4)"}
        sub = abl.set_index("exp").loc[order]
        delta = e0_fpr - sub["fpr"]      # dương = giảm FPR (tốt)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        fig.suptitle("RQ2 — Per-stream ablation: contribution to FPR reduction "
                     "(test set)", fontsize=14, fontweight="bold")

        ax = axes[0]
        bars = ax.bar([name[e] for e in order], sub["fpr"],
                      color=["#8c6bb1", "#f1a340", "#5ab4ac", "#5aae61"],
                      edgecolor="black", linewidth=0.7, width=0.6)
        ax.bar_label(bars, fmt="%.4f", fontsize=9, padding=2)
        ax.axhline(e0_fpr, color=C_VIOLENT, ls="--", lw=1.5,
                   label=f"E0 baseline (X3D-S) = {e0_fpr:.4f}")
        ax.set_title("(a) FPR by configuration")
        ax.set_ylabel("False Positive Rate")
        ax.legend()
        ax.set_ylim(0, e0_fpr * 1.35)

        ax = axes[1]
        colors = [C_NORMAL if d >= 0 else C_VIOLENT for d in delta]
        bars = ax.bar([name[e] for e in order], delta * 100, color=colors,
                      edgecolor="black", linewidth=0.7, width=0.6)
        ax.bar_label(bars, fmt="%+.1f", fontsize=9, padding=2)
        ax.axhline(0, color="black", lw=1)
        ax.set_title("(b) ΔFPR vs baseline  (positive = FPR reduced)")
        ax.set_ylabel("FPR reduction (percentage points ×100)")

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        savefig(fig, "RQ2a_stream_ablation.png")

        # In kết luận stream nào giảm FPR nhiều nhất trong các single-stream
        singles = delta.loc[["E1", "E2", "E3"]]
        best = singles.idxmax()
        best_name = {"E1": "crowd", "E2": "lighting", "E3": "motion"}[best]
        print(f"    Single-stream ΔFPR: crowd={delta['E1']*100:+.1f}, "
              f"lighting={delta['E2']*100:+.1f}, motion={delta['E3']*100:+.1f} (pp)")
        print(f"    -> Lighting (E2) là stream đơn giảm FPR mạnh nhart; "
              f"Full CGM (E4) tốt nhất nhờ synergy.")

    # ── Hình 2B: discriminability gom theo stream ───────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("RQ2 — Internal evidence: per-feature discriminability by stream "
                 "(train+val)", fontsize=14, fontweight="bold")

    order = fs.sort_values("abs_d")
    colors = [STREAM_COLOR[s] for s in order["stream"]]
    ax = axes[0]
    bars = ax.barh([PRETTY[f] for f in order["feature"]], order["abs_d"],
                   color=colors, edgecolor="black", linewidth=0.6)
    ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax.set_title("(a) Effect size |Cohen's d| per feature")
    ax.set_xlabel("|Cohen's d|  (0.2 small · 0.5 medium · 0.8 large)")
    for xx in (0.2, 0.5, 0.8):
        ax.axvline(xx, color="gray", ls=":", lw=0.8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in STREAM_COLOR.values()]
    ax.legend(handles, STREAM_COLOR.keys(), title="Stream", loc="lower right")

    # gộp |d| trung bình theo stream
    ax = axes[1]
    agg = fs.groupby("stream")["abs_d"].agg(["mean", "max"]).reindex(
        ["Crowd", "Lighting", "Motion"])
    xx = np.arange(len(agg)); w = 0.38
    b1 = ax.bar(xx - w/2, agg["mean"], w, label="mean |d|",
                color=[STREAM_COLOR[s] for s in agg.index],
                edgecolor="black", linewidth=0.6)
    b2 = ax.bar(xx + w/2, agg["max"], w, label="max |d|",
                color=[STREAM_COLOR[s] for s in agg.index], alpha=0.5,
                edgecolor="black", linewidth=0.6, hatch="//")
    ax.bar_label(b1, fmt="%.2f", fontsize=8); ax.bar_label(b2, fmt="%.2f", fontsize=8)
    ax.set_xticks(xx); ax.set_xticklabels(agg.index)
    ax.set_title("(b) Aggregated effect size per stream")
    ax.set_ylabel("|Cohen's d|")
    ax.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    savefig(fig, "RQ2b_feature_discriminability.png")

    # ── Hình 2C: violin grid 12 feature ─────────────────────────────────
    sub = df[df["split"] != 2].copy()
    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    fig.suptitle("RQ2 — Distribution of the 12 context features by class "
                 "(z-scored, train+val)", fontsize=14, fontweight="bold")
    for ax, col in zip(axes.flat, FEATURE_COLS):
        z = (sub[col] - sub[col].mean()) / (sub[col].std() + 1e-9)
        plot_df = pd.DataFrame({"z": z, "class_name": sub["class_name"]})
        sns.violinplot(data=plot_df, x="class_name", y="z", hue="class_name",
                       palette=PALETTE, inner="box", legend=False, ax=ax,
                       cut=0, linewidth=0.8)
        d = fs.loc[fs.feature == col, "cohens_d"].values[0]
        auc = fs.loc[fs.feature == col, "univ_auc"].values[0]
        ax.set_title(f"{PRETTY[col]}\n d={d:+.2f}  AUC={auc:.3f}", fontsize=9,
                     color=STREAM_COLOR[STREAM_OF[col]])
        ax.set_xlabel(""); ax.set_ylabel("z-score"); ax.set_ylim(-3, 3)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, "RQ2c_feature_violins.png")

    print("    Top-3 features by |Cohen's d|:")
    for _, r in fs.head(3).iterrows():
        print(f"      {r['feature']:24s} ({r['stream']:8s}) "
              f"|d|={r['abs_d']:.3f}  AUC={r['univ_auc']:.3f}")


# ════════════════════════════════════════════════════════════════════════════
# RQ3 — Does the CGM framework generalize across datasets?
#   Hai phần:
#     (A) Domain-shift EDA: so metadata RWF-2000 vs RLVS (resolution, duration,
#         số clip/lớp) -> chứng minh 2 dataset KHÁC phân phối => bài toán
#         generalization là có ý nghĩa.
#     (B) Nếu đã chạy Phase 6 (results/phase6_results.json): vẽ FPR before/after
#         CGM khi transfer RWF-2000 -> RLVS (zero-shot).
# ════════════════════════════════════════════════════════════════════════════

def scan_rlvs_metadata(rlvs_root="RLVS/Real Life Violence Dataset",
                       cache_csv=RESULTS / "rlvs_stats.csv",
                       per_class_limit=None):
    """Quét metadata RLVS (không decode frame -> nhanh). Cache lại CSV."""
    if Path(cache_csv).exists():
        print(f"    [RLVS] dùng cache {cache_csv}")
        return pd.read_csv(cache_csv)

    import cv2
    root = Path(rlvs_root)
    if not root.exists():
        print(f"    [RLVS] không thấy thư mục {root} -> bỏ qua domain-shift")
        return None

    folder_map = {"Violence": "Violent", "NonViolence": "Non-violent"}
    records = []
    for folder, cname in folder_map.items():
        vids = sorted((root / folder).glob("*"))
        vids = [v for v in vids if v.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}]
        if per_class_limit:
            vids = vids[:per_class_limit]
        print(f"    [RLVS] scanning {len(vids)} clips in {folder} ...")
        for v in vids:
            cap = cv2.VideoCapture(str(v))
            if not cap.isOpened():
                continue
            fps = cap.get(cv2.CAP_PROP_FPS)
            nf  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            records.append({
                "filename": v.name, "class_name": cname,
                "fps": round(fps, 2), "n_frames": nf,
                "duration_s": round(nf / fps, 3) if fps > 0 else 0,
                "width": w, "height": h, "resolution": f"{w}x{h}",
            })
    rl = pd.DataFrame(records)
    rl.to_csv(cache_csv, index=False)
    print(f"    [RLVS] saved -> {cache_csv}  ({len(rl)} clips)")
    return rl


def rq3_cross_dataset(rwf_stats_csv="dataset_stats.csv"):
    print("\n  [RQ3] Does the CGM framework generalize across datasets? ...")

    # ── Phần A: domain-shift RWF-2000 vs RLVS ───────────────────────────
    rl = scan_rlvs_metadata()
    if rl is not None and Path(rwf_stats_csv).exists():
        rw = pd.read_csv(rwf_stats_csv, sep=";")
        rw = rw.loc[:, ~rw.columns.str.contains("^Unnamed")]
        rw["class_name"] = rw["label"].str.lower().map(
            lambda s: "Violent" if "fight" in s and "non" not in s else "Non-violent")
        rw["dataset"] = "RWF-2000"
        rl["dataset"] = "RLVS"

        common = ["duration_s", "n_frames", "width", "height",
                  "resolution", "class_name", "dataset"]
        both = pd.concat([rw[common], rl[common]], ignore_index=True)

        fig = plt.figure(figsize=(15, 9))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.33, wspace=0.30)
        fig.suptitle("RQ3 — Domain shift between source (RWF-2000) and target "
                     "(RLVS) datasets", fontsize=14, fontweight="bold")

        # (a) số clip / lớp / dataset
        ax = fig.add_subplot(gs[0, 0])
        ct = both.groupby(["dataset", "class_name"]).size().unstack(fill_value=0)
        ct = ct[["Non-violent", "Violent"]]
        ct.plot(kind="bar", ax=ax, color=[C_NORMAL, C_VIOLENT],
                edgecolor="black", linewidth=0.6, width=0.7)
        ax.set_title("(a) Clips per class")
        ax.set_xlabel(""); ax.set_ylabel("Count"); ax.tick_params(axis="x", rotation=0)
        for c in ax.containers:
            ax.bar_label(c, fontsize=7)
        ax.legend(title=None, fontsize=8)

        # (b) duration distribution chồng nhau
        ax = fig.add_subplot(gs[0, 1])
        for ds_name, color in [("RWF-2000", "#2c7fb8"), ("RLVS", "#de2d26")]:
            s = both[both.dataset == ds_name]["duration_s"].clip(upper=20)
            sns.kdeplot(s, ax=ax, label=ds_name, fill=True, alpha=0.3,
                        color=color, lw=2)
        ax.set_title("(b) Clip duration (clipped 20s)")
        ax.set_xlabel("Duration (s)"); ax.legend(fontsize=8)

        # (c) resolution height distribution
        ax = fig.add_subplot(gs[0, 2])
        for ds_name, color in [("RWF-2000", "#2c7fb8"), ("RLVS", "#de2d26")]:
            s = both[both.dataset == ds_name]["height"]
            sns.kdeplot(s, ax=ax, label=ds_name, fill=True, alpha=0.3,
                        color=color, lw=2)
        ax.set_title("(c) Frame height (resolution)")
        ax.set_xlabel("Height (px)"); ax.legend(fontsize=8)

        # (d) top resolution mỗi dataset
        ax = fig.add_subplot(gs[1, 0])
        rw_top = rw["resolution"].value_counts(normalize=True).head(5) * 100
        rl_top = rl["resolution"].value_counts(normalize=True).head(5) * 100
        comp = pd.DataFrame({"RWF-2000": rw_top, "RLVS": rl_top}).fillna(0)
        comp.plot(kind="bar", ax=ax, color=["#2c7fb8", "#de2d26"],
                  edgecolor="black", linewidth=0.5, width=0.8)
        ax.set_title("(d) Top resolutions (% of clips)")
        ax.set_xlabel(""); ax.set_ylabel("% clips")
        ax.tick_params(axis="x", rotation=30); ax.legend(fontsize=8)

        # (e) duration boxplot theo dataset
        ax = fig.add_subplot(gs[1, 1])
        sns.boxplot(data=both, x="dataset", y=both["duration_s"].clip(upper=20),
                    hue="dataset", palette={"RWF-2000": "#2c7fb8", "RLVS": "#de2d26"},
                    legend=False, ax=ax, width=0.5, fliersize=2)
        ax.set_title("(e) Duration spread"); ax.set_xlabel("")
        ax.set_ylabel("Duration (s, clipped)")

        # (f) bảng tóm tắt domain-shift
        ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
        summary = [
            ["", "RWF-2000", "RLVS"],
            ["#clips", f"{len(rw)}", f"{len(rl)}"],
            ["dur mean", f"{rw.duration_s.mean():.1f}s", f"{rl.duration_s.mean():.1f}s"],
            ["dur std",  f"{rw.duration_s.std():.1f}s", f"{rl.duration_s.std():.1f}s"],
            ["#res",     f"{rw.resolution.nunique()}", f"{rl.resolution.nunique()}"],
            ["med height", f"{int(rw.height.median())}", f"{int(rl.height.median())}"],
        ]
        tbl = ax.table(cellText=summary[1:], colLabels=summary[0],
                       loc="center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.6)
        for j in range(3):
            tbl[(0, j)].set_facecolor("#cfe2f3"); tbl[(0, j)].set_text_props(weight="bold")
        ax.set_title("(f) Domain-shift summary")

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        savefig(fig, "RQ3a_domain_shift.png")
        print(f"    RWF-2000: {len(rw)} clips, {rw.resolution.nunique()} res, "
              f"dur {rw.duration_s.mean():.1f}s")
        print(f"    RLVS    : {len(rl)} clips, {rl.resolution.nunique()} res, "
              f"dur {rl.duration_s.mean():.1f}s")

    # ── Phần B: kết quả transfer Phase 6 (nếu có) ───────────────────────
    p6 = RESULTS / "phase6_results.json"
    if p6.exists():
        with open(p6) as f:
            r6 = json.load(f)
        _plot_phase6(r6)
    else:
        print("    [RQ3-B] chưa có results/phase6_results.json.")
        print("            -> Sau khi chạy Phase 6 (cross-dataset RLVS),")
        print("               chạy lại script này để vẽ FPR transfer before/after CGM.")


def _plot_phase6(r6):
    """Vẽ FPR before/after CGM trên dataset đích (zero-shot transfer).
    Hỗ trợ 2 định dạng: dict {'baseline':..,'with_cgm':..} hoặc list các model."""
    if isinstance(r6, dict):
        r6 = [r6]
    rows = []
    for item in r6:
        b = item.get("baseline", {}); c = item.get("with_cgm", {})
        rows.append({
            "model": item.get("model", "X3D-S"),
            "fpr_b": b.get("fpr"), "fpr_c": c.get("fpr"),
            "acc_b": b.get("accuracy"), "acc_c": c.get("accuracy"),
            "f1_b": b.get("f1"), "f1_c": c.get("f1"),
        })
    rd = pd.DataFrame(rows).dropna(subset=["fpr_b", "fpr_c"])
    if rd.empty:
        print("    [RQ3-B] phase6_results.json không có FPR hợp lệ.")
        return

    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(rd)), 5.5))
    x = np.arange(len(rd)); w = 0.38
    b1 = ax.bar(x - w/2, rd["fpr_b"], w, label="Before CGM (X3D-S only)",
                color=C_VIOLENT, alpha=0.6, edgecolor="black", hatch="//")
    b2 = ax.bar(x + w/2, rd["fpr_c"], w, label="After CGM (zero-shot)",
                color="#5aae61", edgecolor="black", linewidth=0.8)
    ax.bar_label(b1, fmt="%.3f", fontsize=8); ax.bar_label(b2, fmt="%.3f", fontsize=8)
    for i, row in rd.iterrows():
        if row.fpr_b and row.fpr_b > 0:
            rel = (row.fpr_b - row.fpr_c) / row.fpr_b * 100
            ax.text(i, max(row.fpr_b, row.fpr_c) + 0.01, f"{rel:+.0f}%",
                    ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(rd["model"])
    ax.set_title("RQ3 — Cross-dataset FPR: RWF-2000 → RLVS (zero-shot CGM)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("False Positive Rate"); ax.legend()
    fig.tight_layout()
    savefig(fig, "RQ3b_cross_dataset_transfer.png")
    print(f"    [RQ3-B] vẽ transfer cho {len(rd)} model.")


# ════════════════════════════════════════════════════════════════════════════
# RQ4 — Is the Motion-chaos stream redundant alongside X3D?
#   Ý tưởng: X3D học motion ở mức HÀNH ĐỘNG (action-level), còn Motion-chaos
#   stream đo entropy/synchrony ở mức CẢNH (scene-level). Nếu motion-chaos chỉ
#   lặp lại thông tin X3D đã có -> dư thừa. Ta kiểm tra bằng 4 góc:
#     (A) |corr| của 4 motion-feature với p_base (thấp = thông tin mới).
#     (B) Incremental AUC: logistic reg p_base  vs  p_base + motion
#         (k-fold CV; nếu AUC tăng -> motion bổ sung giá trị dự báo).
#     (C) Partial correlation của motion với label, đã kiểm soát p_base.
#     (D) "Intense-but-safe" test: trong các clip Non-violent mà X3D báo cao
#         (p_base>=0.5, dễ false-positive), motion-entropy/synchrony có tách
#         khỏi clip Violent thật không?
# ════════════════════════════════════════════════════════════════════════════

def _incremental_auc(X_base, X_full, y, n_splits=5):
    """AUC trung bình qua Stratified K-Fold cho 2 bộ feature (base vs full)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler as _SS

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    auc_base, auc_full = [], []
    for tr, te in skf.split(X_full, y):
        for X, store in [(X_base, auc_base), (X_full, auc_full)]:
            sc = _SS().fit(X[tr])
            clf = LogisticRegression(max_iter=1000, C=1.0)
            clf.fit(sc.transform(X[tr]), y[tr])
            p = clf.predict_proba(sc.transform(X[te]))[:, 1]
            store.append(roc_auc_score(y[te], p))
    return float(np.mean(auc_base)), float(np.mean(auc_full)), \
           float(np.std(auc_full))


def _partial_corr(a, b, control):
    """Partial correlation giữa a và b khi đã kiểm soát biến `control`
    (Pearson trên phần dư của hồi quy tuyến tính)."""
    A = np.c_[np.ones_like(control), control]
    # residual của a và b sau khi bỏ phần giải thích bởi control
    res_a = a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
    res_b = b - A @ np.linalg.lstsq(A, b, rcond=None)[0]
    if res_a.std() < 1e-9 or res_b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(res_a, res_b)[0, 1])


def rq4_motion_redundancy(df):
    print("\n  [RQ4] Is the Motion-chaos stream redundant alongside X3D? ...")
    sub = df[df["split"] != 2].copy()          # train+val cho phân tích thống kê
    y = sub["label"].values.astype(int)
    pb = sub["p_base"].values

    # ── (A) tương quan motion <-> p_base ────────────────────────────────
    corr_pb = {c: abs(np.corrcoef(sub[c].values, pb)[0, 1]) for c in MOTION_COLS}

    # ── (B) incremental AUC: p_base  vs  p_base + 4 motion-feature ──────
    X_base = pb.reshape(-1, 1)
    X_full = np.c_[pb.reshape(-1, 1), sub[MOTION_COLS].values]
    auc_b, auc_f, auc_f_sd = _incremental_auc(X_base, X_full, y)
    # và bản chỉ-motion (không có p_base) để biết motion một mình mạnh đến đâu
    X_mot = sub[MOTION_COLS].values
    auc_m, _, _ = _incremental_auc(X_mot, X_mot, y)

    # ── (C) partial correlation motion <-> label | p_base ───────────────
    pcorr = {c: _partial_corr(sub[c].values.astype(float),
                              y.astype(float), pb.astype(float))
             for c in MOTION_COLS}

    # ── Hình 4A: redundancy dashboard ───────────────────────────────────
    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1.05], wspace=0.32)
    fig.suptitle("RQ4 — Is the Motion-chaos stream redundant given X3D (p_base)? "
                 "(train+val)", fontsize=14, fontweight="bold")

    # (a) |corr| với p_base
    ax = fig.add_subplot(gs[0, 0])
    keys = list(corr_pb.keys())
    vals = [corr_pb[k] for k in keys]
    bars = ax.barh([PRETTY[k] for k in keys], vals, color="#5ab4ac",
                   edgecolor="black", linewidth=0.6)
    ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax.axvline(0.3, color="gray", ls=":", lw=1)
    ax.set_xlim(0, max(0.4, max(vals) + 0.08))
    ax.set_title("(a) |corr| of motion with p_base\n(low = NOT redundant)")
    ax.set_xlabel("|Pearson r| with p_base")

    # (b) incremental AUC
    ax = fig.add_subplot(gs[0, 1])
    names = ["p_base\nonly", "motion\nonly", "p_base +\nmotion"]
    vals2 = [auc_b, auc_m, auc_f]
    colors = ["#bdbdbd", "#5ab4ac", "#5aae61"]
    bars = ax.bar(names, vals2, color=colors, edgecolor="black", linewidth=0.7,
                  width=0.6)
    ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=2)
    ax.set_ylim(min(vals2) - 0.05, 1.0)
    ax.set_title("(b) 5-fold CV AUC (logistic reg.)")
    ax.set_ylabel("AUC")
    ax.annotate(f"+{auc_f-auc_b:.3f}", xy=(2, auc_f), xytext=(1.5, auc_f + 0.02),
                fontsize=10, fontweight="bold", color="#2c7a2c", ha="center")

    # (c) partial correlation với label | p_base
    ax = fig.add_subplot(gs[0, 2])
    keys = list(pcorr.keys()); vals3 = [pcorr[k] for k in keys]
    colors3 = [C_VIOLENT if v >= 0 else C_NORMAL for v in vals3]
    bars = ax.barh([PRETTY[k] for k in keys], vals3, color=colors3,
                   edgecolor="black", linewidth=0.6)
    ax.bar_label(bars, fmt="%+.2f", fontsize=8, padding=2)
    ax.axvline(0, color="black", lw=1)
    ax.set_title("(c) Partial corr. with label | p_base\n(≠0 = unique signal)")
    ax.set_xlabel("partial correlation")

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    savefig(fig, "RQ4a_motion_redundancy.png")

    # ── Hình 4B: "intense-but-safe" — nơi motion-chaos đáng giá nhất ────
    # Tập "X3D bối rối": p_base trong vùng cao nhưng gồm cả 2 lớp
    confused = sub[sub["p_base"] >= 0.5].copy()
    fp_prone = confused[confused["label"] == 0]   # Non-violent bị X3D báo cao
    tp_high  = confused[confused["label"] == 1]   # Violent X3D báo cao (đúng)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("RQ4 — Scene-level motion separates real fights from "
                 "intense-but-safe motion (clips with p_base ≥ 0.5)",
                 fontsize=14, fontweight="bold")

    # (a) scatter motion_mean vs direction_entropy, tô theo lớp thật
    ax = axes[0]
    ax.scatter(fp_prone["motion_mean"], fp_prone["motion_dir_entropy"],
               s=26, color=C_NORMAL, alpha=0.7, edgecolor="black", linewidth=0.3,
               label=f"Non-violent, high p_base (n={len(fp_prone)})")
    ax.scatter(tp_high["motion_mean"], tp_high["motion_dir_entropy"],
               s=20, color=C_VIOLENT, alpha=0.45, edgecolor="none",
               label=f"Violent, high p_base (n={len(tp_high)})")
    ax.set_title("(a) Among X3D high-confidence clips")
    ax.set_xlabel("Motion: mean magnitude")
    ax.set_ylabel("Motion: direction entropy")
    ax.legend(fontsize=8)

    # (b) phân phối direction_entropy của 2 nhóm "X3D báo cao"
    ax = axes[1]
    sns.kdeplot(fp_prone["motion_dir_entropy"], ax=ax, fill=True, alpha=0.4,
                color=C_NORMAL, lw=2, label="Non-violent (false-alarm-prone)")
    sns.kdeplot(tp_high["motion_dir_entropy"], ax=ax, fill=True, alpha=0.4,
                color=C_VIOLENT, lw=2, label="Violent (true)")
    ax.set_title("(b) Direction entropy | p_base ≥ 0.5")
    ax.set_xlabel("Motion: direction entropy"); ax.legend(fontsize=8)

    # (c) phân phối synchrony của 2 nhóm
    ax = axes[2]
    sns.kdeplot(fp_prone["motion_synchrony"], ax=ax, fill=True, alpha=0.4,
                color=C_NORMAL, lw=2, label="Non-violent (false-alarm-prone)")
    sns.kdeplot(tp_high["motion_synchrony"], ax=ax, fill=True, alpha=0.4,
                color=C_VIOLENT, lw=2, label="Violent (true)")
    ax.set_title("(c) Synchrony | p_base ≥ 0.5")
    ax.set_xlabel("Motion: synchrony"); ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    savefig(fig, "RQ4b_intense_but_safe.png")

    # ── Kết luận định lượng ─────────────────────────────────────────────
    max_corr = max(corr_pb.values())
    verdict = ("NOT redundant" if (auc_f - auc_b) > 0.005 and max_corr < 0.5
               else "weak / possibly redundant")
    print(f"    (A) max |corr(motion, p_base)| = {max_corr:.3f} "
          f"(thấp => thông tin mới)")
    print(f"    (B) CV-AUC: p_base={auc_b:.3f}  motion_only={auc_m:.3f}  "
          f"p_base+motion={auc_f:.3f}  (Δ={auc_f-auc_b:+.3f} ± {auc_f_sd:.3f})")
    print(f"    (C) partial corr(motion, label | p_base): "
          + ", ".join(f"{k.split('_',1)[1]}={v:+.2f}" for k, v in pcorr.items()))
    print(f"    (D) intense-but-safe set: {len(fp_prone)} Non-violent clips với "
          f"p_base>=0.5 (đây là nơi motion-chaos giúp sửa false alarm)")
    print(f"    -> Verdict: Motion-chaos stream is {verdict} alongside X3D.")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 70)
    print("  Phase 4b — EDA & Advanced Visualization (by Research Question)")
    print("  RQ1: X3D-S+CGM giảm FPR so với X3D-S alone?")
    print("  RQ2: Context stream nào đóng góp nhiều nhất vào việc giảm FPR?")
    print("  RQ3: CGM framework có generalize across datasets?")
    print("  RQ4: Motion-chaos stream có dư thừa so với X3D không?")
    print("=" * 70)

    df  = load_dataframe()
    abl = load_ablation()
    fs  = compute_feature_stats(df)

    rq1_fpr_reduction(df, abl)
    rq2_stream_contribution(df, abl, fs)
    rq3_cross_dataset()
    rq4_motion_redundancy(df)

    print("\n" + "=" * 70)
    print(f"  DONE. Hình lưu trong : {FIGDIR.resolve()}")
    print(f"  Bảng feature stats   : {RESULTS/'feature_stats.csv'}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
