"""
Phase 6b — In-domain CGM trên RLVS (đối chứng với zero-shot)
============================================================
Câu hỏi (RQ3 mở rộng): Gate collapse ở zero-shot (Δα≈0.0006) là do BẢN CHẤT
của CGM, hay chỉ do DOMAIN SHIFT? Nếu train CGM ngay trên RLVS (X3D-S vẫn
frozen, chỉ học gate + ctx head trên context-features của RLVS) thì gate có
"sống lại" không?

Thiết kế CHẶT để sống sót phản biện:
  * Tách RLVS thành train/val/test phân tầng (stratified). TEST không bao giờ
    được dùng để train CGM hay fit scaler -> không rò rỉ.
  * Scaler fit CHỈ trên train RLVS (in-domain, khác zero-shot dùng scaler RWF).
  * Early-stop theo val.
  * Báo cáo TRÊN TEST: ngưỡng 0.5, FPR @ cùng recall (matched-recall),
    Youden, AUC, và Δα (gate có phân biệt lớp không).
  * Lặp lại nhiều SEED -> mean ± std (tránh cherry-pick).

So sánh 3 cấu hình trên cùng test set:
  (a) baseline  : p_base (X3D-S frozen, train trên RWF)
  (b) zero-shot : CGM train trên RWF  (checkpoints/cgm_e4.pth + scaler RWF)
  (c) in-domain : CGM train trên RLVS-train (script này)

Chạy:
  python phase6b_train_rlvs.py                 # 5 seeds, mặc định
  python phase6b_train_rlvs.py --seeds 10 --epochs 300
"""

import sys
import json
import pickle
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RLVS_CACHE = Path("cache/rlvs")
CKPT = Path("checkpoints")
RESULTS = Path("results"); RESULTS.mkdir(exist_ok=True)


# ── CGM (giống Phase 4 / Phase 6) ────────────────────────────────────────────
class ContextGatingModule(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1), nn.Sigmoid())
        self.ctx = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, x, p_base):
        alpha = self.gate(x).squeeze(1)
        p_ctx = self.ctx(x).squeeze(1)
        return alpha * p_base + (1 - alpha) * p_ctx, alpha, p_ctx


# ── metrics helpers ──────────────────────────────────────────────────────────
def metrics_at_threshold(y, probs, thr=0.5):
    preds = (np.asarray(probs) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    return dict(fpr=fpr, fnr=1 - tpr, tpr=tpr,
                acc=(tp + tn) / len(y),
                f1=2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)


def fpr_at_recall(y, probs, target_tpr):
    """FPR nhỏ nhất khi TPR >= target_tpr (quét toàn ROC)."""
    fpr, tpr, thr = roc_curve(y, probs)
    ok = tpr >= target_tpr - 1e-9
    if not ok.any():
        return None
    return float(fpr[np.argmax(ok)])


# ── train CGM trên 1 split RLVS ──────────────────────────────────────────────
def train_one(Xtr, ytr, pbtr, Xva, yva, pbva, input_dim,
              epochs=300, lr=1e-3, patience=30, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    model = ContextGatingModule(input_dim).to(DEVICE)
    crit = nn.BCELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max",
                                                       factor=0.5, patience=10)

    def t(a): return torch.tensor(a).to(DEVICE)
    Xtr, ytr, pbtr = t(Xtr), t(ytr), t(pbtr)
    Xva, yva, pbva = t(Xva), t(yva), t(pbva)

    best_auc, best_state, wait = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        pf, _, _ = model(Xtr, pbtr)
        loss = crit(pf, ytr)
        opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pfv, _, _ = model(Xva, pbva)
        try:
            vauc = roc_auc_score(yva.cpu().numpy(), pfv.cpu().numpy())
        except ValueError:
            vauc = 0.0
        sched.step(vauc)
        if vauc > best_auc:
            best_auc, wait = vauc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if wait >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model


def run_seed(p_base, Z, y, seed, epochs):
    # 60/20/20 stratified: train / val / test
    idx = np.arange(len(y))
    tr_idx, te_idx = train_test_split(idx, test_size=0.20, stratify=y,
                                      random_state=seed)
    tr_idx, va_idx = train_test_split(tr_idx, test_size=0.25, stratify=y[tr_idx],
                                      random_state=seed)  # 0.25*0.8 = 0.2

    X = np.concatenate([p_base.reshape(-1, 1), Z], axis=1)
    scaler = StandardScaler().fit(X[tr_idx])          # fit CHỈ trên train RLVS
    Xn = scaler.transform(X).astype(np.float32)

    model = train_one(
        Xn[tr_idx], y[tr_idx].astype(np.float32), p_base[tr_idx].astype(np.float32),
        Xn[va_idx], y[va_idx].astype(np.float32), p_base[va_idx].astype(np.float32),
        input_dim=X.shape[1], epochs=epochs, seed=seed)

    # ---- đánh giá TRÊN TEST ----
    with torch.no_grad():
        pf, alpha, p_ctx = model(torch.tensor(Xn[te_idx]).to(DEVICE),
                                 torch.tensor(p_base[te_idx].astype(np.float32)).to(DEVICE))
    pf = pf.cpu().numpy(); alpha = alpha.cpu().numpy()
    yte = y[te_idx].astype(int)
    pbte = p_base[te_idx]

    b05 = metrics_at_threshold(yte, pbte, 0.5)
    a05 = metrics_at_threshold(yte, pf, 0.5)
    auc_b = roc_auc_score(yte, pbte)
    auc_a = roc_auc_score(yte, pf)
    # matched recall = recall của baseline @0.5
    fpr_b_mr = fpr_at_recall(yte, pbte, b05["tpr"])
    fpr_a_mr = fpr_at_recall(yte, pf,   b05["tpr"])
    d_alpha = abs(alpha[yte == 1].mean() - alpha[yte == 0].mean())

    return dict(
        fpr_b=b05["fpr"], fpr_a=a05["fpr"],
        fnr_b=b05["fnr"], fnr_a=a05["fnr"],
        acc_b=b05["acc"], acc_a=a05["acc"],
        f1_b=b05["f1"],   f1_a=a05["f1"],
        auc_b=auc_b,      auc_a=auc_a,
        fpr_b_mr=fpr_b_mr, fpr_a_mr=fpr_a_mr,
        d_alpha=d_alpha,
        alpha_v=float(alpha[yte == 1].mean()),
        alpha_n=float(alpha[yte == 0].mean()),
    )


def agg(runs, key):
    vals = [r[key] for r in runs if r[key] is not None]
    return float(np.mean(vals)), float(np.std(vals))


def main(args):
    y = np.load(RLVS_CACHE / "labels.npy").astype(int)
    p_base = np.load(RLVS_CACHE / "p_base.npy")
    Z = np.concatenate([np.load(RLVS_CACHE / "z_crowd.npy"),
                        np.load(RLVS_CACHE / "z_light.npy"),
                        np.load(RLVS_CACHE / "z_motion.npy")], axis=1)

    print("=" * 70)
    print(f"PHASE 6b — In-domain CGM trên RLVS  ({args.seeds} seeds, test split 20%)")
    print(f"Device={DEVICE}  | clips={len(y)}  | input_dim={1 + Z.shape[1]}")
    print("=" * 70)

    runs = [run_seed(p_base, Z, y, s, args.epochs) for s in range(args.seeds)]

    def line(name, kb, ka, pct=False):
        mb, sb = agg(runs, kb); ma, sa = agg(runs, ka)
        d = ma - mb
        arrow = ""
        print(f"  {name:18}{mb:6.4f}±{sb:.3f}   {ma:6.4f}±{sa:.3f}   Δ={d:+.4f}{arrow}")

    print("\n  Trên TEST set (mean ± std qua seeds):")
    print(f"  {'metric':18}{'baseline':>14}   {'+CGM(RLVS)':>14}")
    line("FPR @0.5",      "fpr_b", "fpr_a")
    line("FNR @0.5",      "fnr_b", "fnr_a")
    line("Accuracy @0.5", "acc_b", "acc_a")
    line("F1 @0.5",       "f1_b",  "f1_a")
    line("AUC-ROC",       "auc_b", "auc_a")
    line("FPR @matched-recall", "fpr_b_mr", "fpr_a_mr")

    da_m, da_s = agg(runs, "d_alpha")
    av_m, _ = agg(runs, "alpha_v"); an_m, _ = agg(runs, "alpha_n")
    print(f"\n  |Δα| (gate phân biệt lớp) = {da_m:.4f} ± {da_s:.4f}")
    print(f"  α_violent={av_m:.4f}  α_normal={an_m:.4f}")

    # ---- phán quyết tự động dựa trên 2 tiêu chí THẬT ----
    auc_mb, _ = agg(runs, "auc_b"); auc_ma, _ = agg(runs, "auc_a")
    mr_b, _ = agg(runs, "fpr_b_mr"); mr_a, _ = agg(runs, "fpr_a_mr")
    print("\n" + "=" * 70)
    print("PHÁN QUYẾT (dựa trên matched-recall FPR + AUC + Δα, KHÔNG dùng FPR@0.5):")
    cond_auc = auc_ma >= auc_mb - 0.002
    cond_mr  = mr_a < mr_b - 1e-4
    cond_gate = da_m > 0.02
    print(f"  [{'OK' if cond_mr  else 'XX'}] FPR@matched-recall giảm thật: "
          f"{mr_b:.4f} -> {mr_a:.4f}")
    print(f"  [{'OK' if cond_auc else 'XX'}] AUC không giảm: {auc_mb:.4f} -> {auc_ma:.4f}")
    print(f"  [{'OK' if cond_gate else 'XX'}] Gate sống (|Δα|>0.02): {da_m:.4f}")
    if cond_mr and cond_auc and cond_gate:
        print("\n  => IN-DOMAIN CGM HOẠT ĐỘNG THẬT. Narrative 'gate cần domain data' "
              "được\n     chứng minh. Đưa vào paper được.")
    elif cond_gate and (cond_mr or cond_auc):
        print("\n  => Tín hiệu TÍCH CỰC nhưng chưa trọn vẹn. Cần xem kỹ từng tiêu chí.")
    else:
        print("\n  => Ngay cả in-domain CGM cũng KHÔNG cải thiện thật. Vấn đề nằm ở "
              "feature\n     context (ít thông tin) chứ không chỉ domain shift.")
    print("=" * 70)

    out = dict(
        n_seeds=args.seeds, n_clips=int(len(y)), test_frac=0.20,
        baseline={k.replace("_b", ""): agg(runs, k)[0]
                  for k in ["fpr_b", "fnr_b", "acc_b", "f1_b", "auc_b", "fpr_b_mr"]},
        in_domain_cgm={k.replace("_a", ""): agg(runs, k)[0]
                       for k in ["fpr_a", "fnr_a", "acc_a", "f1_a", "auc_a", "fpr_a_mr"]},
        delta_alpha=da_m, alpha_violent=av_m, alpha_normal=an_m,
        per_seed=runs,
    )
    def to_native(o):
        if isinstance(o, dict):
            return {k: to_native(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [to_native(v) for v in o]
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        return o

    with open(RESULTS / "phase6b_indomain_results.json", "w") as f:
        json.dump(to_native(out), f, indent=2)
    print("  saved -> results/phase6b_indomain_results.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()
    main(args)
