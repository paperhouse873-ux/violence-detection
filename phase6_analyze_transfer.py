"""
Phase 6b — Phân tích sâu kết quả zero-shot transfer (RLVS)
==========================================================
Mục tiêu: kiểm chứng xem lợi ích FPR của CGM trên RLVS là THẬT (gating
thích nghi) hay chỉ do RECALIBRATION (dịch điểm vận hành ở ngưỡng 0.5).

So sánh công bằng:
  1) FPR tại CÙNG mức recall (TPR) — quét ngưỡng để khớp FNR của baseline.
  2) Toàn bộ ROC (AUC) — thước đo độc lập ngưỡng.
  3) Threshold tối ưu (Youden's J) cho từng model.
  4) Phân bố alpha theo lớp — gating có "mở/đóng" theo ngữ cảnh không?

Chỉ đọc cache có sẵn (cache/rlvs/*.npy) + checkpoint CGM. Không trích lại.
"""

import sys
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DEVICE = torch.device("cpu")
RLVS_CACHE = Path("cache/rlvs")
CKPT = Path("checkpoints")


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


def metrics_at_threshold(labels, probs, thr):
    preds = (np.asarray(probs) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    acc = (tp + tn) / len(labels)
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return dict(thr=thr, fpr=fpr, fnr=1 - tpr, tpr=tpr, acc=acc, f1=f1,
                tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn))


def fpr_at_recall(labels, probs, target_tpr):
    """FPR nhỏ nhất đạt được khi TPR >= target_tpr (quét toàn ROC)."""
    fpr, tpr, thr = roc_curve(labels, probs)
    ok = tpr >= target_tpr
    if not ok.any():
        return None, None
    i = np.argmax(ok)  # điểm đầu tiên đạt target_tpr
    return fpr[i], thr[i] if i < len(thr) else thr[-1]


def main():
    labels = np.load(RLVS_CACHE / "labels.npy")
    p_base = np.load(RLVS_CACHE / "p_base.npy")
    z_crowd = np.load(RLVS_CACHE / "z_crowd.npy")
    z_light = np.load(RLVS_CACHE / "z_light.npy")
    z_motion = np.load(RLVS_CACHE / "z_motion.npy")

    ck = torch.load(CKPT / "cgm_e4.pth", map_location=DEVICE, weights_only=False)
    cgm = ContextGatingModule(ck["input_dim"]).to(DEVICE)
    cgm.load_state_dict(ck["model_state_dict"]); cgm.eval()
    scaler = pickle.load(open(CKPT / "cgm_scaler.pkl", "rb"))

    X = np.concatenate([p_base.reshape(-1, 1), z_crowd, z_light, z_motion], axis=1)
    Xn = scaler.transform(X).astype(np.float32)
    with torch.no_grad():
        pf, alpha, p_ctx = cgm(torch.tensor(Xn), torch.tensor(p_base.astype(np.float32)))
    p_final = pf.numpy(); alpha = alpha.numpy(); p_ctx = p_ctx.numpy()

    y = labels.astype(int)
    auc_b = roc_auc_score(y, p_base)
    auc_a = roc_auc_score(y, p_final)

    print("=" * 68)
    print("PHÂN TÍCH ZERO-SHOT TRANSFER — RLVS")
    print("=" * 68)

    # ---- (A) Ngưỡng cố định 0.5 (như báo cáo gốc) ----
    b05 = metrics_at_threshold(y, p_base, 0.5)
    a05 = metrics_at_threshold(y, p_final, 0.5)
    print("\n[A] Ngưỡng 0.5 cố định (so sánh GỐC):")
    print(f"  {'':12}{'baseline':>12}{'+CGM':>12}")
    for k in ["fpr", "fnr", "acc", "f1"]:
        print(f"  {k:12}{b05[k]:>12.4f}{a05[k]:>12.4f}")
    print(f"  {'AUC':12}{auc_b:>12.4f}{auc_a:>12.4f}")

    # ---- (B) So tại CÙNG recall = recall của baseline @0.5 ----
    target_tpr = b05["tpr"]
    fpr_b_r, _ = fpr_at_recall(y, p_base, target_tpr)
    fpr_a_r, thr_a = fpr_at_recall(y, p_final, target_tpr)
    print(f"\n[B] FPR tại CÙNG recall (TPR={target_tpr:.4f}) — so sánh CÔNG BẰNG:")
    print(f"  baseline FPR = {fpr_b_r:.4f}")
    print(f"  +CGM     FPR = {fpr_a_r:.4f}  (thr≈{thr_a:.3f})")
    if fpr_a_r is not None and fpr_b_r is not None:
        d = fpr_b_r - fpr_a_r
        rel = d / fpr_b_r * 100 if fpr_b_r else 0
        verdict = "CGM giảm FPR ở cùng recall -> lợi ích THẬT" if d > 1e-4 else \
                  ("Tương đương -> lợi ích @0.5 chủ yếu do RECALIBRATION" if abs(d) <= 1e-4
                   else "CGM TỆ hơn ở cùng recall -> lợi ích @0.5 chỉ do dịch ngưỡng")
        print(f"  Δ={d:+.4f} ({rel:+.1f}%)  -> {verdict}")

    # ---- (C) Threshold tối ưu Youden cho mỗi model ----
    def youden(probs):
        fpr, tpr, thr = roc_curve(y, probs)
        j = np.argmax(tpr - fpr)
        return thr[j], fpr[j], tpr[j]
    tb, fb, rb = youden(p_base)
    ta, fa, ra = youden(p_final)
    print("\n[C] Tại ngưỡng tối ưu (Youden's J):")
    print(f"  baseline: thr={tb:.3f}  FPR={fb:.4f}  TPR={rb:.4f}")
    print(f"  +CGM    : thr={ta:.3f}  FPR={fa:.4f}  TPR={ra:.4f}")

    # ---- (D) Phân bố alpha ----
    print("\n[D] Cổng gating alpha (mong đợi: khác nhau giữa 2 lớp):")
    av, an = alpha[y == 1], alpha[y == 0]
    print(f"  alpha mean   = {alpha.mean():.4f}  (std={alpha.std():.4f})")
    print(f"  alpha violent= {av.mean():.4f}  (std={av.std():.4f})")
    print(f"  alpha normal = {an.mean():.4f}  (std={an.std():.4f})")
    print(f"  |Δ alpha|    = {abs(av.mean()-an.mean()):.4f}  "
          f"(càng lớn = gating càng phân biệt ngữ cảnh)")
    print(f"  p_ctx mean   = {p_ctx.mean():.4f}  "
          f"(violent={p_ctx[y==1].mean():.4f}, normal={p_ctx[y==0].mean():.4f})")

    # ---- (E) Kết luận tự động ----
    print("\n" + "=" * 68)
    print("KẾT LUẬN")
    print("=" * 68)
    if fpr_a_r is not None and (fpr_b_r - fpr_a_r) > 1e-4:
        print("  + Lợi ích FPR là THẬT (giữ ở cùng recall). An tâm đưa vào paper,")
        print("    nên báo cáo theo 'FPR @ matched recall' thay vì @0.5.")
    else:
        print("  ! Ở cùng recall, CGM KHÔNG tốt hơn baseline. Cải thiện @0.5")
        print("    chủ yếu do recalibration. AUC giảm củng cố điều này.")
        print("    -> Cần khung trình bày trung thực (xem gợi ý bên dưới).")
    print()


if __name__ == "__main__":
    main()
