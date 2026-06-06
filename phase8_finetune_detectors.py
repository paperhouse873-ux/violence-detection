"""
Phase 8 (Hướng 2) — Fine-tune detector THỨ 2/3 để chứng minh model-agnostic
===========================================================================
Mục tiêu: gắn CGM lên các HỌ kiến trúc KHÁC X3D-S để câu "model-agnostic"
là thật (không chỉ "scale-agnostic"):
    * slowfast_r50   — CNN hai luồng (slow + fast pathway)
    * mvit_base_16x4 — Transformer video (không phải CNN)

Tái dùng đúng "công thức thành công" của X3D-S:
    proj head -> Linear(.,1) ; bỏ activation cuối ; BCEWithLogitsLoss ;
    KHÔNG sigmoid trước loss.

QUAN TRỌNG — chạy SMOKE TEST ở LOCAL (CPU) TRƯỚC khi thuê GPU:
    python phase8_finetune_detectors.py --model slowfast_r50 --smoke_test
    python phase8_finetune_detectors.py --model mvit_base_16x4 --smoke_test

  Smoke test kiểm tra 3 điều, KHÔNG cần GPU:
    (1) Head sửa đúng -> forward ra logit (không kẹt softmax/double-act).
    (2) Output KHÔNG dao động: thử overfit 1 mini-batch ~8 clip;
        loss phải GIẢM rõ rệt sau ~30 step (nếu giảm được => kiến trúc OK,
        sẽ KHÔNG bị tình trạng oscillate như 2D-CNN+LSTM/HF cũ).
    (3) In dải logit/prob để mắt thường thấy có phân tách.

  Nếu smoke test PASS -> gần như chắc chắn train full trên A100 sẽ ổn.

Train full (trên A100):
    python phase8_finetune_detectors.py --model slowfast_r50 \
        --root RWF-2000 --split split.json --epochs 20 --batch_size 8
"""

import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, confusion_matrix

sys.path.append(str(Path(__file__).parent))
from phase1_dataset import RWF2000Dataset

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT = Path("checkpoints"); CKPT.mkdir(exist_ok=True)
RESULTS = Path("results"); RESULTS.mkdir(exist_ok=True)

# SlowFast: tỉ lệ lấy mẫu giữa fast và slow pathway (chuẩn pytorchvideo = 4)
SLOWFAST_ALPHA = 4

# Số frame input theo từng họ model (khác nhau theo thiết kế pretrained):
#   slowfast_r50   cần 32 frame (fast=32, slow=32/alpha=8 -> khớp pooling kT=8)
#   mvit_base_16x4 cần 16 frame
N_FRAMES = {"slowfast_r50": 32, "mvit_base_16x4": 16}


# ════════════════════════════════════════════════════════════════════════════
# Đóng gói input theo từng họ model
# ════════════════════════════════════════════════════════════════════════════

def pack_input(videos, model_name):
    """videos: (B, 3, T, H, W). Trả về định dạng input đúng cho từng model.
    - x3d / mvit: giữ nguyên tensor (B,3,T,H,W).
    - slowfast  : list [slow_pathway, fast_pathway].
        fast = toàn bộ T frame; slow = lấy thưa T/alpha frame.
    """
    if model_name == "slowfast_r50":
        fast = videos
        T = videos.shape[2]
        idx = torch.linspace(0, T - 1, T // SLOWFAST_ALPHA).long().to(videos.device)
        slow = torch.index_select(videos, 2, idx)
        return [slow, fast]
    return videos


# ════════════════════════════════════════════════════════════════════════════
# Build model + sửa head (theo đúng công thức X3D-S)
# ════════════════════════════════════════════════════════════════════════════

def build_model(model_name):
    print(f"\n  Loading {model_name} (pretrained Kinetics-400)...")
    import pytorchvideo.models.hub as hub

    if model_name == "slowfast_r50":
        model = hub.slowfast_r50(pretrained=True)
        # Head: model.blocks[-1] = ResNetBasicHead, có .proj (Linear 2304->400)
        head = model.blocks[-1]
        in_f = head.proj.in_features
        head.proj = nn.Linear(in_f, 1)
        # bỏ activation cuối (Softmax) nếu có
        if getattr(head, "activation", None) is not None:
            head.activation = nn.Identity()
        print(f"  SlowFast head: proj {in_f}->1, activation removed")

    elif model_name == "mvit_base_16x4":
        model = hub.mvit_base_16x4(pretrained=True)
        # Head: model.head = VisionTransformerBasicHead, có .proj (Linear 768->400)
        head = model.head
        in_f = head.proj.in_features
        head.proj = nn.Linear(in_f, 1)
        if getattr(head, "activation", None) is not None:
            head.activation = nn.Identity()
        print(f"  MViT head: proj {in_f}->1, activation removed")

    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(DEVICE)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total:,}")
    return model


# ════════════════════════════════════════════════════════════════════════════
# Metrics
# ════════════════════════════════════════════════════════════════════════════

def compute_metrics(labels, probs, threshold=0.5):
    preds = [1 if p >= threshold else 0 for p in probs]
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    return {
        "accuracy": round(accuracy_score(labels, preds), 4),
        "f1": round(f1_score(labels, preds, zero_division=0), 4),
        "auc_roc": round(roc_auc_score(labels, probs), 4) if len(set(labels)) > 1 else 0.0,
        "fpr": round(fpr, 4), "fnr": round(fnr, 4),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


# ════════════════════════════════════════════════════════════════════════════
# SMOKE TEST — kiểm tra kiến trúc ở local, KHÔNG cần GPU
# ════════════════════════════════════════════════════════════════════════════

def smoke_test(model_name, root, split, n_clips=8, steps=40):
    print("=" * 68)
    print(f"SMOKE TEST — {model_name}  (overfit {n_clips} clip, {steps} step, CPU)")
    print("=" * 68)
    nf = N_FRAMES[model_name]
    ds = RWF2000Dataset(root, split, "train", n_frames=nf, augment=False)

    # lấy n_clips cân bằng nhãn nếu có thể
    labels_all = [s["label"] for s in ds.samples]
    pos = [i for i, l in enumerate(labels_all) if l == 1][:n_clips // 2]
    neg = [i for i, l in enumerate(labels_all) if l == 0][:n_clips // 2]
    pick = pos + neg
    vids = torch.stack([ds[i][0] for i in pick]).to(DEVICE)
    ys = torch.tensor([labels_all[i] for i in pick], dtype=torch.float32).to(DEVICE)
    print(f"  batch: videos={tuple(vids.shape)}  labels={ys.tolist()}")

    model = build_model(model_name)
    model.train()
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    x = pack_input(vids, model_name)

    # (1) forward 1 lần kiểm tra logit
    with torch.no_grad():
        logit0 = model(pack_input(vids, model_name)).squeeze(1)
    print(f"\n  [1] forward OK. logit ban đầu: "
          f"min={logit0.min():.3f} max={logit0.max():.3f} "
          f"mean={logit0.mean():.3f}")
    p0 = torch.sigmoid(logit0)
    if p0.std() < 1e-4:
        print("      CẢNH BÁO: prob gần như hằng số -> nghi kẹt activation.")

    # (2) overfit mini-batch — loss phải giảm
    # Lưu ý: SlowFast biến đổi list input TẠI CHỖ -> phải pack lại mỗi forward.
    print(f"\n  [2] thử overfit {n_clips} clip:")
    losses = []
    t0 = time.time()
    for st in range(steps):
        opt.zero_grad()
        logit = model(pack_input(vids, model_name)).squeeze(1)
        loss = crit(logit, ys)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if st % 5 == 0 or st == steps - 1:
            with torch.no_grad():
                p = torch.sigmoid(logit)
            acc = ((p >= 0.5).float() == ys).float().mean().item()
            print(f"      step {st:02d}: loss={loss.item():.4f}  acc={acc:.2f}  "
                  f"p[min={p.min():.2f},max={p.max():.2f}]")
    dt = time.time() - t0

    drop = losses[0] - losses[-1]
    print(f"\n  [3] loss {losses[0]:.4f} -> {losses[-1]:.4f}  (giảm {drop:.4f}) "
          f"| {dt:.1f}s cho {steps} step")
    # phán quyết
    ok_drop = losses[-1] < losses[0] - 0.15 and losses[-1] < 0.55
    print("=" * 68)
    if ok_drop:
        print(f"  PASS: {model_name} overfit được mini-batch -> kiến trúc + head OK.")
        print("        Không có dấu hiệu oscillate. An toàn để train full trên GPU.")
    else:
        print(f"  FAIL/NGỜ: loss không giảm đủ. KHÔNG nên train full vội.")
        print("        Cần kiểm tra: head/activation, learning rate, pack_input.")
    print("=" * 68)
    return ok_drop


# ════════════════════════════════════════════════════════════════════════════
# TRAIN FULL (cho A100)
# ════════════════════════════════════════════════════════════════════════════

def evaluate(model, loader, model_name, crit):
    model.eval()
    ys, ps = [], []
    tot = 0.0
    with torch.no_grad():
        for videos, labels in loader:
            videos = videos.to(DEVICE)
            x = pack_input(videos, model_name)
            logit = model(x).squeeze(1)
            tot += crit(logit, labels.float().to(DEVICE)).item()
            ps.extend(torch.sigmoid(logit).cpu().tolist())
            ys.extend(labels.tolist())
    m = compute_metrics(ys, ps)
    m["loss"] = round(tot / len(loader), 4)
    return m


def train_full(args):
    print(f"\n  Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    nf = N_FRAMES[args.model]
    tr = RWF2000Dataset(args.root, args.split, "train", n_frames=nf, augment=True)
    va = RWF2000Dataset(args.root, args.split, "val", n_frames=nf, augment=False)
    nw = args.num_workers
    trl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=nw, pin_memory=True)
    val = DataLoader(va, batch_size=args.batch_size, shuffle=False, num_workers=nw, pin_memory=True)

    model = build_model(args.model)
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)

    best_f1, hist = 0.0, []
    for ep in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); tot = 0.0; ys = []; ps = []
        for videos, labels in trl:
            videos = videos.to(DEVICE); yb = labels.float().to(DEVICE)
            x = pack_input(videos, args.model)
            opt.zero_grad()
            logit = model(x).squeeze(1)
            loss = crit(logit, yb)
            loss.backward(); opt.step()
            tot += loss.item()
            ps.extend(torch.sigmoid(logit.detach()).cpu().tolist()); ys.extend(labels.tolist())
        trm = compute_metrics(ys, ps); trm["loss"] = round(tot / len(trl), 4)
        vam = evaluate(model, val, args.model, crit)
        sch.step(vam["f1"])
        print(f"  E{ep:02d} ({time.time()-t0:.0f}s) train f1={trm['f1']:.4f} "
              f"fpr={trm['fpr']:.4f} | val f1={vam['f1']:.4f} fpr={vam['fpr']:.4f} "
              f"auc={vam['auc_roc']:.4f}")
        hist.append({"epoch": ep, "train": trm, "val": vam})
        if vam["f1"] > best_f1:
            best_f1 = vam["f1"]
            torch.save({"epoch": ep, "model_state_dict": model.state_dict(),
                        "val_metrics": vam, "model_name": args.model},
                       CKPT / f"{args.model}_best.pth")
            print(f"    ** saved best (val F1={best_f1:.4f}) **")
    with open(RESULTS / f"{args.model}_history.json", "w") as f:
        json.dump(hist, f, indent=2)
    print(f"\n  DONE. best val F1={best_f1:.4f} -> checkpoints/{args.model}_best.pth")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["slowfast_r50", "mvit_base_16x4"])
    ap.add_argument("--root", default="RWF-2000")
    ap.add_argument("--split", default="split.json")
    ap.add_argument("--smoke_test", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--smoke_clips", type=int, default=8)
    ap.add_argument("--smoke_steps", type=int, default=40)
    args = ap.parse_args()

    if args.smoke_test:
        smoke_test(args.model, args.root, args.split,
                   n_clips=args.smoke_clips, steps=args.smoke_steps)
    else:
        train_full(args)
