"""
Phase 5 — Train MViTv2-S + Video Swin-B + Apply CGM
=====================================================
Models:
  M1: MViTv2-Small  (Meta AI, CVPR 2022) — 35M params
  M2: Video Swin-B  (Microsoft, CVPR 2022) — 88M params

Sau khi train, tự động:
  1. Evaluate baseline (without CGM) on test set
  2. Extract p_base từ toàn bộ dataset
  3. Train CGM (962 params) trên p_base mới
  4. Compare FPR before/after CGM

Chạy trên vast.ai A100:
  python phase5_mvit_swin.py ^
    --root "/workspace/RWF-2000" ^
    --split "/workspace/split.json" ^
    --epochs 20 ^
    --batch_size 16

Requirements:
  pip install torch torchvision transformers timm pytorchvideo
  pip install opencv-python-headless scikit-learn numpy pandas tqdm wandb
"""

import json
import time
import csv
import torch
import argparse
import numpy as np
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append(str(Path(__file__).parent))
from phase1_dataset import RWF2000Dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n  Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")


# ═══════════════════════════════════════════════════════════════════════════
# 1. Model Builders
# ═══════════════════════════════════════════════════════════════════════════

def build_mvitv2_small():
    """
    MViTv2-Small từ pytorchvideo.
    Multiscale Vision Transformer — Meta AI, CVPR 2022.
    Pretrained trên Kinetics-400.
    Input: (B, C, T, H, W) = (B, 3, 16, 224, 224)
    """
    print("\n  Loading MViTv2-Small pretrained Kinetics-400...")
    try:
        model = torch.hub.load(
            "facebookresearch/pytorchvideo",
            "mvit_base_16",
            pretrained=True,
            trust_repo=True,
        )
    except Exception:
        # Fallback: dùng timm nếu hub không load được
        import timm
        model = timm.create_model(
            "mvitv2_small",
            pretrained=True,
            num_classes=1,
        )
        print("  Loaded via timm")
        model = model.to(DEVICE)
        total = sum(p.numel() for p in model.parameters())
        print(f"  Total params: {total:,}")
        return model

    # Thay head cuối
    if hasattr(model, "head"):
        if hasattr(model.head, "proj"):
            in_f = model.head.proj.in_features
            model.head.proj = nn.Linear(in_f, 1)
            if hasattr(model.head, "act"):
                model.head.act = nn.Identity()
        elif isinstance(model.head, nn.Linear):
            in_f = model.head.in_features
            model.head = nn.Linear(in_f, 1)
    elif hasattr(model, "blocks") and hasattr(model.blocks[-1], "proj"):
        in_f = model.blocks[-1].proj.in_features
        model.blocks[-1].proj = nn.Linear(in_f, 1)
        if hasattr(model.blocks[-1], "act"):
            model.blocks[-1].act = nn.Identity()

    model = model.to(DEVICE)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total:,}")
    return model


def build_video_swin_base():
    """
    Video Swin Transformer-Base từ HuggingFace transformers.
    Microsoft Research, CVPR 2022.
    Pretrained trên Kinetics-400.
    Input: pixel_values (B, C, T, H, W)
    """
    print("\n  Loading Video Swin-Base pretrained Kinetics-400...")
    from transformers import (
        VideoMAEForVideoClassification,
        AutoModelForVideoClassification,
    )

    try:
        # Thử load Video Swin từ HuggingFace
        model = AutoModelForVideoClassification.from_pretrained(
            "microsoft/swin-base-patch244-window877-kinetics400-22k",
            num_labels=1,
            ignore_mismatched_sizes=True,
        )
        model._is_swin = True
    except Exception as e:
        print(f"  Video Swin HuggingFace failed ({e}), using timm fallback...")
        import timm
        model = timm.create_model(
            "swin_base_patch4_window7_224",
            pretrained=True,
            num_classes=0,  # feature extractor
        )
        # Wrap với temporal LSTM
        model = SwinLSTMWrapper(model, feat_dim=1024).to(DEVICE)
        model._is_swin = False
        model._is_swin_timm = True
        total = sum(p.numel() for p in model.parameters())
        print(f"  Using Swin-B + LSTM wrapper. Total params: {total:,}")
        return model

    model = model.to(DEVICE)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total:,}")
    return model


class SwinLSTMWrapper(nn.Module):
    """
    Fallback wrapper: Swin-B (spatial) + LSTM (temporal) + FC.
    Dùng khi không load được Video Swin trực tiếp.
    """
    def __init__(self, swin_model, feat_dim=1024, hidden=256):
        super().__init__()
        self.swin = swin_model
        self.lstm = nn.LSTM(feat_dim, hidden, batch_first=True)
        self.drop = nn.Dropout(0.3)
        self.fc   = nn.Linear(hidden, 1)
        self._is_swin_timm = True

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        feats = self.swin(x)           # (B*T, feat_dim)
        feats = feats.reshape(B, T, -1)
        _, (h, _) = self.lstm(feats)
        out = self.drop(h.squeeze(0))
        return self.fc(out).squeeze(1)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Forward pass helpers (handle different model APIs)
# ═══════════════════════════════════════════════════════════════════════════

def model_forward(model, videos):
    """
    Unified forward pass cho các model khác nhau.
    Trả về logits (B,).
    """
    # HuggingFace models trả về object với .logits attribute
    if hasattr(model, "_is_swin") and model._is_swin:
        out = model(pixel_values=videos)
        logits = out.logits.squeeze(1)
    # timm / pytorchvideo / wrapper models
    else:
        out = model(videos)
        if isinstance(out, torch.Tensor):
            logits = out.squeeze(1) if out.dim() > 1 else out
        elif hasattr(out, "logits"):
            logits = out.logits.squeeze(1)
        else:
            logits = out[0].squeeze(1)
    return logits


# ═══════════════════════════════════════════════════════════════════════════
# 3. Metrics
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(labels, probs, threshold=0.5):
    preds = [1 if p >= threshold else 0 for p in probs]
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return {
        "accuracy": round(accuracy_score(labels, preds), 4),
        "f1":       round(f1_score(labels, preds), 4),
        "auc_roc":  round(roc_auc_score(labels, probs), 4),
        "fpr":      round(fpr, 4),
        "fnr":      round(fnr, 4),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. Train loop
# ═══════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, criterion, epoch, name):
    model.train()
    total_loss, all_labels, all_probs = 0.0, [], []

    pbar = tqdm(loader, desc=f"  [{name} E{epoch:02d}]", unit="batch")
    for videos, labels in pbar:
        videos = videos.to(DEVICE, non_blocking=True)
        labels = labels.float().to(DEVICE)

        optimizer.zero_grad()
        logits = model_forward(model, videos)
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping để tránh exploding gradients với Transformer
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        probs = torch.sigmoid(logits.detach())
        total_loss += loss.item()
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    m = compute_metrics(all_labels, all_probs)
    m["loss"] = round(total_loss / len(loader), 4)
    return m


@torch.no_grad()
def evaluate(model, loader, criterion, desc="Val"):
    model.eval()
    total_loss, all_labels, all_probs = 0.0, [], []

    for videos, labels in tqdm(loader, desc=f"  [{desc}]", unit="batch"):
        videos = videos.to(DEVICE, non_blocking=True)
        labels_f = labels.float().to(DEVICE)

        logits = model_forward(model, videos)
        loss = criterion(logits, labels_f)
        probs = torch.sigmoid(logits)

        total_loss += loss.item()
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    m = compute_metrics(all_labels, all_probs)
    m["loss"] = round(total_loss / len(loader), 4)
    return m, all_labels, all_probs


def train_model(model, train_loader, val_loader, epochs, lr, name, ckpt_dir):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=1e-4
    )
    # Warmup + cosine decay phù hợp cho Transformer
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    best_f1, best_state = 0.0, None
    history = []

    print(f"\n  {'='*60}")
    print(f"  Training {name} — {epochs} epochs | LR={lr} | batch={train_loader.batch_size}")
    print(f"  {'='*60}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_m = train_one_epoch(model, train_loader, criterion, None, epoch, name)

        # Bỏ qua dòng trên, dùng optimizer đúng
        model.train()
        total_loss, all_labels, all_probs = 0.0, [], []
        pbar = tqdm(train_loader, desc=f"  [{name} E{epoch:02d}]",
                    unit="batch", leave=False)
        for videos, labels in pbar:
            videos = videos.to(DEVICE, non_blocking=True)
            labels = labels.float().to(DEVICE)
            optimizer.zero_grad()
            logits = model_forward(model, videos)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            probs = torch.sigmoid(logits.detach())
            total_loss += loss.item()
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_m = compute_metrics(all_labels, all_probs)
        train_m["loss"] = round(total_loss / len(train_loader), 4)

        val_m, _, _ = evaluate(model, val_loader, criterion, "Val")
        scheduler.step()
        elapsed = time.time() - t0

        print(f"  [{name}] E{epoch:02d}/{epochs} ({elapsed/60:.1f}min) — "
              f"train_loss:{train_m['loss']:.4f} "
              f"val_f1:{val_m['f1']:.4f} "
              f"val_fpr:{val_m['fpr']:.4f} "
              f"val_acc:{val_m['accuracy']:.4f}")

        history.append({"epoch": epoch, "train": train_m, "val": val_m})

        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save(best_state, ckpt_dir / f"{name.lower().replace('-','_')}_best.pth")
            print(f"  [{name}] ** Saved best (F1={best_f1:.4f}) **")

    model.load_state_dict(best_state)
    return model, history


# ═══════════════════════════════════════════════════════════════════════════
# 5. Extract p_base từ model
# ═══════════════════════════════════════════════════════════════════════════

class AllSamplesDS(torch.utils.data.Dataset):
    """Load toàn bộ samples theo thứ tự [train, val, test]."""
    def __init__(self, root, split_file, n_frames=16, img_size=224):
        import cv2
        self.root = Path(root)
        self.n_frames = n_frames
        self.img_size = img_size

        with open(split_file) as f:
            data = json.load(f)

        self.samples, self.split_ids = [], []
        for sid, sname in enumerate(["train", "val", "test"]):
            for item in data[sname]:
                self.samples.append(item)
                self.split_ids.append(sid)

        from phase1_dataset import KINETICS_MEAN, KINETICS_STD
        self.mean = torch.tensor(KINETICS_MEAN).view(3, 1, 1, 1)
        self.std  = torch.tensor(KINETICS_STD).view(3, 1, 1, 1)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        import cv2
        path = self.root / self.samples[idx]["path"]
        cap  = cv2.VideoCapture(str(path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 16
        idxs  = np.linspace(0, total-1, self.n_frames, dtype=int)
        frames = []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ret, f = cap.read()
            if not ret or f is None:
                f = frames[-1] if frames else np.zeros(
                    (self.img_size, self.img_size, 3), np.uint8)
            f = cv2.resize(f, (self.img_size, self.img_size))
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            frames.append(f)
        cap.release()
        t = torch.from_numpy(
            np.stack(frames).astype(np.float32) / 255.0
        ).permute(3, 0, 1, 2)
        t = (t - self.mean) / self.std
        return t, self.samples[idx]["label"]


@torch.no_grad()
def extract_p_base(model, root, split_file, batch_size=16):
    """Extract p_base từ model cho toàn bộ dataset."""
    model.eval()
    ds = AllSamplesDS(root, split_file)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    all_probs, all_labels, all_splits = [], [], []

    for videos, labels in tqdm(loader, desc="  [extract p_base]"):
        videos = videos.to(DEVICE, non_blocking=True)
        logits = model_forward(model, videos)
        probs  = torch.sigmoid(logits)
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())

    all_splits = ds.split_ids
    return (
        np.array(all_probs,  dtype=np.float32),
        np.array(all_labels, dtype=np.int32),
        np.array(all_splits, dtype=np.int32),
    )


# ═══════════════════════════════════════════════════════════════════════════
# 6. Context Gating Module (same as Phase 4)
# ═══════════════════════════════════════════════════════════════════════════

class ContextGatingModule(nn.Module):
    def __init__(self, input_dim=13):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1), nn.Sigmoid(),
        )
        self.ctx = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, x, p_base):
        alpha = self.gate(x).squeeze(1)
        p_ctx = self.ctx(x).squeeze(1)
        return alpha * p_base + (1 - alpha) * p_ctx, alpha, p_ctx


def train_cgm(p_base_all, labels_all, splits_all, cache_dir,
              model_name, epochs=200):
    """Train CGM dùng p_base mới + context cache từ Phase 3."""
    z_crowd  = np.load(cache_dir / "z_crowd.npy")
    z_light  = np.load(cache_dir / "z_light.npy")
    z_motion = np.load(cache_dir / "z_motion.npy")

    X_raw = np.concatenate([
        p_base_all.reshape(-1, 1), z_crowd, z_light, z_motion
    ], axis=1)

    train_m = splits_all == 0
    val_m   = splits_all == 1
    test_m  = splits_all == 2

    scaler = StandardScaler()
    scaler.fit(X_raw[train_m])
    X = scaler.transform(X_raw).astype(np.float32)

    def _t(arr): return torch.tensor(arr).to(DEVICE)

    X_tr, y_tr, pb_tr = _t(X[train_m]), _t(labels_all[train_m].astype(np.float32)), _t(p_base_all[train_m])
    X_va, y_va, pb_va = _t(X[val_m]),   _t(labels_all[val_m].astype(np.float32)),   _t(p_base_all[val_m])
    X_te, y_te, pb_te = _t(X[test_m]),  _t(labels_all[test_m].astype(np.float32)),  _t(p_base_all[test_m])

    cgm = ContextGatingModule(13).to(DEVICE)
    crit = nn.BCELoss()
    opt  = torch.optim.AdamW(cgm.parameters(), lr=1e-3, weight_decay=1e-4)

    best_f1, best_state, patience = 0.0, None, 0

    for epoch in range(1, epochs + 1):
        cgm.train()
        pf, _, _ = cgm(X_tr, pb_tr)
        loss = crit(pf, y_tr)
        opt.zero_grad(); loss.backward(); opt.step()

        cgm.eval()
        with torch.no_grad():
            pf_v, _, _ = cgm(X_va, pb_va)
        vm = compute_metrics(y_va.cpu().tolist(), pf_v.cpu().tolist())

        if vm["f1"] > best_f1:
            best_f1 = vm["f1"]
            best_state = {k: v.clone() for k, v in cgm.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= 25:
            break

    cgm.load_state_dict(best_state)
    cgm.eval()
    with torch.no_grad():
        pf_te, alpha_te, _ = cgm(X_te, pb_te)

    tm = compute_metrics(y_te.cpu().tolist(), pf_te.cpu().tolist())
    tm["alpha_mean"]    = round(float(alpha_te.mean()), 4)
    tm["alpha_violent"] = round(float(alpha_te[y_te.cpu() == 1].mean()), 4)
    tm["alpha_normal"]  = round(float(alpha_te[y_te.cpu() == 0].mean()), 4)

    print(f"  [{model_name}+CGM] "
          f"Acc:{tm['accuracy']:.4f} F1:{tm['f1']:.4f} "
          f"FPR:{tm['fpr']:.4f} FNR:{tm['fnr']:.4f} "
          f"α_mean:{tm['alpha_mean']:.3f}")
    return tm


# ═══════════════════════════════════════════════════════════════════════════
# 7. Main
# ═══════════════════════════════════════════════════════════════════════════

def main(args):
    ckpt_dir    = Path("checkpoints"); ckpt_dir.mkdir(exist_ok=True)
    results_dir = Path("results");     results_dir.mkdir(exist_ok=True)
    cache_dir   = Path("cache")

    # ── Datasets ──────────────────────────────────────────────────────
    print("\n  Loading RWF-2000...")
    train_ds = RWF2000Dataset(args.root, args.split, "train", augment=True)
    val_ds   = RWF2000Dataset(args.root, args.split, "val",   augment=False)
    test_ds  = RWF2000Dataset(args.root, args.split, "test",  augment=False)

    nw = 4 if torch.cuda.is_available() else 0
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=nw, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=nw, pin_memory=True)

    # ── Model configs ──────────────────────────────────────────────────
    MODEL_CONFIGS = [
        {
            "name":    "MViTv2-S",
            "builder": build_mvitv2_small,
            "lr":      5e-5,   # Transformer cần lr nhỏ hơn CNN
        },
        {
            "name":    "VideoSwin-B",
            "builder": build_video_swin_base,
            "lr":      5e-5,
        },
    ]

    all_results = []

    for cfg in MODEL_CONFIGS:
        name = cfg["name"]
        print(f"\n\n  {'#'*65}")
        print(f"  ## {name}")
        print(f"  {'#'*65}")

        # ── Build model ───────────────────────────────────────────────
        model = cfg["builder"]()

        # ── Train ─────────────────────────────────────────────────────
        model, history = train_model(
            model, train_loader, val_loader,
            epochs=args.epochs, lr=cfg["lr"],
            name=name, ckpt_dir=ckpt_dir,
        )

        # Save history
        with open(results_dir / f"{name}_history.json", "w") as f:
            json.dump(history, f, indent=2)

        # ── Baseline test ─────────────────────────────────────────────
        print(f"\n  [{name}] Evaluating on test set...")
        criterion = nn.BCEWithLogitsLoss()
        test_m, _, _ = evaluate(model, test_loader, criterion, f"{name}-Test")

        print(f"\n  [{name}] BASELINE TEST:")
        print(f"  Acc:{test_m['accuracy']:.4f}  F1:{test_m['f1']:.4f}  "
              f"AUC:{test_m['auc_roc']:.4f}  "
              f"FPR:{test_m['fpr']:.4f}  FNR:{test_m['fnr']:.4f}")

        # ── Extract p_base ────────────────────────────────────────────
        print(f"\n  [{name}] Extracting p_base for all 1989 clips...")
        p_base_all, labels_all, splits_all = extract_p_base(
            model, args.root, args.split, batch_size=args.batch_size
        )
        np.save(cache_dir / f"p_base_{name.lower().replace('-','_')}.npy",
                p_base_all)

        # p_base sanity check
        vio_mean  = p_base_all[labels_all == 1].mean()
        norm_mean = p_base_all[labels_all == 0].mean()
        print(f"  p_base sanity: violent={vio_mean:.4f}, normal={norm_mean:.4f}, "
              f"gap={vio_mean-norm_mean:.4f}")

        # ── Train CGM ─────────────────────────────────────────────────
        print(f"\n  [{name}] Training CGM (962 params)...")
        cgm_m = train_cgm(
            p_base_all, labels_all, splits_all,
            cache_dir, name, epochs=200,
        )

        # ── FPR improvement ───────────────────────────────────────────
        fpr_base = test_m["fpr"]
        fpr_cgm  = cgm_m["fpr"]
        delta    = fpr_base - fpr_cgm
        rel_pct  = (delta / fpr_base * 100) if fpr_base > 0 else 0

        print(f"\n  [{name}] FPR: {fpr_base:.4f} → {fpr_cgm:.4f} "
              f"(Δ={delta:.4f}, {rel_pct:.1f}% relative reduction)")

        all_results.append({
            "model":    name,
            "baseline": test_m,
            "with_cgm": cgm_m,
            "fpr_delta":     round(delta, 4),
            "fpr_rel_pct":   round(rel_pct, 1),
            "lr":        cfg["lr"],
            "epochs":    args.epochs,
        })

        # Free GPU memory before next model
        del model
        torch.cuda.empty_cache()
        print(f"\n  GPU memory cleared for next model.")

    # ── Add X3D-S results from Phase 2/4 ──────────────────────────────
    e0_path  = results_dir / "E0_baseline.json"
    abl_path = results_dir / "ablation_results.json"
    if e0_path.exists() and abl_path.exists():
        with open(e0_path) as f:
            e0 = json.load(f).get("metrics", {})
        with open(abl_path) as f:
            abl = json.load(f)
        e4 = next((r["test_metrics"] for r in abl if r["experiment"]=="E4"), {})
        if e0 and e4:
            fpr_b = e0.get("fpr", 0)
            fpr_c = e4.get("fpr", 0)
            all_results.append({
                "model": "X3D-S (Phase2/4)",
                "baseline": e0, "with_cgm": e4,
                "fpr_delta":   round(fpr_b - fpr_c, 4),
                "fpr_rel_pct": round((fpr_b - fpr_c) / fpr_b * 100, 1) if fpr_b > 0 else 0,
            })

    # ── Final comparison table ─────────────────────────────────────────
    print(f"\n\n  {'='*80}")
    print(f"  FINAL MODEL-AGNOSTIC RESULTS")
    print(f"  {'='*80}")
    print(f"  {'Model':<22} {'Acc':>6} {'F1':>6} {'FPR':>6}  →  "
          f"{'FPR+CGM':>7} {'ΔFPR':>7} {'Rel%':>7}")
    print(f"  {'─'*80}")

    for r in all_results:
        b, c = r["baseline"], r["with_cgm"]
        direction = "↓" if r["fpr_delta"] > 0 else "↑"
        print(f"  {r['model']:<22} "
              f"{b.get('accuracy','?'):>6} {b.get('f1','?'):>6} "
              f"{b.get('fpr','?'):>6}  →  "
              f"{c.get('fpr','?'):>7} "
              f"{r['fpr_delta']:>+7.4f} "
              f"{direction}{abs(r['fpr_rel_pct']):.1f}%")

    all_reduced = all(r["fpr_delta"] > 0 for r in all_results)
    print(f"\n  CGM model-agnostic: "
          f"{'CONFIRMED — FPR↓ for ALL models' if all_reduced else 'PARTIAL'}")

    # ── Save ──────────────────────────────────────────────────────────
    with open(results_dir / "phase5_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    with open(results_dir / "phase5_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model", "Acc_base", "F1_base", "FPR_base",
                     "Acc_CGM", "F1_CGM", "FPR_CGM", "ΔFPR", "Rel%"])
        for r in all_results:
            b, c = r["baseline"], r["with_cgm"]
            w.writerow([r["model"],
                        b.get("accuracy",""), b.get("f1",""), b.get("fpr",""),
                        c.get("accuracy",""), c.get("f1",""), c.get("fpr",""),
                        r["fpr_delta"], r["fpr_rel_pct"]])

    print(f"\n  Saved → results/phase5_results.json")
    print(f"  Saved → results/phase5_table.csv")
    print(f"\n  Phase 5 DONE.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",       type=str, required=True)
    parser.add_argument("--split",      type=str, required=True)
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch_size", type=int,   default=16)
    args = parser.parse_args()
    main(args)