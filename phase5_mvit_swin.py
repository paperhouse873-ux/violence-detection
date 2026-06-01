"""
Phase 5 — Train 3 modern base detectors + Apply CGM
=====================================================
Models (all 2022+, timm pretrained, A100-compatible):
  M1: SwinV2-S + LSTM      — Swin Transformer V2 Small (Liu et al., CVPR 2022)
  M2: ConvNeXt-S + LSTM    — ConvNeXt Small (Liu et al., CVPR 2022)
  M3: EfficientNetV2-S+LSTM— EfficientNetV2 Small (Tan & Le, ICML 2021/2022)

Architecture: per-frame 2D CNN → LSTM temporal → binary classifier
Proven approach: no HuggingFace API complexity, no shape issues.

Chạy:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python phase5_mvit_swin.py \
    --root "/workspace/RWF2000_full/RWF-2000" \
    --split "/workspace/code/split.json" \
    --epochs 20 --batch_size 4
"""

import json, time, csv
import torch
import argparse
import numpy as np
import torch.nn as nn
import timm
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
# 1. 3 Model configs — all 2022+, all different architectures
# ═══════════════════════════════════════════════════════════════════════════

MODEL_CONFIGS = [
    {
        "name":        "SwinV2-S",
        "backbone":    "swinv2_small_window8_256",
        "feat_dim":    768,
        "img_size":    256,          # SwinV2 uses 256x256
        "lr_backbone": 1e-5,
        "lr_head":     1e-3,
        "paper":       "Liu et al., CVPR 2022",
        "desc":        "Swin Transformer V2 Small — hierarchical shifted-window attention",
    },
    {
        "name":        "ConvNeXt-S",
        "backbone":    "convnext_small",
        "feat_dim":    768,
        "img_size":    224,
        "lr_backbone": 1e-5,
        "lr_head":     1e-3,
        "paper":       "Liu et al., CVPR 2022",
        "desc":        "ConvNeXt Small — modernized CNN with Transformer design principles",
    },
    {
        "name":        "EfficientNetV2-S",
        "backbone":    "tf_efficientnetv2_s",
        "feat_dim":    1280,
        "img_size":    300,          # EfficientNetV2-S uses 300x300
        "lr_backbone": 1e-5,
        "lr_head":     1e-3,
        "paper":       "Tan & Le, ICML 2021",
        "desc":        "EfficientNetV2 Small — progressive learning, Fused-MBConv",
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# 2. VideoClassifier — per-frame 2D CNN + LSTM
# ═══════════════════════════════════════════════════════════════════════════

class VideoClassifier(nn.Module):
    """
    Input : (B, C, T, H, W)
    Output: logits (B,)  — no sigmoid, use BCEWithLogitsLoss
    """
    def __init__(self, backbone_name: str, feat_dim: int,
                 hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )
        self.feat_dim = feat_dim
        self.lstm = nn.LSTM(feat_dim, hidden, num_layers=1,
                            batch_first=True, dropout=0)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden, 1)
        # Init FC bias to 0 — avoids all-positive / all-negative at start
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        feats = self.backbone(x).reshape(B, T, self.feat_dim)
        _, (h_n, _) = self.lstm(feats)
        logits = self.fc(self.drop(h_n[-1])).squeeze(1)
        return logits

    def param_groups(self, lr_backbone: float, lr_head: float):
        """Return param groups with different LR for backbone vs head."""
        backbone_params = list(self.backbone.parameters())
        head_params = list(self.lstm.parameters()) + \
                      list(self.drop.parameters()) + \
                      list(self.fc.parameters())
        return [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params,     "lr": lr_head},
        ]


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
        "f1":       round(f1_score(labels, preds, zero_division=0), 4),
        "auc_roc":  round(roc_auc_score(labels, probs), 4),
        "fpr":      round(fpr, 4),
        "fnr":      round(fnr, 4),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. Train / Evaluate
# ═══════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, criterion, name, epoch):
    model.train()
    total_loss, all_labels, all_probs = 0.0, [], []
    pbar = tqdm(loader, desc=f"  [{name} E{epoch:02d}]", unit="batch")
    for videos, labels in pbar:
        videos = videos.to(DEVICE, non_blocking=True)
        labels = labels.float().to(DEVICE)
        optimizer.zero_grad()
        logits = model(videos)
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
    for videos, labels in tqdm(loader, desc=f"  [{desc}]",
                                unit="batch", leave=False):
        videos   = videos.to(DEVICE, non_blocking=True)
        labels_f = labels.float().to(DEVICE)
        logits   = model(videos)
        loss     = criterion(logits, labels_f)
        probs    = torch.sigmoid(logits)
        total_loss += loss.item()
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
    m = compute_metrics(all_labels, all_probs)
    m["loss"] = round(total_loss / len(loader), 4)
    return m, all_labels, all_probs


def build_dataloader(root, split_file, split, batch_size, img_size, augment):
    """DataLoader với img_size tuỳ chỉnh theo model."""
    ds = RWF2000Dataset(
        root=root, split_file=split_file,
        split=split, img_size=img_size, augment=augment,
    )
    return DataLoader(ds, batch_size=batch_size,
                      shuffle=(split=="train"),
                      num_workers=4, pin_memory=True)


def train_model(model, cfg, train_loader, val_loader,
                epochs, name, ckpt_dir):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.param_groups(cfg["lr_backbone"], cfg["lr_head"]),
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=cfg["lr_backbone"] * 0.01)

    best_f1, best_state = 0.0, None
    print(f"\n  {'='*65}")
    print(f"  {name} | {cfg['paper']}")
    print(f"  {cfg['desc']}")
    print(f"  LR backbone={cfg['lr_backbone']} head={cfg['lr_head']} | batch={train_loader.batch_size}")
    print(f"  {'='*65}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_m = train_epoch(model, train_loader, optimizer, criterion, name, epoch)
        val_m, _, _ = evaluate(model, val_loader, criterion, "Val")
        scheduler.step()
        elapsed = time.time() - t0

        print(f"  [{name}] E{epoch:02d}/{epochs} ({elapsed/60:.1f}min) "
              f"loss:{train_m['loss']:.4f} "
              f"val_f1:{val_m['f1']:.4f} "
              f"val_fpr:{val_m['fpr']:.4f} "
              f"val_acc:{val_m['accuracy']:.4f}")

        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save(best_state,
                ckpt_dir / f"{name.lower().replace('-','_')}_best.pth")
            print(f"  [{name}] ** Saved best (F1={best_f1:.4f}) **")

    model.load_state_dict(best_state)
    return model


# ═══════════════════════════════════════════════════════════════════════════
# 5. Extract p_base
# ═══════════════════════════════════════════════════════════════════════════

class AllSamplesDS(torch.utils.data.Dataset):
    def __init__(self, root, split_file, img_size=224):
        import cv2
        self.root = Path(root)
        self.n_frames = 16
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
        idxs  = np.linspace(0, total - 1, self.n_frames, dtype=int)
        frames = []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ret, f = cap.read()
            if not ret or f is None:
                f = frames[-1].copy() if frames else \
                    np.zeros((self.img_size, self.img_size, 3), np.uint8)
            f = cv2.resize(f, (self.img_size, self.img_size))
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            frames.append(f)
        cap.release()
        if not frames:
            frames = [np.zeros((self.img_size, self.img_size, 3), np.uint8)
                      ] * self.n_frames
        t = torch.from_numpy(
            np.stack(frames).astype(np.float32) / 255.0
        ).permute(3, 0, 1, 2)
        t = (t - self.mean) / self.std
        return t, self.samples[idx]["label"]


@torch.no_grad()
def extract_p_base(model, root, split_file, batch_size, img_size):
    model.eval()
    ds = AllSamplesDS(root, split_file, img_size=img_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    all_probs, all_labels = [], []
    for videos, labels in tqdm(loader, desc="  [extract p_base]"):
        videos = videos.to(DEVICE, non_blocking=True)
        probs  = torch.sigmoid(model(videos))
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())
    return (
        np.array(all_probs,   dtype=np.float32),
        np.array(all_labels,  dtype=np.int32),
        np.array(ds.split_ids, dtype=np.int32),
    )


# ═══════════════════════════════════════════════════════════════════════════
# 6. Context Gating Module (same as Phase 4)
# ═══════════════════════════════════════════════════════════════════════════

class CGM(nn.Module):
    def __init__(self, dim=13):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim,32), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(32,1), nn.Sigmoid())
        self.ctx  = nn.Sequential(nn.Linear(dim,32), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(32,1), nn.Sigmoid())
    def forward(self, x, pb):
        a = self.gate(x).squeeze(1)
        c = self.ctx(x).squeeze(1)
        return a * pb + (1 - a) * c, a, c


def train_cgm(p_base_all, labels_all, splits_all, cache_dir, name):
    z_c = np.load(cache_dir / "z_crowd.npy")
    z_l = np.load(cache_dir / "z_light.npy")
    z_m = np.load(cache_dir / "z_motion.npy")
    X   = np.concatenate([p_base_all.reshape(-1,1), z_c, z_l, z_m], axis=1)

    tr, va, te = splits_all==0, splits_all==1, splits_all==2
    sc = StandardScaler(); sc.fit(X[tr])
    X  = sc.transform(X).astype(np.float32)

    def _t(a, fl=True):
        return torch.tensor(a.astype(np.float32) if fl else a).to(DEVICE)

    X_tr, y_tr, pb_tr = _t(X[tr]), _t(labels_all[tr]), _t(p_base_all[tr])
    X_va, y_va, pb_va = _t(X[va]), _t(labels_all[va]), _t(p_base_all[va])
    X_te, y_te, pb_te = _t(X[te]), _t(labels_all[te]), _t(p_base_all[te])

    cgm  = CGM(13).to(DEVICE)
    crit = nn.BCELoss()
    opt  = torch.optim.AdamW(cgm.parameters(), lr=1e-3, weight_decay=1e-4)
    best_f1, best_st, pat = 0.0, None, 0

    for ep in range(1, 301):
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
            best_st = {k: v.clone() for k, v in cgm.state_dict().items()}
            pat = 0
        else:
            pat += 1
        if pat >= 30: break

    cgm.load_state_dict(best_st); cgm.eval()
    with torch.no_grad():
        pf_te, alpha_te, _ = cgm(X_te, pb_te)
    tm = compute_metrics(y_te.cpu().tolist(), pf_te.cpu().tolist())
    tm["alpha_mean"]    = round(float(alpha_te.mean()), 4)
    tm["alpha_violent"] = round(float(alpha_te[y_te.cpu()==1].mean()), 4)
    tm["alpha_normal"]  = round(float(alpha_te[y_te.cpu()==0].mean()), 4)
    print(f"  [{name}+CGM] Acc:{tm['accuracy']:.4f} F1:{tm['f1']:.4f} "
          f"FPR:{tm['fpr']:.4f} FNR:{tm['fnr']:.4f} "
          f"α:{tm['alpha_mean']:.3f}")
    return tm


# ═══════════════════════════════════════════════════════════════════════════
# 7. Main
# ═══════════════════════════════════════════════════════════════════════════

def main(args):
    ckpt_dir    = Path("checkpoints"); ckpt_dir.mkdir(exist_ok=True)
    results_dir = Path("results");     results_dir.mkdir(exist_ok=True)
    cache_dir   = Path("cache")

    all_results = []

    for cfg in MODEL_CONFIGS:
        name = cfg["name"]
        print(f"\n\n  {'#'*65}")
        print(f"  ## {name} — {cfg['paper']}")
        print(f"  {'#'*65}")

        # DataLoaders với img_size riêng của từng model
        img_size = cfg["img_size"]
        print(f"\n  Loading datasets (img_size={img_size})...")
        train_loader = build_dataloader(args.root, args.split, "train",
                                        args.batch_size, img_size, True)
        val_loader   = build_dataloader(args.root, args.split, "val",
                                        args.batch_size, img_size, False)
        test_loader  = build_dataloader(args.root, args.split, "test",
                                        args.batch_size, img_size, False)

        # Build model
        model = VideoClassifier(cfg["backbone"], cfg["feat_dim"]).to(DEVICE)
        total = sum(p.numel() for p in model.parameters())
        print(f"  Total params: {total/1e6:.1f}M")

        # Train
        model = train_model(model, cfg, train_loader, val_loader,
                            args.epochs, name, ckpt_dir)

        # Test baseline
        criterion = nn.BCEWithLogitsLoss()
        test_m, _, _ = evaluate(model, test_loader, criterion, f"{name}-Test")
        print(f"\n  [{name}] BASELINE: "
              f"Acc:{test_m['accuracy']:.4f} F1:{test_m['f1']:.4f} "
              f"FPR:{test_m['fpr']:.4f} FNR:{test_m['fnr']:.4f}")

        # Extract p_base
        print(f"\n  [{name}] Extracting p_base for 1989 clips...")
        p_base, labels_all, splits_all = extract_p_base(
            model, args.root, args.split, args.batch_size, img_size)
        np.save(cache_dir / f"p_base_{name.lower().replace('-','_')}.npy", p_base)

        vio  = p_base[labels_all==1].mean()
        norm = p_base[labels_all==0].mean()
        print(f"  p_base gap: violent={vio:.4f}, normal={norm:.4f} "
              f"(gap={vio-norm:.4f})")

        # Train CGM
        print(f"\n  [{name}] Training CGM (962 params)...")
        cgm_m = train_cgm(p_base, labels_all, splits_all, cache_dir, name)

        fpr_b = test_m["fpr"]
        fpr_c = cgm_m["fpr"]
        delta = fpr_b - fpr_c
        rel   = (delta / fpr_b * 100) if fpr_b > 0 else 0

        print(f"\n  [{name}] FPR: {fpr_b:.4f} → {fpr_c:.4f} "
              f"(Δ={delta:.4f}, {rel:.1f}% relative)")

        all_results.append({
            "model": name, "paper": cfg["paper"],
            "desc":  cfg["desc"],
            "baseline": test_m, "with_cgm": cgm_m,
            "fpr_delta": round(delta, 4),
            "fpr_rel_pct": round(rel, 1),
        })

        del model; torch.cuda.empty_cache()

    # Load X3D-S results from Phase 2/4
    e0_p  = results_dir / "E0_baseline.json"
    abl_p = results_dir / "ablation_results.json"
    if e0_p.exists() and abl_p.exists():
        with open(e0_p)  as f: e0  = json.load(f).get("metrics", {})
        with open(abl_p) as f: abl = json.load(f)
        e4 = next((r["test_metrics"] for r in abl
                   if r.get("experiment") == "E4"), {})
        if e0 and e4:
            fb, fc = e0.get("fpr", 0), e4.get("fpr", 0)
            all_results.append({
                "model": "X3D-S",
                "paper": "Fan et al., CVPR 2020",
                "desc":  "X3D — efficient 3D CNN expansion",
                "baseline": e0, "with_cgm": e4,
                "fpr_delta": round(fb - fc, 4),
                "fpr_rel_pct": round((fb-fc)/fb*100, 1) if fb > 0 else 0,
            })

    # Final comparison table
    print(f"\n\n  {'='*80}")
    print(f"  FINAL — MODEL-AGNOSTIC PROOF")
    print(f"  {'='*80}")
    print(f"  {'Model':<20} {'Paper':<22} {'FPR':>6}  →  {'FPR+CGM':>7} "
          f"{'ΔFPR':>7} {'Rel%':>7} {'✓':>4}")
    print(f"  {'─'*80}")

    confirmed = 0
    for r in all_results:
        b, c = r["baseline"], r["with_cgm"]
        ok = r["fpr_delta"] > 0
        if ok: confirmed += 1
        mark = "✓" if ok else "✗"
        print(f"  {r['model']:<20} {r['paper']:<22} "
              f"{b.get('fpr','?'):>6}  →  "
              f"{c.get('fpr','?'):>7} "
              f"{r['fpr_delta']:>+7.4f} "
              f"{r['fpr_rel_pct']:>6.1f}% "
              f"{mark:>4}")

    print(f"  {'─'*80}")
    print(f"\n  CGM confirmed on {confirmed}/{len(all_results)} models "
          f"→ Model-agnostic: "
          f"{'CONFIRMED' if confirmed == len(all_results) else 'PARTIAL'}")

    # Save
    with open(results_dir / "phase5_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    with open(results_dir / "phase5_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model","Paper","Acc_base","F1_base","FPR_base",
                     "Acc_CGM","F1_CGM","FPR_CGM","ΔFPR","Rel%"])
        for r in all_results:
            b, c = r["baseline"], r["with_cgm"]
            w.writerow([r["model"], r["paper"],
                b.get("accuracy",""), b.get("f1",""), b.get("fpr",""),
                c.get("accuracy",""), c.get("f1",""), c.get("fpr",""),
                r["fpr_delta"], r["fpr_rel_pct"]])

    print(f"\n  Saved → results/phase5_results.json")
    print(f"  Saved → results/phase5_table.csv")
    print(f"  Phase 5 DONE.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",       type=str, required=True)
    parser.add_argument("--split",      type=str, required=True)
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch_size", type=int,   default=4)
    args = parser.parse_args()
    main(args)