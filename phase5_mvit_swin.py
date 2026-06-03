"""
Phase 5 — Train 3 video models + Apply CGM (model-agnostic proof)
===================================================================
Dùng ĐÚNG approach đã thành công với X3D-S ở Phase 2:
  - Load từ pytorchvideo hub (pretrained Kinetics-400)
  - Thay head: Linear(feat_dim → 1)
  - Xóa internal Softmax: act = Identity()
  - BCEWithLogitsLoss (không double sigmoid)
  - lr = 1e-3, unfreeze toàn bộ

Models:
  M1: X3D-M       — X3D expanded (Fan et al., CVPR 2020)
  M2: Slow-R50    — SlowFast single-stream (Feichtenhofer et al., ICCV 2019)
  M3: SlowFast-R50— Dual-pathway slow+fast (Feichtenhofer et al., ICCV 2019)

Chạy:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python phase5_mvit_swin.py \
    --root "/workspace/RWF2000_full/RWF-2000" \
    --split "/workspace/code/split.json" \
    --epochs 20 --batch_size 8
"""

import json, time, csv
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
# 1. Model builders — exact same approach as X3D-S Phase 2
# ═══════════════════════════════════════════════════════════════════════════

def remove_head_activation(model):
    """Xóa Softmax bên trong head — nguyên nhân FPR=1.0"""
    last = model.blocks[-1]
    if hasattr(last, "act"):
        last.act = nn.Identity()
    if hasattr(last, "activation"):
        last.activation = nn.Identity()
    if hasattr(last, "output_act"):
        last.output_act = nn.Identity()


def build_x3d_m():
    print("\n  Loading X3D-M pretrained Kinetics-400...")
    model = torch.hub.load(
        "facebookresearch/pytorchvideo",
        "x3d_m", pretrained=True, trust_repo=True,
    )
    in_f = model.blocks[-1].proj.in_features
    model.blocks[-1].proj = nn.Linear(in_f, 1)
    remove_head_activation(model)
    model._slowfast = False
    print(f"  X3D-M head: Linear({in_f} → 1), Softmax removed")
    return model


def build_slow_r50():
    print("\n  Loading Slow-R50 pretrained Kinetics-400...")
    model = torch.hub.load(
        "facebookresearch/pytorchvideo",
        "slow_r50", pretrained=True, trust_repo=True,
    )
    # Slow-R50 head: model.blocks[-1].proj
    in_f = model.blocks[-1].proj.in_features
    model.blocks[-1].proj = nn.Linear(in_f, 1)
    remove_head_activation(model)
    model._slowfast = False
    model._slow_only = True
    print(f"  Slow-R50 head: Linear({in_f} → 1), Softmax removed")
    return model


def build_slowfast_r50():
    print("\n  Loading SlowFast-R50 pretrained Kinetics-400...")
    model = torch.hub.load(
        "facebookresearch/pytorchvideo",
        "slowfast_r50", pretrained=True, trust_repo=True,
    )
    in_f = model.blocks[-1].proj.in_features
    model.blocks[-1].proj = nn.Linear(in_f, 1)
    remove_head_activation(model)
    model._slowfast = True
    print(f"  SlowFast-R50 head: Linear({in_f} → 1), Softmax removed")
    return model


def model_forward(model, videos):
    """
    Unified forward pass.
    SlowFast cần list [slow, fast]; các model khác nhận tensor trực tiếp.
    """
    if getattr(model, "_slowfast", False):
        # SlowFast: slow=every 4th frame, fast=all frames
        slow = videos[:, :, ::4, :, :]   # (B, C, T/4, H, W)
        fast = videos                      # (B, C, T,   H, W)
        out = model([slow, fast])
    elif getattr(model, "_slow_only", False):
        # Slow-R50: expects list of one tensor
        out = model([videos])
    else:
        out = model(videos)

    # Normalize output shape
    if isinstance(out, torch.Tensor):
        if out.dim() == 2:
            return out.squeeze(1)
        return out
    return out


MODEL_CONFIGS = [
    {
        "name":    "X3D-M",
        "builder": build_x3d_m,
        "lr":      1e-3,
        "paper":   "Fan et al., CVPR 2020",
        "desc":    "X3D-M — expanded X3D, larger temporal/spatial resolution",
    },
    {
        "name":    "Slow-R50",
        "builder": build_slow_r50,
        "lr":      1e-3,
        "paper":   "Feichtenhofer et al., ICCV 2019",
        "desc":    "Slow-R50 — single-stream 3D ResNet, slow temporal stride",
    },
    {
        "name":    "SlowFast-R50",
        "builder": build_slowfast_r50,
        "lr":      1e-3,
        "paper":   "Feichtenhofer et al., ICCV 2019",
        "desc":    "SlowFast-R50 — dual-pathway slow+fast temporal fusion",
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# 2. Metrics
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
# 3. Train / Evaluate — identical to Phase 2
# ═══════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, criterion, name, epoch):
    model.train()
    total_loss, all_labels, all_probs = 0.0, [], []
    pbar = tqdm(loader, desc=f"  [{name} E{epoch:02d}]", unit="batch")
    for videos, labels in pbar:
        videos = videos.to(DEVICE, non_blocking=True)
        labels = labels.float().to(DEVICE)
        optimizer.zero_grad()
        logits = model_forward(model, videos)
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
        logits   = model_forward(model, videos)
        loss     = criterion(logits, labels_f)
        probs    = torch.sigmoid(logits)
        total_loss += loss.item()
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
    m = compute_metrics(all_labels, all_probs)
    m["loss"] = round(total_loss / len(loader), 4)
    return m, all_labels, all_probs


def train_model(model, train_loader, val_loader, epochs, lr, name, ckpt_dir):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3)

    best_f1, best_state = 0.0, None
    print(f"\n  {'='*60}")
    print(f"  {name} | LR={lr} | batch={train_loader.batch_size} | BCEWithLogitsLoss")
    print(f"  {'='*60}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_m = train_epoch(model, train_loader, optimizer,
                              criterion, name, epoch)
        val_m, _, _ = evaluate(model, val_loader, criterion, "Val")
        scheduler.step(val_m["f1"])
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
# 4. Extract p_base
# ═══════════════════════════════════════════════════════════════════════════

class AllSamplesDS(torch.utils.data.Dataset):
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
            frames = [np.zeros((self.img_size, self.img_size, 3),
                      np.uint8)] * self.n_frames
        t = torch.from_numpy(
            np.stack(frames).astype(np.float32) / 255.0
        ).permute(3, 0, 1, 2)
        t = (t - self.mean) / self.std
        return t, self.samples[idx]["label"]


@torch.no_grad()
def extract_p_base(model, root, split_file, batch_size=8):
    model.eval()
    ds = AllSamplesDS(root, split_file)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    all_probs, all_labels = [], []
    for videos, labels in tqdm(loader, desc="  [extract p_base]"):
        videos = videos.to(DEVICE, non_blocking=True)
        probs  = torch.sigmoid(model_forward(model, videos))
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())
    return (
        np.array(all_probs,    dtype=np.float32),
        np.array(all_labels,   dtype=np.int32),
        np.array(ds.split_ids, dtype=np.int32),
    )


# ═══════════════════════════════════════════════════════════════════════════
# 5. Context Gating Module
# ═══════════════════════════════════════════════════════════════════════════

class CGM(nn.Module):
    def __init__(self, dim=13):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1),   nn.Sigmoid())
        self.ctx = nn.Sequential(
            nn.Linear(dim, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1),   nn.Sigmoid())

    def forward(self, x, pb):
        a = self.gate(x).squeeze(1)
        c = self.ctx(x).squeeze(1)
        return a * pb + (1 - a) * c, a, c


def train_cgm(p_base_all, labels_all, splits_all, cache_dir, name):
    z_c = np.load(cache_dir / "z_crowd.npy")
    z_l = np.load(cache_dir / "z_light.npy")
    z_m = np.load(cache_dir / "z_motion.npy")
    X   = np.concatenate([p_base_all.reshape(-1, 1), z_c, z_l, z_m], axis=1)

    tr, va, te = splits_all == 0, splits_all == 1, splits_all == 2
    sc = StandardScaler(); sc.fit(X[tr])
    X  = sc.transform(X).astype(np.float32)

    def _t(a): return torch.tensor(a.astype(np.float32)).to(DEVICE)
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
            best_f1 = vm["f1"]; pat = 0
            best_st = {k: v.clone() for k, v in cgm.state_dict().items()}
        else:
            pat += 1
        if pat >= 30: break

    cgm.load_state_dict(best_st); cgm.eval()
    with torch.no_grad():
        pf_te, alpha_te, _ = cgm(X_te, pb_te)
    tm = compute_metrics(y_te.cpu().tolist(), pf_te.cpu().tolist())
    tm["alpha_mean"]    = round(float(alpha_te.mean()), 4)
    tm["alpha_violent"] = round(float(alpha_te[y_te.cpu() == 1].mean()), 4)
    tm["alpha_normal"]  = round(float(alpha_te[y_te.cpu() == 0].mean()), 4)
    print(f"  [{name}+CGM] Acc:{tm['accuracy']:.4f} F1:{tm['f1']:.4f} "
          f"FPR:{tm['fpr']:.4f} FNR:{tm['fnr']:.4f} "
          f"α:{tm['alpha_mean']:.3f}")
    return tm


# ═══════════════════════════════════════════════════════════════════════════
# 6. Main
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
        print(f"  ## {cfg['desc']}")
        print(f"  {'#'*65}")

        # Build model
        model = cfg["builder"]().to(DEVICE)
        total = sum(p.numel() for p in model.parameters())
        print(f"  Total params: {total/1e6:.1f}M")

        # DataLoaders
        print("\n  Loading datasets...")
        train_ds = RWF2000Dataset(args.root, args.split, "train", augment=True)
        val_ds   = RWF2000Dataset(args.root, args.split, "val",   augment=False)
        test_ds  = RWF2000Dataset(args.root, args.split, "test",  augment=False)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True,  num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                                  shuffle=False, num_workers=4, pin_memory=True)
        test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                                  shuffle=False, num_workers=4, pin_memory=True)

        # Train
        model = train_model(model, train_loader, val_loader,
                            args.epochs, cfg["lr"], name, ckpt_dir)

        # Baseline test
        criterion = nn.BCEWithLogitsLoss()
        test_m, _, _ = evaluate(model, test_loader, criterion, f"{name}-Test")
        print(f"\n  [{name}] BASELINE: Acc:{test_m['accuracy']:.4f} "
              f"F1:{test_m['f1']:.4f} FPR:{test_m['fpr']:.4f} "
              f"FNR:{test_m['fnr']:.4f}")

        # Extract p_base
        print(f"\n  [{name}] Extracting p_base for 1989 clips...")
        p_base, labels_all, splits_all = extract_p_base(
            model, args.root, args.split, args.batch_size)
        np.save(cache_dir / f"p_base_{name.lower().replace('-','_')}.npy",
                p_base)
        vio  = p_base[labels_all == 1].mean()
        norm = p_base[labels_all == 0].mean()
        print(f"  p_base: violent={vio:.4f} normal={norm:.4f} "
              f"gap={vio-norm:.4f}")

        # Train CGM
        print(f"\n  [{name}] Training CGM (962 params)...")
        cgm_m = train_cgm(p_base, labels_all, splits_all, cache_dir, name)

        fpr_b = test_m["fpr"]
        fpr_c = cgm_m["fpr"]
        delta = fpr_b - fpr_c
        rel   = (delta / fpr_b * 100) if fpr_b > 0 else 0

        print(f"\n  [{name}] FPR: {fpr_b:.4f} → {fpr_c:.4f} "
              f"(Δ={delta:.4f}, {rel:.1f}%)")

        all_results.append({
            "model": name, "paper": cfg["paper"],
            "baseline": test_m, "with_cgm": cgm_m,
            "fpr_delta": round(delta, 4),
            "fpr_rel_pct": round(rel, 1),
        })

        del model; torch.cuda.empty_cache()

    # Load X3D-S from Phase 2/4
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
                "baseline": e0, "with_cgm": e4,
                "fpr_delta": round(fb - fc, 4),
                "fpr_rel_pct": round((fb-fc)/fb*100, 1) if fb > 0 else 0,
            })

    # Final table
    print(f"\n\n  {'='*80}")
    print(f"  FINAL — MODEL-AGNOSTIC PROOF")
    print(f"  {'='*80}")
    print(f"  {'Model':<16} {'Paper':<28} {'FPR':>6} → {'FPR+CGM':>7} "
          f"{'ΔFPR':>7} {'Rel%':>7} {'OK':>4}")
    print(f"  {'─'*80}")
    ok_count = 0
    for r in all_results:
        b, c = r["baseline"], r["with_cgm"]
        ok = r["fpr_delta"] > 0
        if ok: ok_count += 1
        print(f"  {r['model']:<16} {r['paper']:<28} "
              f"{b.get('fpr','?'):>6} → {c.get('fpr','?'):>7} "
              f"{r['fpr_delta']:>+7.4f} {r['fpr_rel_pct']:>6.1f}% "
              f"{'✓' if ok else '✗':>4}")
    print(f"\n  CGM confirmed: {ok_count}/{len(all_results)} models")

    with open(results_dir / "phase5_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    with open(results_dir / "phase5_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model","Paper","Acc_b","F1_b","FPR_b",
                     "Acc_c","F1_c","FPR_c","ΔFPR","Rel%"])
        for r in all_results:
            b, c = r["baseline"], r["with_cgm"]
            w.writerow([r["model"], r["paper"],
                b.get("accuracy",""), b.get("f1",""), b.get("fpr",""),
                c.get("accuracy",""), c.get("f1",""), c.get("fpr",""),
                r["fpr_delta"], r["fpr_rel_pct"]])
    print(f"  Saved → results/phase5_results.json")
    print(f"  Saved → results/phase5_table.csv\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",       type=str, required=True)
    parser.add_argument("--split",      type=str, required=True)
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch_size", type=int,   default=8)
    args = parser.parse_args()
    main(args)