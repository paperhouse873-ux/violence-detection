"""
Phase 6 — Cross-dataset Zero-shot Transfer (RWF-2000 -> RLVS)
=============================================================
Mục tiêu (RQ3): CGM huấn luyện trên RWF-2000 có giảm FPR khi áp ZERO-SHOT
lên RLVS (không train lại) hay không?

Quy trình:
  1) Train CGM "E4" trên cache RWF-2000 (Phase 3/4) — lưu cgm + scaler.
     (Phase 4 không lưu trọng số CGM nên ta tái tạo đúng cấu hình E4.)
  2) Trích xuất 4 luồng cho RLVS: p_base (X3D-S frozen), crowd, lighting, motion.
     -> cache riêng từng phần, resume được.
  3) Chuẩn hóa 13-dim RLVS bằng CHÍNH scaler của RWF (zero-shot, không fit lại).
  4) Áp CGM -> p_final. So FPR/accuracy/F1 BEFORE (p_base) vs AFTER (p_final).
  5) Lưu results/phase6_results.json (đúng format cho phase4b _plot_phase6)
     + results/phase6_table.csv.

Chạy thử nhanh (vd 20 clip/lớp) để kiểm tra pipeline:
  python phase6_cross_dataset.py --limit 20

Chạy đầy đủ (2000 clip — lâu trên CPU):
  python phase6_cross_dataset.py

Tăng tốc trên CPU (lấy thưa frame hơn):
  python phase6_cross_dataset.py --crowd_every 8 --motion_stride 3
"""

import os
import json
import csv
import time
import pickle
import argparse
import warnings
from pathlib import Path

# Ép stdout sang UTF-8 để in được tiếng Việt trên console Windows (cp1252)
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import cv2
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score, confusion_matrix,
)

warnings.filterwarnings("ignore")

import sys
sys.path.append(str(Path(__file__).parent))
# Tái dùng hàm trích xuất đã kiểm chứng ở Phase 3
from phase3_extract_context import (
    load_frozen_x3ds,
    extract_crowd_features,
    extract_lighting_features,
    extract_motion_features,
)
from phase1_dataset import KINETICS_MEAN, KINETICS_STD

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE   = Path("cache")
RLVS_CACHE = Path("cache/rlvs"); RLVS_CACHE.mkdir(parents=True, exist_ok=True)
RESULTS = Path("results"); RESULTS.mkdir(exist_ok=True)
CKPT    = Path("checkpoints")

RLVS_ROOT = Path("RLVS/Real Life Violence Dataset")
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


# ════════════════════════════════════════════════════════════════════════════
# 0. Context Gating Module (giống Phase 4)
# ════════════════════════════════════════════════════════════════════════════

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

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def compute_metrics(labels, probs, threshold=0.5):
    preds = [1 if p >= threshold else 0 for p in probs]
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return {
        "accuracy": round(accuracy_score(labels, preds), 4),
        "f1":       round(f1_score(labels, preds, zero_division=0), 4),
        "auc_roc":  round(roc_auc_score(labels, probs), 4)
                    if len(set(labels)) > 1 else 0.0,
        "fpr":      round(fpr, 4),
        "fnr":      round(fnr, 4),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


# ════════════════════════════════════════════════════════════════════════════
# 1. Train CGM "E4" trên RWF-2000 cache (rồi lưu cgm + scaler)
# ════════════════════════════════════════════════════════════════════════════

def train_cgm_on_rwf(epochs=200, lr=1e-3, seed=42):
    """Tái tạo cấu hình E4 (full 3 streams) từ cache Phase 3, lưu lại trọng số."""
    torch.manual_seed(seed); np.random.seed(seed)
    p_base  = np.load(CACHE / "p_base.npy")
    z_crowd = np.load(CACHE / "z_crowd.npy")
    z_light = np.load(CACHE / "z_light.npy")
    z_motion= np.load(CACHE / "z_motion.npy")
    labels  = np.load(CACHE / "labels.npy")
    splits  = np.load(CACHE / "splits.npy")

    X = np.concatenate([p_base.reshape(-1, 1), z_crowd, z_light, z_motion], axis=1)
    tr, va = splits == 0, splits == 1

    scaler = StandardScaler().fit(X[tr])    # fit CHỈ trên train RWF
    Xn = scaler.transform(X).astype(np.float32)

    def t(a): return torch.tensor(a).to(DEVICE)
    X_tr, y_tr, pb_tr = t(Xn[tr]), t(labels[tr].astype(np.float32)), t(p_base[tr].astype(np.float32))
    X_va, y_va, pb_va = t(Xn[va]), t(labels[va].astype(np.float32)), t(p_base[va].astype(np.float32))

    model = ContextGatingModule(X.shape[1]).to(DEVICE)
    crit = nn.BCELoss()
    opt  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max",
                                                       factor=0.5, patience=10)
    best_f1, best_state, patience = 0.0, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        pf, _, _ = model(X_tr, pb_tr)
        loss = crit(pf, y_tr)
        opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            pf_v, _, _ = model(X_va, pb_va)
        vm = compute_metrics(y_va.cpu().tolist(), pf_v.cpu().tolist())
        sched.step(vm["f1"])
        if vm["f1"] > best_f1:
            best_f1, patience = vm["f1"], 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if patience >= 25:
            break

    model.load_state_dict(best_state)
    CKPT.mkdir(exist_ok=True)
    torch.save({"model_state_dict": best_state, "input_dim": X.shape[1],
                "best_val_f1": best_f1}, CKPT / "cgm_e4.pth")
    with open(CKPT / "cgm_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"  [CGM] trained on RWF — best val F1={best_f1:.4f}, "
          f"params={model.count_params()} -> saved cgm_e4.pth + cgm_scaler.pkl")
    return model, scaler


def load_or_train_cgm(force=False):
    cgm_p, sc_p = CKPT / "cgm_e4.pth", CKPT / "cgm_scaler.pkl"
    if cgm_p.exists() and sc_p.exists() and not force:
        ck = torch.load(cgm_p, map_location=DEVICE, weights_only=False)
        model = ContextGatingModule(ck["input_dim"]).to(DEVICE)
        model.load_state_dict(ck["model_state_dict"]); model.eval()
        scaler = pickle.load(open(sc_p, "rb"))
        print(f"  [CGM] loaded cgm_e4.pth (val F1={ck.get('best_val_f1','?')})")
        return model, scaler
    return train_cgm_on_rwf()


# ════════════════════════════════════════════════════════════════════════════
# 2. Liệt kê clip RLVS
# ════════════════════════════════════════════════════════════════════════════

def list_rlvs(limit=None):
    """Trả về (paths, labels). Violence=1, NonViolence=0."""
    samples = []
    for folder, label in [("Violence", 1), ("NonViolence", 0)]:
        vids = sorted([p for p in (RLVS_ROOT / folder).glob("*")
                       if p.suffix.lower() in VIDEO_EXTS])
        if limit:
            vids = vids[:limit]
        samples += [(str(p), label) for p in vids]
    paths  = [s[0] for s in samples]
    labels = np.array([s[1] for s in samples], dtype=np.int32)
    return paths, labels


# ════════════════════════════════════════════════════════════════════════════
# 3. Trích xuất p_base (X3D-S frozen) cho RLVS
# ════════════════════════════════════════════════════════════════════════════

class RLVSVideoDS(torch.utils.data.Dataset):
    def __init__(self, paths, n_frames=16, img_size=224):
        self.paths = paths; self.n_frames = n_frames; self.img_size = img_size
        self.mean = torch.tensor(KINETICS_MEAN).view(3, 1, 1, 1)
        self.std  = torch.tensor(KINETICS_STD).view(3, 1, 1, 1)

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        cap = cv2.VideoCapture(self.paths[idx])
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or self.n_frames
        idxs = np.linspace(0, max(total - 1, 0), self.n_frames, dtype=int)
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
        t = torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0
                             ).permute(3, 0, 1, 2)
        t = (t - self.mean) / self.std
        return t, idx


@torch.no_grad()
def extract_p_base_rlvs(paths, batch_size=4):
    out = RLVS_CACHE / "p_base.npy"
    if out.exists() and len(np.load(out)) == len(paths):
        print("  [SKIP] RLVS p_base cached."); return np.load(out)
    model = load_frozen_x3ds(str(CKPT / "x3ds_best.pth"))
    ds = RLVSVideoDS(paths)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size,
                                         shuffle=False, num_workers=0)
    probs = np.zeros(len(paths), dtype=np.float32)
    for videos, idxs in tqdm(loader, desc="  [RLVS p_base]", unit="batch"):
        videos = videos.to(DEVICE)
        p = torch.sigmoid(model(videos).squeeze(1)).cpu().numpy()
        for j, ix in zip(p, idxs.numpy()):
            probs[ix] = j
    np.save(out, probs)
    print(f"  saved -> {out}  shape={probs.shape}")
    return probs


# ════════════════════════════════════════════════════════════════════════════
# 4. Trích xuất 3 context streams cho RLVS (cache + resume)
# ════════════════════════════════════════════════════════════════════════════

def extract_stream(paths, name, fn, dim=4, **kwargs):
    """[Tuần tự] Trích 1 stream với resume: lưu (N,dim), bỏ qua dòng đã có."""
    out = RLVS_CACHE / f"{name}.npy"
    if out.exists() and len(np.load(out)) == len(paths):
        arr = np.load(out)
        if (np.abs(arr).sum(axis=1) > 0).all():
            print(f"  [SKIP] RLVS {name} cached."); return arr
    else:
        arr = np.zeros((len(paths), dim), dtype=np.float32)
    done = np.abs(arr).sum(axis=1) > 0
    todo = np.where(~done)[0]
    print(f"  [{name}] con {len(todo)}/{len(paths)} clip can xu ly (tuan tu)...")
    for k, i in enumerate(tqdm(todo, desc=f"  [RLVS {name}]")):
        try:
            arr[i] = fn(paths[i], **kwargs)
        except Exception:
            arr[i] = np.zeros(dim, dtype=np.float32)
        if k % 50 == 0:                 # lưu định kỳ để resume
            np.save(out, arr)
    np.save(out, arr)
    print(f"  saved -> {out}")
    return arr


# ── Multiprocessing: worker cấp module (picklable trên Windows) ──────────────

def _stream_worker(task):
    """Worker cho 1 clip. task = (stream_name, index, path, kwargs_dict).
    Trả về (index, feature_vector). Giới hạn thread nội bộ để tránh
    oversubscription khi nhiều process cùng chạy."""
    name, idx, path, kwargs = task
    try:
        import cv2 as _cv2
        _cv2.setNumThreads(1)          # mỗi process chỉ dùng 1 thread OpenCV
    except Exception:
        pass
    try:
        import torch as _torch
        _torch.set_num_threads(1)      # tránh torch (YOLO) chiếm nhiều thread
    except Exception:
        pass
    import numpy as _np
    try:
        if name == "z_crowd":
            return idx, extract_crowd_features(path, **kwargs)
        elif name == "z_light":
            return idx, extract_lighting_features(path)
        elif name == "z_motion":
            return idx, extract_motion_features(path)
        else:
            return idx, _np.zeros(4, dtype=_np.float32)
    except Exception:
        return idx, _np.zeros(4, dtype=_np.float32)


def extract_stream_parallel(paths, name, dim=4, workers=8, **kwargs):
    """[Song song] Trích 1 stream trên nhiều process. Cache + resume y hệt
    bản tuần tự. Logic tính toán KHÔNG đổi (chỉ chia việc cho nhiều nhân) nên
    kết quả giống bản tuần tự -> giữ nguyên tính hợp lệ zero-shot."""
    from concurrent.futures import ProcessPoolExecutor

    out = RLVS_CACHE / f"{name}.npy"
    if out.exists() and len(np.load(out)) == len(paths):
        arr = np.load(out)
        if (np.abs(arr).sum(axis=1) > 0).all():
            print(f"  [SKIP] RLVS {name} cached."); return arr
    else:
        arr = np.zeros((len(paths), dim), dtype=np.float32)

    done = np.abs(arr).sum(axis=1) > 0
    todo = [int(i) for i in np.where(~done)[0]]
    if not todo:
        np.save(out, arr); return arr
    print(f"  [{name}] con {len(todo)}/{len(paths)} clip can xu ly "
          f"({workers} workers song song)...")

    tasks = [(name, i, paths[i], kwargs) for i in todo]
    cnt = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        # chunksize nhỏ để cân tải tốt khi thời lượng clip dao động lớn
        for idx, feat in tqdm(ex.map(_stream_worker, tasks, chunksize=2),
                              total=len(tasks), desc=f"  [RLVS {name}]"):
            arr[idx] = feat
            cnt += 1
            if cnt % 100 == 0:          # lưu định kỳ để resume
                np.save(out, arr)
    np.save(out, arr)
    print(f"  saved -> {out}")
    return arr


# ════════════════════════════════════════════════════════════════════════════
# 5. Main
# ════════════════════════════════════════════════════════════════════════════

def main(args):
    t0 = time.time()
    print(f"\n  {'='*60}")
    print(f"  Phase 6 — Cross-dataset Zero-shot (RWF-2000 -> RLVS)")
    print(f"  Device: {DEVICE}")
    print(f"  {'='*60}\n")

    # (1) CGM + scaler từ RWF
    cgm, scaler = load_or_train_cgm(force=args.retrain_cgm)

    # (2) liệt kê RLVS
    paths, labels = list_rlvs(limit=args.limit)
    print(f"\n  RLVS clips: {len(paths)} "
          f"(violent={int((labels==1).sum())}, normal={int((labels==0).sum())})")

    # (3) trích xuất 4 luồng
    print("\n  -- Extracting RLVS features --")
    p_base = extract_p_base_rlvs(paths, batch_size=args.batch_size)

    if args.workers and args.workers > 1:
        # crowd dùng ít worker hơn vì YOLO ngốn RAM/CPU nhiều hơn các luồng khác
        crowd_workers = max(1, min(args.workers, args.crowd_workers))
        z_crowd  = extract_stream_parallel(paths, "z_crowd", dim=4,
                       workers=crowd_workers, sample_every=args.crowd_every)
        z_light  = extract_stream_parallel(paths, "z_light", dim=4,
                       workers=args.workers)
        z_motion = extract_stream_parallel(paths, "z_motion", dim=4,
                       workers=args.workers)
    else:
        z_crowd  = extract_stream(paths, "z_crowd",  extract_crowd_features,
                                  dim=4, sample_every=args.crowd_every)
        z_light  = extract_stream(paths, "z_light",  extract_lighting_features,
                                  dim=4)
        z_motion = extract_stream(paths, "z_motion", extract_motion_features,
                                  dim=4)
    np.save(RLVS_CACHE / "labels.npy", labels)

    # (4) chuẩn hóa bằng scaler RWF (ZERO-SHOT) + áp CGM
    X = np.concatenate([p_base.reshape(-1, 1), z_crowd, z_light, z_motion], axis=1)
    Xn = scaler.transform(X).astype(np.float32)
    with torch.no_grad():
        pf, alpha, p_ctx = cgm(torch.tensor(Xn).to(DEVICE),
                               torch.tensor(p_base.astype(np.float32)).to(DEVICE))
    p_final = pf.cpu().numpy()
    alpha_np = alpha.cpu().numpy()

    # (5) metrics BEFORE vs AFTER
    y = labels.tolist()
    before = compute_metrics(y, p_base.tolist())
    after  = compute_metrics(y, p_final.tolist())
    after["alpha_mean"]    = round(float(alpha_np.mean()), 4)
    after["alpha_violent"] = round(float(alpha_np[labels == 1].mean()), 4)
    after["alpha_normal"]  = round(float(alpha_np[labels == 0].mean()), 4)

    delta = before["fpr"] - after["fpr"]
    rel = (delta / before["fpr"] * 100) if before["fpr"] > 0 else 0.0

    print(f"\n  {'='*60}")
    print(f"  RESULTS — RLVS (zero-shot transfer)")
    print(f"  {'='*60}")
    print(f"  {'Metric':<10}{'Before (X3D-S)':>16}{'After (+CGM)':>16}")
    for m in ["fpr", "fnr", "accuracy", "f1", "auc_roc"]:
        print(f"  {m:<10}{before[m]:>16}{after[m]:>16}")
    print(f"\n  FPR: {before['fpr']:.4f} -> {after['fpr']:.4f}  "
          f"(Δ={delta:.4f}, {rel:.1f}% relative)")
    print(f"  α mean={after['alpha_mean']} "
          f"(violent={after['alpha_violent']}, normal={after['alpha_normal']})")
    verdict = ("CGM transfers (FPR giảm)" if delta > 0
               else "CGM KHÔNG giảm FPR trên RLVS (cần phân tích)")
    print(f"  -> {verdict}")

    # (6) lưu kết quả (format khớp phase4b _plot_phase6)
    result = [{
        "model": "X3D-S",
        "dataset": "RLVS",
        "n_clips": len(paths),
        "baseline": before,
        "with_cgm": after,
        "fpr_delta": round(delta, 4),
        "fpr_rel_pct": round(rel, 1),
        "transfer": "zero-shot (CGM trained on RWF-2000)",
    }]
    with open(RESULTS / "phase6_results.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(RESULTS / "phase6_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "fpr", "fnr", "accuracy", "f1", "auc_roc"])
        w.writerow(["before", before["fpr"], before["fnr"], before["accuracy"],
                    before["f1"], before["auc_roc"]])
        w.writerow(["after",  after["fpr"], after["fnr"], after["accuracy"],
                    after["f1"], after["auc_roc"]])
    print(f"\n  saved -> results/phase6_results.json + phase6_table.csv")
    print(f"  Done in {(time.time()-t0)/60:.1f} min.")
    print(f"  (Chạy lại phase4b_eda_visualization.py để vẽ RQ3b transfer.)\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Giới hạn số clip MỖI LỚP (để test nhanh). None=tất cả.")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8,
                    help="So process song song cho crowd/light/motion. "
                         "1 = tuan tu. Mac dinh 8 (Ryzen 9 = 8 nhan).")
    ap.add_argument("--crowd_workers", type=int, default=4,
                    help="Gioi han worker rieng cho YOLO crowd (ngon RAM hon).")
    ap.add_argument("--crowd_every", type=int, default=4,
                    help="YOLO chay moi N frame (tang de nhanh hon tren CPU).")
    ap.add_argument("--motion_stride", type=int, default=1,
                    help="(giữ chỗ) bước nhảy frame cho optical flow.")
    ap.add_argument("--retrain_cgm", action="store_true",
                    help="Train lại CGM trên RWF dù đã có checkpoint.")
    args = ap.parse_args()
    main(args)
