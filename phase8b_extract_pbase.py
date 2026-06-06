"""
Phase 8b — Trích p_base cho detector MỚI (SlowFast / MViT)
==========================================================
Sau khi fine-tune xong (phase8), ta cần p_base của detector mới trên TOÀN BỘ
1989 clip để gắn CGM (Giai đoạn C). Context streams (crowd/light/motion) KHÔNG
phụ thuộc detector -> TÁI DÙNG cache cũ, chỉ trích lại p_base.

CỰC KỲ QUAN TRỌNG — thứ tự clip:
  Phải GIỐNG HỆT Phase 3: duyệt split.json theo train -> val -> test.
  Nhờ vậy p_base_<model>.npy khớp hàng với cache/labels.npy, splits.npy,
  z_crowd.npy, z_light.npy, z_motion.npy (cùng index).

Output:
  cache/p_base_<model>.npy        (shape (1989,), float32)

Kiểm tra ở LOCAL trước (không cần checkpoint thật, chỉ test pipeline + thứ tự):
  python phase8b_extract_pbase.py --model slowfast_r50 --limit 6 --allow_pretrained
  python phase8b_extract_pbase.py --model mvit_base_16x4 --limit 6 --allow_pretrained

Chạy thật trên A100 (sau khi đã có checkpoint fine-tuned):
  python phase8b_extract_pbase.py --model slowfast_r50
  python phase8b_extract_pbase.py --model mvit_base_16x4
"""

import sys
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))
from phase1_dataset import KINETICS_MEAN, KINETICS_STD
from phase8_finetune_detectors import build_model, pack_input, N_FRAMES, DEVICE

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CACHE = Path("cache"); CACHE.mkdir(exist_ok=True)
CKPT = Path("checkpoints")


class ClipDataset(torch.utils.data.Dataset):
    """Đọc clip -> tensor (3, T, H, W). Cùng cách sample frame như Phase 3."""
    def __init__(self, samples, root, n_frames, img_size=224):
        self.samples = samples
        self.root = Path(root)
        self.n_frames = n_frames
        self.img_size = img_size
        self.mean = torch.tensor(KINETICS_MEAN).view(3, 1, 1, 1)
        self.std = torch.tensor(KINETICS_STD).view(3, 1, 1, 1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.root / self.samples[idx]["path"]
        cap = cv2.VideoCapture(str(path))
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


def build_ordered_samples(split_file):
    """Tái tạo CHÍNH XÁC thứ tự Phase 3: train -> val -> test."""
    with open(split_file) as f:
        sd = json.load(f)
    samples, split_ids = [], []
    for sid, sname in enumerate(["train", "val", "test"]):
        for item in sd[sname]:
            samples.append(item)
            split_ids.append(sid)
    return samples, np.array(split_ids, dtype=np.int32)


@torch.no_grad()
def main(args):
    model_name = args.model
    nf = N_FRAMES[model_name]
    print("=" * 66)
    print(f"  Trích p_base — {model_name}  (n_frames={nf}, device={DEVICE})")
    print("=" * 66)

    # 1) thứ tự clip giống Phase 3
    samples, split_ids = build_ordered_samples(args.split)
    if args.limit:
        samples = samples[:args.limit]
        split_ids = split_ids[:args.limit]
    N = len(samples)
    print(f"  clips: {N}")

    # 2) build model + nạp checkpoint fine-tuned
    ckpt_path = CKPT / f"{model_name}_best.pth"
    model = build_model(model_name)
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        print(f"  loaded checkpoint: {ckpt_path}  (val F1={ck.get('val_metrics',{}).get('f1','?')})")
    else:
        if not args.allow_pretrained:
            raise FileNotFoundError(
                f"Khong tim thay {ckpt_path}. Hay fine-tune truoc (phase8), "
                f"hoac dung --allow_pretrained de TEST pipeline o local.")
        print(f"  [CẢNH BÁO] chưa có {ckpt_path} -> dùng trọng số pretrained "
              f"CHỈ để test pipeline (p_base sẽ KHÔNG hợp lệ cho paper).")
    model.eval()

    # 3) trích p_base theo thứ tự
    ds = ClipDataset(samples, args.root, nf)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
                                         shuffle=False, num_workers=args.num_workers)
    probs = np.zeros(N, dtype=np.float32)
    for videos, idxs in tqdm(loader, desc="  [p_base]", unit="batch"):
        videos = videos.to(DEVICE)
        x = pack_input(videos, model_name)            # pack tươi mỗi batch
        p = torch.sigmoid(model(x).squeeze(1)).cpu().numpy()
        for j, ix in zip(p, idxs.numpy()):
            probs[ix] = j

    # 4) lưu (chỉ lưu bản đầy đủ; bản --limit chỉ để test, không ghi đè)
    if args.limit:
        print(f"\n  [TEST MODE limit={args.limit}] p_base mẫu: "
              f"{np.round(probs[:args.limit], 3).tolist()}")
        print("  Pipeline + thứ tự OK. KHÔNG lưu file (đang ở chế độ test).")
    else:
        out = CACHE / f"p_base_{model_name}.npy"
        np.save(out, probs)
        # sanity: phải khớp độ dài với labels cache
        if (CACHE / "labels.npy").exists():
            nlab = len(np.load(CACHE / "labels.npy"))
            ok = (nlab == N)
            print(f"\n  saved -> {out}  shape={probs.shape}")
            print(f"  khớp với labels.npy (N={nlab}): {'OK' if ok else 'LECH!'}")
        print(f"  p_base mean={probs.mean():.4f}  min={probs.min():.4f}  max={probs.max():.4f}")
    print("  done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["slowfast_r50", "mvit_base_16x4"])
    ap.add_argument("--root", default="RWF-2000")
    ap.add_argument("--split", default="split.json")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None,
                    help="Chỉ xử lý N clip đầu để TEST pipeline ở local.")
    ap.add_argument("--allow_pretrained", action="store_true",
                    help="Cho phép chạy khi chưa có checkpoint (chỉ để test).")
    args = ap.parse_args()
    main(args)
