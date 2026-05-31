"""
Phase 1 — Dataset class + DataLoader test
==========================================
File này gồm 2 phần:

  Phần 1: class RWF2000Dataset  → dùng lại ở tất cả experiment
  Phần 2: __main__              → test DataLoader, kiểm tra speed

Chạy: python phase1_dataset.py --root /path/to/RWF-2000 --split split.json
"""

import cv2
import json
import time
import torch
import random
import argparse
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# ── Kinetics-400 mean/std (chuẩn X3D dùng) ──────────────────────────────
KINETICS_MEAN = [0.45, 0.45, 0.45]
KINETICS_STD  = [0.225, 0.225, 0.225]


# ═══════════════════════════════════════════════════════════════════════════
# Phần 1 — Dataset class
# ═══════════════════════════════════════════════════════════════════════════

class RWF2000Dataset(Dataset):
    """
    Dataset RWF-2000 cho PyTorch.

    Input:  video path
    Output: tensor (C, T, H, W) = (3, 16, 224, 224), label (0 hoặc 1)

    Cách sample frames:
      Lấy T=16 frames cách đều nhau trong video.
      Ví dụ: video 150 frames → lấy frame [0, 9, 18, ..., 140, 149].
      Nếu video ngắn hơn T frames → lặp lại frame cuối cho đủ.

    Args:
        root:       Path đến thư mục gốc RWF-2000
        split_file: Path đến split.json (từ phase0_step4)
        split:      "train", "val", hoặc "test"
        n_frames:   Số frames sample (mặc định 16 — chuẩn X3D-S)
        img_size:   Kích thước resize (mặc định 224)
        augment:    Bật data augmentation khi train
    """

    def __init__(
        self,
        root:       str | Path,
        split_file: str | Path,
        split:      str = "train",
        n_frames:   int = 16,
        img_size:   int = 224,
        augment:    bool = False,
    ):
        self.root      = Path(root)
        self.n_frames  = n_frames
        self.img_size  = img_size
        self.augment   = augment

        # Load split
        with open(split_file) as f:
            data = json.load(f)

        if split not in data:
            raise ValueError(f"Split '{split}' not in {split_file}. "
                             f"Available: {list(data.keys())}")

        self.samples = data[split]  # list of {"path": ..., "label": ...}
        self.split   = split

        print(f"  RWF2000Dataset [{split}]: {len(self.samples)} clips")

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_frames(self, path: Path) -> np.ndarray:
        """
        Đọc video và sample T frames đều.
        Trả về array (T, H, W, C) theo thứ tự BGR (OpenCV).
        """
        cap = cv2.VideoCapture(str(path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total <= 0:
            # Fallback: đọc hết rồi sample
            frames_raw = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames_raw.append(frame)
            cap.release()
            total = len(frames_raw)
        else:
            frames_raw = None

        # Tính indices cần đọc
        if total >= self.n_frames:
            indices = np.linspace(0, total - 1, self.n_frames, dtype=int)
        else:
            # Video ngắn hơn n_frames: dùng hết, pad bằng frame cuối
            indices = list(range(total)) + \
                      [total - 1] * (self.n_frames - total)
            indices = np.array(indices)

        frames = []
        if frames_raw is not None:
            # Đã đọc hết từ trước
            for idx in indices:
                idx = min(idx, len(frames_raw) - 1)
                frames.append(frames_raw[idx])
        else:
            # Dùng seek
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if not ret or frame is None:
                    # Dùng frame cuối đọc được
                    frame = frames[-1] if frames else np.zeros(
                        (self.img_size, self.img_size, 3), dtype=np.uint8)
                frames.append(frame)
            cap.release()

        return np.stack(frames)  # (T, H_orig, W_orig, 3)

    def _preprocess(self, frames: np.ndarray) -> torch.Tensor:
        """
        frames: (T, H, W, 3) BGR uint8
        → tensor (3, T, H, W) float32, normalized
        """
        processed = []
        for frame in frames:
            # Resize
            frame = cv2.resize(frame, (self.img_size, self.img_size))
            # BGR → RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Augmentation (chỉ khi train)
            if self.augment:
                frame = self._augment(frame)
            # uint8 → float [0,1]
            frame = frame.astype(np.float32) / 255.0
            processed.append(frame)

        # Stack: (T, H, W, 3) → (3, T, H, W)
        tensor = torch.from_numpy(np.stack(processed))  # (T, H, W, 3)
        tensor = tensor.permute(3, 0, 1, 2)             # (3, T, H, W)

        # Normalize theo Kinetics mean/std
        mean = torch.tensor(KINETICS_MEAN).view(3, 1, 1, 1)
        std  = torch.tensor(KINETICS_STD).view(3, 1, 1, 1)
        tensor = (tensor - mean) / std

        return tensor  # (3, 16, 224, 224)

    def _augment(self, frame: np.ndarray) -> np.ndarray:
        """Augmentation đơn giản: horizontal flip ngẫu nhiên"""
        if random.random() > 0.5:
            frame = cv2.flip(frame, 1)
        return frame

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        path   = self.root / sample["path"]
        label  = sample["label"]

        frames = self._sample_frames(path)
        tensor = self._preprocess(frames)

        return tensor, label


# ═══════════════════════════════════════════════════════════════════════════
# Phần 2 — Test DataLoader
# ═══════════════════════════════════════════════════════════════════════════

def test_dataloader(root: str, split_file: str):
    print(f"\n{'='*55}")
    print(f"  Phase 1 — DataLoader Test")
    print(f"{'='*55}\n")

    # ── Tạo dataset ──────────────────────────────────────────────────────
    train_ds = RWF2000Dataset(
        root=root, split_file=split_file,
        split="train", augment=True
    )
    val_ds = RWF2000Dataset(
        root=root, split_file=split_file,
        split="val", augment=False
    )
    test_ds = RWF2000Dataset(
        root=root, split_file=split_file,
        split="test", augment=False
    )

    # ── DataLoader ───────────────────────────────────────────────────────
    train_loader = DataLoader(
    train_ds,
    batch_size=8,       # Tăng batch size để bù
    shuffle=True,
    num_workers=0,      # Windows: phải là 0
    pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=4, shuffle=False, num_workers=2
    )

    # ── Test 1 batch ─────────────────────────────────────────────────────
    print(f"  Testing 1 batch from train_loader...")
    start = time.time()
    videos, labels = next(iter(train_loader))
    elapsed = time.time() - start

    print(f"\n  [Shape check]")
    print(f"    videos.shape : {tuple(videos.shape)}")
    print(f"    Expected     : (4, 3, 16, 224, 224)")
    shape_ok = tuple(videos.shape) == (4, 3, 16, 224, 224)
    print(f"    Status       : {'PASS' if shape_ok else 'FAIL'}")

    print(f"\n  [Label check]")
    print(f"    labels       : {labels.tolist()}")
    print(f"    Unique values: {labels.unique().tolist()}")
    label_ok = set(labels.tolist()).issubset({0, 1})
    print(f"    Status       : {'PASS' if label_ok else 'FAIL'}")

    print(f"\n  [Normalization check]")
    print(f"    Mean: {videos.mean():.4f}  (should be near 0)")
    print(f"    Std : {videos.std():.4f}   (should be near 1)")

    print(f"\n  [Speed test] — 10 batches")
    start = time.time()
    n_clips = 0
    for i, (v, l) in enumerate(train_loader):
        n_clips += v.shape[0]
        if i >= 9:
            break
    elapsed_10 = time.time() - start
    speed = n_clips / elapsed_10

    print(f"    {n_clips} clips in {elapsed_10:.2f}s")
    print(f"    Speed: {speed:.1f} clips/s")
    speed_ok = speed >= 5  # Ngưỡng tối thiểu 5 clips/s
    print(f"    Status: {'PASS' if speed_ok else 'SLOW — giảm num_workers hoặc tăng batch_size'}")

    # ── Tổng kết ─────────────────────────────────────────────────────────
    all_pass = shape_ok and label_ok and speed_ok
    print(f"\n{'─'*55}")
    print(f"  Overall: {'ALL PASS — DataLoader sẵn sàng' if all_pass else 'Có vấn đề — xem chi tiết trên'}")
    print(f"{'='*55}\n")

    if all_pass:
        print("  DataLoader sẵn sàng cho Phase 2 (Fine-tune X3D-S).\n")

    return train_loader, val_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",  type=str,
                        default="./RWF-2000",
                        help="Path đến thư mục gốc RWF-2000")
    parser.add_argument("--split", type=str,
                        default="./split.json",
                        help="Path đến split.json từ phase0_step4")
    args = parser.parse_args()

    test_dataloader(args.root, args.split)