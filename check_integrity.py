"""
Phase 0 — Step 2: Kiểm tra tính toàn vẹn (detect file lỗi/corrupt)
====================================================================
Chạy: python phase0_step2_integrity.py --root /path/to/RWF-2000

Cách hoạt động:
  - Mở từng file video bằng OpenCV
  - Đọc thử frame đầu + frame cuối
  - Kiểm tra số frame > 0
  - Ghi lại file lỗi vào corrupted_files.txt
"""

import os
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm

VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv"}


def is_valid_video(path: Path) -> tuple[bool, str]:
    """Trả về (is_valid, error_message)"""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return False, "Cannot open file"

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return False, f"Frame count = {total_frames}"

    # Đọc thử frame đầu tiên
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        return False, "Cannot read first frame"

    # Đọc thử frame cuối
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        return False, "Cannot read last frame"

    cap.release()
    return True, ""


def check_integrity(root: Path):
    print(f"\n{'='*55}")
    print(f"  RWF-2000 Integrity Check")
    print(f"{'='*55}\n")

    # Thu thập tất cả file video
    all_files = []
    for folder in root.rglob("*"):
        if folder.is_dir():
            for f in folder.iterdir():
                if f.suffix.lower() in VIDEO_EXTS:
                    all_files.append(f)

    print(f"  Scanning {len(all_files)} video files...\n")

    corrupted = []
    for path in tqdm(all_files, desc="  Checking", unit="file"):
        valid, err = is_valid_video(path)
        if not valid:
            corrupted.append((str(path), err))

    # Kết quả
    n_ok   = len(all_files) - len(corrupted)
    n_bad  = len(corrupted)

    print(f"\n{'─'*55}")
    print(f"  Valid files:     {n_ok}")
    print(f"  Corrupted files: {n_bad}")

    if corrupted:
        out = Path("corrupted_files.txt")
        with open(out, "w") as f:
            for path, err in corrupted:
                f.write(f"{path}\t{err}\n")
        print(f"\n  Corrupted list saved → {out}")
        print(f"\n  First 5 corrupted files:")
        for path, err in corrupted[:5]:
            print(f"    {Path(path).name}: {err}")
    else:
        print(f"\n  All files OK — no corruption detected.")

    print(f"{'='*55}\n")
    return corrupted


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./RWF-2000")
    args = parser.parse_args()
    check_integrity(Path(args.root))