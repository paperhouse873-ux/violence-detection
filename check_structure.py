import os
import argparse
from pathlib import Path


EXPECTED = {
    "train/fight":    1000,  
    "train/nonFight": 1000,  
    "val/fight":      200,   
    "val/nonFight":   200,   
}
VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv"}

def check_structure(root: Path):
    print(f"\n{'='*55}")
    print(f"  RWF-2000 Structure Check")
    print(f"  Root: {root}")
    print(f"{'='*55}")

    all_ok = True
    total  = 0

    for rel, expected_count in EXPECTED.items():
        folder = root / rel
        exists = folder.exists()

        if not exists:
            print(f"\n  [MISSING]  {rel}/")
            print(f"             Folder not found. Check path or re-extract zip.")
            all_ok = False
            continue

        
        files = [f for f in folder.iterdir()
                 if f.suffix.lower() in VIDEO_EXTS]
        n = len(files)
        total += n
        status = "OK" if n == expected_count else "WARN"

        print(f"\n  [{status:4s}]  {rel}/")
        print(f"             Found {n:4d} videos  (expected {expected_count})")

        if n != expected_count:
            all_ok = False
            diff = expected_count - n
            print(f"             Missing {diff} files." if diff > 0
                  else f"             Extra {-diff} files.")

    print(f"\n{'─'*55}")
    print(f"  Total videos found: {total}  (expected 2400)")
    print(f"  Overall:            {'PASS' if all_ok else 'FAIL — fix issues above'}")
    print(f"{'='*55}\n")
    return all_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./RWF-2000",
                        help="Path đến thư mục gốc RWF-2000")
    args = parser.parse_args()
    check_structure(Path(args.root))