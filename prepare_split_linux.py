"""
prepare_split_linux.py — Sửa split.json cho khớp filesystem (Linux/A100)
========================================================================
Vấn đề khi đưa split.json (tạo trên Windows) sang Linux:
  1) Dấu phân cách '\' (Windows) -> Linux coi là ký tự thường, sai đường dẫn.
  2) Hoa/thường: split dùng 'fight'/'nonFight', dataset có thể là 'Fight'/'NonFight'.
  3) Dataset Kaggle đôi khi giải nén lồng thêm 1 cấp (vd RWF-2000/RWF-2000/...).

Cách xử lý (an toàn, không sửa code pipeline):
  * Quét toàn bộ file video dưới --root, lập chỉ mục theo TÊN FILE (lowercase).
    Tên clip RWF-2000 là duy nhất nên khớp theo tên file là đủ tin cậy.
  * Với mỗi entry trong split.json: lấy tên file, tra chỉ mục, ghi lại đường
    dẫn THẬT (tương đối so với --root, dùng '/').
  * Báo cáo số khớp / không khớp. Ghi ra --out (mặc định ghi đè split.json).

Dùng trên A100 (sau khi giải nén dataset):
  python prepare_split_linux.py --root RWF-2000
  # nếu cấu trúc lồng:  python prepare_split_linux.py --root RWF-2000/RWF-2000
"""

import os
import sys
import json
import argparse
import shutil
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv"}


def build_index(root: Path):
    """Hai chỉ mục, key đều lowercase + dùng '/':
       by_rel  : {duong_dan_tuong_doi.lower(): duong_dan_that}  (CHÍNH, an toàn)
       by_name : {ten_file.lower(): [duong_dan_that, ...]}      (dự phòng)
    RWF-2000 có ~257 tên file trùng giữa các split -> PHẢI khớp theo đường dẫn
    đầy đủ để không gán nhầm clip sang sai split/lớp."""
    by_rel, by_name = {}, {}
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if Path(fn).suffix.lower() in VIDEO_EXTS:
                rel = os.path.relpath(os.path.join(dirpath, fn), root).replace("\\", "/")
                by_rel[rel.lower()] = rel
                by_name.setdefault(fn.lower(), []).append(rel)
    return by_rel, by_name


def main(args):
    root = Path(args.root)
    if not root.exists():
        print(f"  [LỖI] không thấy thư mục root: {root}")
        sys.exit(1)

    with open(args.split, encoding="utf-8") as f:
        data = json.load(f)

    print(f"  Quét video dưới: {root.resolve()}")
    by_rel, by_name = build_index(root)
    print(f"  Tìm thấy {len(by_rel)} file video")
    if len(by_rel) == 0:
        print("  [LỖI] Không thấy file video nào. Kiểm tra lại --root (có thể bị lồng thêm 1 cấp).")
        sys.exit(1)

    total, fixed, missing, ambig = 0, 0, [], 0
    for split_name in ["train", "val", "test"]:
        if split_name not in data:
            continue
        for item in data[split_name]:
            total += 1
            norm = item["path"].replace("\\", "/")
            # (1) khớp CHÍNH theo đường dẫn đầy đủ (an toàn với tên trùng)
            if norm.lower() in by_rel:
                new = by_rel[norm.lower()]
            else:
                # (2) dự phòng theo tên file — CHỈ khi tên đó là duy nhất
                bn = os.path.basename(norm).lower()
                cand = by_name.get(bn, [])
                if len(cand) == 1:
                    new = cand[0]
                else:
                    if len(cand) > 1:
                        ambig += 1
                    missing.append(item["path"])
                    continue
            if new != item["path"]:
                fixed += 1
            item["path"] = new

    print(f"\n  Tổng entry: {total} | sửa đường dẫn: {fixed} | "
          f"KHÔNG khớp: {len(missing)}" + (f" (trong đó {ambig} do tên trùng)" if ambig else ""))
    if missing:
        print("  [CẢNH BÁO] một số clip không tìm thấy (in tối đa 10):")
        for m in missing[:10]:
            print(f"     - {m}")
        if not args.force:
            print("\n  Dừng lại (chưa ghi). Nếu vẫn muốn ghi, thêm --force. "
                  "Hoặc kiểm tra lại --root.")
            sys.exit(1)

    # sao lưu rồi ghi đè
    out = Path(args.out)
    if out.exists() and out.resolve() == Path(args.split).resolve():
        shutil.copy(out, str(out) + ".bak")
        print(f"  Đã sao lưu bản cũ -> {out}.bak")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  Đã ghi split đã sửa -> {out}")

    # sanity: thử mở 1 clip đầu mỗi split
    print("\n  Kiểm tra tồn tại (1 clip mỗi split):")
    for split_name in ["train", "val", "test"]:
        if data.get(split_name):
            p = root / data[split_name][0]["path"]
            print(f"     {split_name}: {p}  ->  {'OK' if p.exists() else 'THIẾU!'}")
    print("\n  Xong. Giờ chạy được phase8 trên Linux.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="RWF-2000", help="Thư mục gốc chứa train/ val/")
    ap.add_argument("--split", default="split.json")
    ap.add_argument("--out", default="split.json", help="Mặc định ghi đè split.json (có .bak)")
    ap.add_argument("--force", action="store_true", help="Ghi dù còn clip không khớp")
    args = ap.parse_args()
    main(args)
