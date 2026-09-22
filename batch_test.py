# batch_test.py
# -*- coding: utf-8 -*-
"""
批量测试：全量融合匹配（DINOv2 + HOG，alpha 加权）
"""

import os
import re
import sys
import numpy as np
import cv2

sys.path.insert(0, ".")
sys.path.insert(0, "src")
import config
from matcher import load_templates, preprocess_query, recognize

TEST_DIR = os.path.join(config.BASE_DIR, "data", "test_batch")


def parse_cp_from_filename(fname):
    m = re.search(r"[Uu]?([0-9A-Fa-f]{4,6})\.", fname)
    if not m:
        return None
    return int(m.group(1), 16)


def main():
    if not os.path.isdir(TEST_DIR):
        print(f"[错误] 目录不存在: {TEST_DIR}")
        sys.exit(1)

    files = sorted([
        f for f in os.listdir(TEST_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))
    ])
    if not files:
        print(f"[错误] {TEST_DIR} 下没有图片")
        sys.exit(1)

    files_with_cp = []
    for f in files:
        cp = parse_cp_from_filename(f)
        if cp is not None:
            files_with_cp.append((cp, f))
    files_with_cp.sort(key=lambda x: x[0])

    cnn_t, hog_t, codepoints = load_templates()
    print(f"模板库: {len(codepoints)} 个字形，"
          f"DINOv2 {cnn_t.shape[1]} 维，HOG {hog_t.shape[1]} 维")
    print(f"匹配策略: 全量融合  最终 Top-K: {config.TOP_K}")
    print(f"测试图片: {len(files_with_cp)} 张")
    print("=" * 100)

    hit1 = hit3 = hit5 = hit10 = hit20 = total = 0
    missed = []

    for expect_cp, fname in files_with_cp:
        img_path = os.path.join(TEST_DIR, fname)
        data = np.fromfile(img_path, dtype=np.uint8)
        img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"  [跳过] 无法读取: {fname}")
            continue

        cnn_vec, hog_vec, _ = preprocess_query(img_bgr, debug=False)
        if cnn_vec is None:
            print(f"  [跳过] 预处理失败: {fname}")
            continue

        results = recognize(cnn_vec, hog_vec, top_k=20, min_score=0.0)
        total += 1

        rank = None
        for i, r in enumerate(results, 1):
            if int(r["codepoint"], 16) == expect_cp:
                rank = i
                break

        if rank == 1:
            hit1 += 1; hit3 += 1; hit5 += 1; hit10 += 1; hit20 += 1
            tag = "✅ Top-1"
        elif rank and rank <= 3:
            hit3 += 1; hit5 += 1; hit10 += 1; hit20 += 1
            tag = f"🟡 Top-{rank}"
        elif rank and rank <= 5:
            hit5 += 1; hit10 += 1; hit20 += 1
            tag = f"🟠 Top-{rank}"
        elif rank and rank <= 10:
            hit10 += 1; hit20 += 1
            tag = f"🔴 Top-{rank}"
        elif rank and rank <= 20:
            hit20 += 1
            tag = f"⬛ Top-{rank}"
        else:
            tag = "❌ 未进 Top-20"

        top1_cp = results[0]["codepoint"] if results else "----"
        top1_hog = (results[0].get("hog_score", 0.0) * 100
                    if results else 0.0)
        top1_cnn = (results[0].get("cnn_score", 0.0) * 100
                    if results else 0.0)
        exp_hog = next(
            (r.get("hog_score", 0.0) * 100 for r in results
             if int(r["codepoint"], 16) == expect_cp),
            0.0
        )
        exp_cnn = next(
            (r.get("cnn_score", 0.0) * 100 for r in results
             if int(r["codepoint"], 16) == expect_cp),
            0.0
        )

        print(f"  U+{expect_cp:04X}  {fname:14s}  "
              f"Top1 U+{top1_cp}(HOG {top1_hog:.1f}%/CNN {top1_cnn:.1f}%)  "
              f"期望分 HOG{exp_hog:.1f}%/CNN{exp_cnn:.1f}%  {tag}")

        if rank is None or rank > 5:
            missed.append({
                "fname": fname, "expect": expect_cp,
                "rank": rank, "score": exp_hog,
            })

    print("=" * 100)
    print(f"总计: {total} 张")
    if total:
        print(f"  Top-1  命中率: {hit1}/{total}  ({hit1/total*100:.1f}%)")
        print(f"  Top-3  命中率: {hit3}/{total}  ({hit3/total*100:.1f}%)")
        print(f"  Top-5  命中率: {hit5}/{total}  ({hit5/total*100:.1f}%)")
        print(f"  Top-10 命中率: {hit10}/{total}  ({hit10/total*100:.1f}%)")
        print(f"  Top-20 命中率: {hit20}/{total}  ({hit20/total*100:.1f}%)")

    if missed:
        print("\n未进 Top-5 的：")
        for m in missed:
            rank_str = f"Top-{m['rank']}" if m["rank"] else "Top-20 外"
            print(f"  U+{m['expect']:04X}  {m['fname']:14s}  "
                  f"{rank_str}  期望分 {m['score']:.1f}%")


if __name__ == "__main__":
    main()