# check_and_match.py
# -*- coding: utf-8 -*-
"""
验证脚本：检查码点是否在模板库 + 执行匹配。
"""

import os
import sys
import argparse
import numpy as np
import cv2

sys.path.insert(0, ".")
sys.path.insert(0, "src")
import config
from matcher import load_templates, preprocess_query, recognize


def parse_codepoint(s):
    s = s.strip().lower()
    if s.startswith("u+"):
        s = s[2:]
    elif s.startswith("0x"):
        s = s[2:]
    elif s.startswith("u"):
        s = s[1:]
    return int(s, 16)


def check_in_library(cp, codepoints):
    idx = np.where(codepoints == cp)[0]
    if len(idx) == 0:
        return False, None
    return True, int(idx[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("target", help="U 码 或 图片路径")
    parser.add_argument("--expect", help="期望的 U 码（可选）")
    parser.add_argument("--check", action="store_true",
                        help="只检查码点是否存在")
    args = parser.parse_args()

    cnn_t, hog_t, codepoints = load_templates()
    print(f"模板库: {len(codepoints)} 条记录，唯一码点 "
          f"{len(np.unique(codepoints))} 个")
    print()

    # ---- 模式 1：只检查码点 ----
    if args.check:
        cp = parse_codepoint(args.target)
        exists, idx = check_in_library(cp, codepoints)
        if exists:
            print(f"✅ U+{cp:04X} 在模板库中")
        else:
            print(f"❌ U+{cp:04X} 不在模板库中")
        return

    # 判断 target 是图片还是 U 码
    is_image = os.path.isfile(args.target) and args.target.lower().endswith(
        (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
    )

    # ---- 模式 2：图片匹配 ----
    if is_image:
        expect_cp = None
        if args.expect:
            expect_cp = parse_codepoint(args.expect)
            exists, _ = check_in_library(expect_cp, codepoints)
            if exists:
                print(f"✅ 期望字符 U+{expect_cp:04X} 在模板库中")
            else:
                print(f"❌ 期望字符 U+{expect_cp:04X} 不在模板库中")
            print()

        data = np.fromfile(args.target, dtype=np.uint8)
        img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"无法读取图片: {args.target}")
            sys.exit(1)
        print(f"图片尺寸: {img_bgr.shape}")

        cnn_vec, hog_vec, dbg = preprocess_query(img_bgr, debug=True)
        if cnn_vec is None:
            print("预处理失败")
            sys.exit(1)

        out_path = os.path.join(
            config.BASE_DIR, "data", "last_preprocessed.png"
        )
        cv2.imwrite(out_path, dbg)
        print(f"预处理结果已存至: {out_path}")
        print()

        results = recognize(cnn_vec, hog_vec, top_k=20, min_score=0.0)

        print(f"Top {len(results)} 候选:")
        expect_rank = None
        for i, r in enumerate(results, 1):
            marker = ""
            if expect_cp is not None and int(r["codepoint"], 16) == expect_cp:
                marker = "  <-- 期望结果 ✅"
                expect_rank = i
            print(f"  {i:2d}. U+{r['codepoint']:>6s}  "
                  f"融合 {r['score'] * 100:6.2f}%  "
                  f"HOG {r['hog_score'] * 100:6.2f}%  "
                  f"CNN {r['cnn_score'] * 100:6.2f}%{marker}")

        print()
        if expect_cp is not None:
            if expect_rank:
                print(f"🎯 期望字符 U+{expect_cp:04X} 排在第 {expect_rank} 位")
            else:
                print(f"❌ 期望字符 U+{expect_cp:04X} 不在 Top-20 中")
        return

    # ---- 模式 3：自匹配 ----
    try:
        cp = parse_codepoint(args.target)
    except ValueError:
        print(f"无法解析: {args.target}")
        sys.exit(1)

    exists, idx = check_in_library(cp, codepoints)
    if not exists:
        print(f"❌ U+{cp:04X} 不在模板库中")
        sys.exit(1)

    print(f"✅ U+{cp:04X} 在模板库中（索引 {idx}）")
    print()

    query_vec = cnn_t[idx]
    query_hog = hog_t[idx]
    results = recognize(query_vec, query_hog, top_k=10, min_score=0.0)

    print(f"自匹配 Top {len(results)}:")
    for i, r in enumerate(results, 1):
        marker = ""
        if int(r["codepoint"], 16) == cp:
            marker = "  <-- 自己 ✅"
        print(f"  {i:2d}. U+{r['codepoint']:>6s}  "
              f"融合 {r['score'] * 100:6.2f}%{marker}")


if __name__ == "__main__":
    main()