# check_and_match.py
# -*- coding: utf-8 -*-
"""
验证脚本：检查码点是否在模板库 + 执行匹配。

用法：
  # 自匹配：检查 U+4B6D 是否在库，并用它自己的特征向量做匹配
  python check_and_match.py 4B6D

  # 图片匹配：用截图匹配，并标出期望 U 码的排名
  python check_and_match.py data/samples/U4B6D.png --expect 4B6D

  # 只检查是否存在
  python check_and_match.py --check 4B6D
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
    """支持 4B6D / U+4B6D / 0x4B6D / u4b6d 等格式"""
    s = s.strip().lower()
    if s.startswith("u+"):
        s = s[2:]
    elif s.startswith("0x"):
        s = s[2:]
    elif s.startswith("u"):
        s = s[1:]
    return int(s, 16)


def check_in_library(cp, codepoints):
    """检查码点是否在模板库，返回 (是否存在, 索引)"""
    idx = np.where(codepoints == cp)[0]
    if len(idx) == 0:
        return False, None
    return True, int(idx[0])


def main():
    parser = argparse.ArgumentParser(
        description="检查码点是否在模板库，并执行匹配验证"
    )
    parser.add_argument("target", help="U 码 或 图片路径")
    parser.add_argument("--expect", help="期望的 U 码（图片模式下使用）")
    parser.add_argument("--check", action="store_true",
                        help="只检查码点是否存在")
    args = parser.parse_args()

    # 加载模板库
    templates, codepoints = load_templates()
    print(f"模板库: {len(codepoints)} 个字形，特征维度 {templates.shape[1]}")
    print()

    # ---------- 模式 1：只检查码点 ----------
    if args.check:
        cp = parse_codepoint(args.target)
        exists, idx = check_in_library(cp, codepoints)
        if exists:
            print(f"✅ U+{cp:04X} 在模板库中（索引 {idx}）")
        else:
            print(f"❌ U+{cp:04X} 不在模板库中")
        return

    # 判断 target 是图片还是 U 码
    is_image = os.path.isfile(args.target) and args.target.lower().endswith(
        (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
    )

    # ---------- 模式 2：图片匹配 ----------
    if is_image:
        if not args.expect:
            print("图片模式必须指定 --expect 参数")
            sys.exit(1)

        expect_cp = parse_codepoint(args.expect)

        # 先检查期望字符是否在库
        exists, expect_idx = check_in_library(expect_cp, codepoints)
        if exists:
            print(f"✅ 期望字符 U+{expect_cp:04X} 在模板库中"
                  f"（索引 {expect_idx}）")
        else:
            print(f"❌ 期望字符 U+{expect_cp:04X} 不在模板库中")
            print("   即使匹配结果正确，也无法命中该字")
        print()

        # 读取图片
        data = np.fromfile(args.target, dtype=np.uint8)
        img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"无法读取图片: {args.target}")
            sys.exit(1)
        print(f"图片尺寸: {img_bgr.shape}")

        # 预处理
        vec, dbg = preprocess_query(img_bgr, debug=True)
        if vec is None:
            print("预处理失败")
            sys.exit(1)

        out_path = os.path.join(
            config.BASE_DIR, "data", "last_preprocessed.png"
        )
        cv2.imwrite(out_path, dbg)
        print(f"预处理结果已存至: {out_path}")
        print()

        # 匹配 Top-20
        results = recognize(vec, top_k=20, min_score=0.0)

        print(f"Top {len(results)} 候选:")
        expect_rank = None
        for i, r in enumerate(results, 1):
            marker = ""
            if int(r["codepoint"], 16) == expect_cp:
                marker = "  <-- 期望结果 ✅"
                expect_rank = i
            print(f"  {i:2d}. U+{r['codepoint']:>6s}  "
                  f"相似度 {r['score'] * 100:6.2f}%{marker}")

        print()
        if expect_rank:
            print(f"🎯 期望字符 U+{expect_cp:04X} 排在第 {expect_rank} 位")
        else:
            print(f"❌ 期望字符 U+{expect_cp:04X} 不在 Top-20 中")
        return

    # ---------- 模式 3：自匹配 ----------
    try:
        cp = parse_codepoint(args.target)
    except ValueError:
        print(f"无法解析: {args.target}")
        print("用法:")
        print("  python check_and_match.py 4B6D  # 自匹配")
        print("  python check_and_match.py image.png --expect 4B6D")
        sys.exit(1)

    exists, idx = check_in_library(cp, codepoints)
    if not exists:
        print(f"❌ U+{cp:04X} 不在模板库中")
        sys.exit(1)

    print(f"✅ U+{cp:04X} 在模板库中（索引 {idx}）")
    print()

    # 自匹配：用模板自己的特征向量去匹配
    query_vec = templates[idx]
    results = recognize(query_vec, top_k=10, min_score=0.0)

    print(f"自匹配 Top {len(results)}:")
    for i, r in enumerate(results, 1):
        marker = ""
        if int(r["codepoint"], 16) == cp:
            marker = "  <-- 自己 ✅"
        print(f"  {i:2d}. U+{r['codepoint']:>6s}  "
              f"相似度 {r['score'] * 100:6.2f}%{marker}")


if __name__ == "__main__":
    main()

'''
检查某个码点是否在模板库：
python check_and_match.py --check 4B6D
预期输出 ✅ U+4B6D 在模板库中 或 ❌

自匹配验证（不用截图，直接验证模板库质量）：
python check_and_match.py 4B6D
预期 Top-1 是它自己，相似度 100%。如果这个不对，说明模板库构建有问题。

图片匹配验证（你现在最需要的）：
python check_and_match.py data/samples/U4B6D.png --expect 4B6D
输出会做三件事：

先检查 U+4B6D 是否在库

匹配你的截图

显示 Top-20 候选，期望字符的位置会用 ✅ 标出
'''