# src/build_templates.py
# -*- coding: utf-8 -*-
"""
从字体文件批量提取字形，生成归一化向量模板库。

用法（项目根目录下执行）：
    python src/build_templates.py

输出：
    data/templates.npy   shape=(N, 4096) float32
    data/codepoints.npy  shape=(N,) int32
    data/samples/*.png   随机抽检样本，用于目视验证
"""

import os
import sys
import time
import random
import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def render_glyph(pil_font, char, size):
    """将单个字符渲染为 size×size 灰度数组（黑字白底）。"""
    img = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(img)

    bbox = pil_font.getbbox(char)
    if bbox is None:
        return None

    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    if w <= 0 or h <= 0:
        return None

    x = (size - w) // 2 - bbox[0]
    y = (size - h) // 2 - bbox[1]
    draw.text((x, y), char, font=pil_font, fill=0)

    return np.array(img, dtype=np.float32)


def normalize_vector(arr):
    """展平 + 去均值 + L2 归一化。"""
    vec = arr.flatten()
    vec = vec - vec.mean()
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        return None
    return (vec / norm).astype(np.float32)


def build_plane(font_path, codepoint_range, template_size, font_size):
    """处理单个字体文件，返回 (vectors, codepoints)。"""
    print(f"\n加载字体: {font_path}")
    font = TTFont(font_path)
    cmap = font.getBestCmap()
    print(f"  字体 cmap 包含 {len(cmap)} 个字形")

    pil_font = ImageFont.truetype(font_path, font_size)

    vectors, codepoints = [], []
    total = len(cmap)
    t0 = time.time()

    for i, (cp, _glyph_name) in enumerate(cmap.items(), 1):
        if codepoint_range is not None:
            lo, hi = codepoint_range
            if not (lo <= cp <= hi):
                continue

        arr = render_glyph(pil_font, chr(cp), template_size)
        if arr is None:
            continue

        vec = normalize_vector(arr)
        if vec is None:
            continue

        vectors.append(vec)
        codepoints.append(cp)

        if i % 2000 == 0:
            elapsed = time.time() - t0
            print(f"  扫描 {i}/{total}  收集 {len(codepoints)}  "
                  f"({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"  完成: 收集 {len(codepoints)} 个字形，用时 {elapsed:.1f}s")

    if not vectors:
        return (np.zeros((0, template_size * template_size), dtype=np.float32),
                np.zeros((0,), dtype=np.int32))

    return (np.array(vectors, dtype=np.float32),
            np.array(codepoints, dtype=np.int32))


def save_samples_for_plane(codepoints, font_path, template_size,
                           font_size, plane_name, n=8):
    """
    从某个平面的码点中随机抽 n 个，用原始字体重新渲染并存为 PNG。
    直接渲染而非从归一化向量反推，保证对比度正常。
    """
    sample_dir = os.path.join(config.BASE_DIR, "data", "samples")
    os.makedirs(sample_dir, exist_ok=True)

    if len(codepoints) == 0:
        return

    pil_font = ImageFont.truetype(font_path, font_size)
    idxs = random.sample(range(len(codepoints)), min(n, len(codepoints)))

    saved = 0
    for idx in idxs:
        cp = int(codepoints[idx])
        arr = render_glyph(pil_font, chr(cp), template_size)
        if arr is None:
            continue
        arr_uint8 = np.clip(arr, 0, 255).astype(np.uint8)
        filename = f"{plane_name}_U{cp:04X}.png"
        Image.fromarray(arr_uint8).save(os.path.join(sample_dir, filename))
        saved += 1

    print(f"  已保存 {saved} 个抽检样本 ({plane_name})")


def main():
    print("=" * 60)
    print("开始生成字形模板库")
    print("=" * 60)
    print(f"模板尺寸 : {config.TEMPLATE_SIZE} × {config.TEMPLATE_SIZE}")
    print(f"字体字号 : {config.FONT_SIZE}")
    print(f"启用平面 : {len(config.PLANES)} 个")

    all_vectors, all_codepoints = [], []

    for plane in config.PLANES:
        font_path = os.path.join(config.FONT_DIR, plane["file"])
        if not os.path.exists(font_path):
            print(f"\n[跳过] 字体文件不存在: {font_path}")
            continue

        vectors, codepoints = build_plane(
            font_path,
            plane.get("range"),
            config.TEMPLATE_SIZE,
            config.FONT_SIZE,
        )
        all_vectors.append(vectors)
        all_codepoints.append(codepoints)

        # 每个平面抽检样本，直接重新渲染
        plane_name = plane["file"].replace(".ttf", "")
        save_samples_for_plane(
            codepoints, font_path,
            config.TEMPLATE_SIZE, config.FONT_SIZE,
            plane_name, n=8,
        )

    if not all_vectors or all(len(v) == 0 for v in all_vectors):
        print("\n[错误] 没有提取到任何字形，请检查字体文件与配置。")
        sys.exit(1)

    V = np.vstack(all_vectors)
    C = np.concatenate(all_codepoints)

    print("\n" + "=" * 60)
    print("汇总")
    print("=" * 60)
    print(f"总字形数 : {len(C)}")
    print(f"向量矩阵 : {V.shape}  dtype={V.dtype}")
    print(f"内存占用 : {V.nbytes / 1024 / 1024:.1f} MB (float32)")

    os.makedirs(os.path.dirname(config.TEMPLATES_PATH), exist_ok=True)
    np.save(config.TEMPLATES_PATH, V)
    np.save(config.CODEPOINTS_PATH, C)
    print(f"\n已保存模板 : {config.TEMPLATES_PATH}")
    print(f"已保存码点 : {config.CODEPOINTS_PATH}")

    sample_dir = os.path.join(config.BASE_DIR, "data", "samples")
    print(f"\n抽检样本目录: {sample_dir}")
    print("请打开该目录，确认字形正常（不偏移、不空白、不裁边）")

    print("\n完成。")


if __name__ == "__main__":
    main()