# src/build_templates.py
# -*- coding: utf-8 -*-
"""
多字体模板生成：
  - 遍历 PLANES 里的每套字体
  - 每套字体独立渲染、提取 CNN + HOG 特征
  - 最后 vstack 合并成一套模板
"""

import os
import sys
import time
import numpy as np
import cv2
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
sys.path.insert(0, config.BASE_DIR)
from cnn_feature import CNNFeatureExtractor


_HOG = cv2.HOGDescriptor(
    _winSize=(config.TEMPLATE_SIZE, config.TEMPLATE_SIZE),
    _blockSize=(16, 16),
    _blockStride=(8, 8),
    _cellSize=(8, 8),
    _nbins=12,
)


def render_glyph(pil_font, char, size):
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
    return np.array(img, dtype=np.uint8)


def extract_hog(gray_img):
    feat = _HOG.compute(gray_img).flatten().astype(np.float32)
    norm = np.linalg.norm(feat)
    if norm < 1e-6:
        return None
    return feat / norm


def build_one_font(extractor, font_file, codepoint_range, note,
                   template_size, font_size):
    print(f"\n加载字体: {font_file}  [{note}]")
    if not os.path.exists(font_file):
        print(f"  [跳过] 文件不存在")
        return None

    font = TTFont(font_file)
    cmap = font.getBestCmap()
    print(f"  字体 cmap 包含 {len(cmap)} 个字形")

    pil_font = ImageFont.truetype(font_file, font_size)

    # 收集合法码点
    all_cps = []
    for cp in sorted(cmap.keys()):
        if codepoint_range is not None:
            lo, hi = codepoint_range
            if not (lo <= cp <= hi):
                continue
        if 0xE000 <= cp <= 0xF8FF:
            continue
        if 0xF900 <= cp <= 0xFAFF:
            continue
        all_cps.append(cp)
    print(f"  范围内合法码点: {len(all_cps)}")

    cnn_feats, hog_feats, codepoints = [], [], []
    t0 = time.time()
    for i, cp in enumerate(all_cps, 1):
        gray = render_glyph(pil_font, chr(cp), template_size)
        if gray is None:
            continue
        cnn_feat = extractor.extract(gray)
        if cnn_feat is None:
            continue
        hog_feat = extract_hog(gray)
        if hog_feat is None:
            continue
        cnn_feats.append(cnn_feat)
        hog_feats.append(hog_feat)
        codepoints.append(cp)
        if i % 1000 == 0:
            elapsed = time.time() - t0
            speed = elapsed / i * 1000
            print(f"  处理 {i}/{len(all_cps)}  收集 {len(codepoints)}  "
                  f"({elapsed:.1f}s, {speed:.1f}s/1000)")

    elapsed = time.time() - t0
    print(f"  完成: {len(codepoints)} 个字形，用时 {elapsed:.1f}s")

    if not cnn_feats:
        return None
    return (np.array(cnn_feats, dtype=np.float32),
            np.array(hog_feats, dtype=np.float32),
            np.array(codepoints, dtype=np.int32))


def main():
    print("=" * 60)
    print("多字体模板生成（DINOv2 + HOG）")
    print("=" * 60)

    extractor = CNNFeatureExtractor(config.CNN_MODEL_PATH)

    all_cnn, all_hog, all_cps = [], [], []

    for plane in config.PLANES:
        result = build_one_font(
            extractor,
            plane["font_file"],
            plane.get("range"),
            plane.get("note", ""),
            config.TEMPLATE_SIZE,
            config.FONT_SIZE,
        )
        if result is None:
            continue
        cnn_f, hog_f, cps = result
        all_cnn.append(cnn_f)
        all_hog.append(hog_f)
        all_cps.append(cps)

    if not all_cnn:
        print("\n[错误] 没有提取到任何字形。")
        sys.exit(1)

    CNN_V = np.vstack(all_cnn)
    HOG_V = np.vstack(all_hog)
    C = np.concatenate(all_cps)

    print("\n" + "=" * 60)
    print("汇总")
    print("=" * 60)
    print(f"总记录数     : {len(C)}（含多字体重复）")
    print(f"唯一码点数   : {len(np.unique(C))}")
    print(f"DINOv2 特征  : {CNN_V.shape}  "
          f"{CNN_V.nbytes / 1024 / 1024:.1f} MB")
    print(f"HOG 特征     : {HOG_V.shape}  "
          f"{HOG_V.nbytes / 1024 / 1024:.1f} MB")

    os.makedirs(os.path.dirname(config.TEMPLATES_PATH), exist_ok=True)
    np.save(config.TEMPLATES_PATH, CNN_V)
    np.save(config.HOG_TEMPLATES_PATH, HOG_V)
    np.save(config.CODEPOINTS_PATH, C)

    print(f"\n已保存 CNN 模板 : {config.TEMPLATES_PATH}")
    print(f"已保存 HOG 模板 : {config.HOG_TEMPLATES_PATH}")
    print(f"已保存码点      : {config.CODEPOINTS_PATH}")
    print("\n完成。")


if __name__ == "__main__":
    main()