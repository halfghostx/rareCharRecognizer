# src/build_templates.py
# -*- coding: utf-8 -*-
"""
生成两套模板库：
  - DINOv2 CNN 特征 (384 维) —— 用于粗排
  - HOG 特征 (2352 维)       —— 用于精排
"""

import os
import sys
import time
import random
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


def load_fixed_codepoints(path):
    cps = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                cps.append(int(line, 16))
            except ValueError:
                print(f"  [警告] 无法解析: {line}")
    return cps


def build_plane(extractor, font_path, codepoint_range, sample_size,
                template_size, font_size, explicit_cps=None):
    print(f"\n加载字体: {font_path}")
    font = TTFont(font_path)
    cmap = font.getBestCmap()
    print(f"  字体 cmap 包含 {len(cmap)} 个字形")

    pil_font = ImageFont.truetype(font_path, font_size)

    if explicit_cps is not None:
        target_cps = [cp for cp in explicit_cps if cp in cmap]
        print(f"  使用固定码点: {len(target_cps)} 个")
    else:
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
        if sample_size and sample_size < len(all_cps):
            step = len(all_cps) / sample_size
            indices = [int(i * step) for i in range(sample_size)]
            target_cps = [all_cps[i] for i in indices]
            print(f"  均匀采样: {len(target_cps)} 个")
        else:
            target_cps = all_cps

    cnn_feats, hog_feats, codepoints = [], [], []
    t0 = time.time()
    for i, cp in enumerate(target_cps, 1):
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
        if i % 100 == 0:
            elapsed = time.time() - t0
            print(f"  处理 {i}/{len(target_cps)}  收集 {len(codepoints)}  "
                  f"({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"  完成: 收集 {len(codepoints)} 个字形，用时 {elapsed:.1f}s")

    if not cnn_feats:
        return None, None, None
    return (np.array(cnn_feats, dtype=np.float32),
            np.array(hog_feats, dtype=np.float32),
            np.array(codepoints, dtype=np.int32))


def save_samples_for_plane(codepoints, font_path, template_size,
                            font_size, plane_name, n=8):
    sample_dir = os.path.join(config.BASE_DIR, "data", "samples")
    os.makedirs(sample_dir, exist_ok=True)
    if len(codepoints) == 0:
        return
    pil_font = ImageFont.truetype(font_path, font_size)
    idxs = random.sample(range(len(codepoints)), min(n, len(codepoints)))
    saved = 0
    for idx in idxs:
        cp = int(codepoints[idx])
        gray = render_glyph(pil_font, chr(cp), template_size)
        if gray is None:
            continue
        filename = f"{plane_name}_U{cp:04X}.png"
        Image.fromarray(gray).save(os.path.join(sample_dir, filename))
        saved += 1
    print(f"  已保存 {saved} 个抽检样本 ({plane_name})")


def main():
    print("=" * 60)
    print("生成双特征模板库（DINOv2 粗排 + HOG 精排）")
    print("=" * 60)

    explicit_cps = None
    if os.path.exists(config.SAMPLE_CODEPOINTS_FILE):
        explicit_cps = load_fixed_codepoints(config.SAMPLE_CODEPOINTS_FILE)
        print(f"从文件加载固定码点: {len(explicit_cps)} 个")
    else:
        print("未找到固定码点文件，使用范围模式")

    extractor = CNNFeatureExtractor(config.CNN_MODEL_PATH)

    all_cnn, all_hog, all_cps = [], [], []
    for plane in config.PLANES:
        font_path = os.path.join(config.FONT_DIR, plane["file"])
        if not os.path.exists(font_path):
            print(f"\n[跳过] 字体文件不存在: {font_path}")
            continue

        cnn_f, hog_f, cps = build_plane(
            extractor, font_path, plane.get("range"),
            plane.get("sample"),
            config.TEMPLATE_SIZE, config.FONT_SIZE,
            explicit_cps=explicit_cps,
        )
        if cnn_f is None:
            continue

        all_cnn.append(cnn_f)
        all_hog.append(hog_f)
        all_cps.append(cps)

        plane_name = plane["file"].replace(".ttf", "")
        save_samples_for_plane(
            cps, font_path, config.TEMPLATE_SIZE, config.FONT_SIZE,
            plane_name, n=8,
        )

    if not all_cnn:
        print("\n[错误] 没有提取到任何字形。")
        sys.exit(1)

    CNN_V = np.vstack(all_cnn)
    HOG_V = np.vstack(all_hog)
    C = np.concatenate(all_cps)

    print("\n" + "=" * 60)
    print("汇总")
    print("=" * 60)
    print(f"总字形数     : {len(C)}")
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