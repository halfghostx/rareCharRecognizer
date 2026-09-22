# src/matcher.py
# -*- coding: utf-8 -*-
"""
两阶段匹配：
  1. 粗排：DINOv2 特征从全量模板中取 Top-COARSE_TOP_K
  2. 精排：在候选池里做 CNN + HOG 的 alpha 加权融合
"""

import os
import sys
import numpy as np
import cv2

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

_cnn_templates = None
_hog_templates = None
_codepoints = None
_extractor = None


def load_templates():
    global _cnn_templates, _hog_templates, _codepoints, _extractor
    if _cnn_templates is None:
        if not os.path.exists(config.TEMPLATES_PATH):
            raise FileNotFoundError(
                f"CNN 模板不存在: {config.TEMPLATES_PATH}\n"
                f"请先运行 python src/build_templates.py"
            )
        if not os.path.exists(config.HOG_TEMPLATES_PATH):
            raise FileNotFoundError(
                f"HOG 模板不存在: {config.HOG_TEMPLATES_PATH}\n"
                f"请先运行 python src/build_templates.py"
            )
        _cnn_templates = np.load(config.TEMPLATES_PATH)
        _hog_templates = np.load(config.HOG_TEMPLATES_PATH)
        _codepoints = np.load(config.CODEPOINTS_PATH)
    if _extractor is None:
        _extractor = CNNFeatureExtractor(config.CNN_MODEL_PATH)
    return _cnn_templates, _hog_templates, _codepoints


def _crop_and_normalize(img_bgr, target_size):
    if len(img_bgr.shape) == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    coords = cv2.findNonZero(closed)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    if w < 4 or h < 4:
        return None

    pad = 5
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(gray.shape[1] - x, w + 2 * pad)
    h = min(gray.shape[0] - y, h + 2 * pad)
    crop = gray[y:y + h, x:x + w]

    side = max(w, h)
    pad_x = (side - w) // 2
    pad_y = (side - h) // 2
    square = cv2.copyMakeBorder(
        crop, pad_y, side - h - pad_y, pad_x, side - w - pad_x,
        cv2.BORDER_CONSTANT, value=255,
    )

    resized = cv2.resize(
        square, (target_size, target_size),
        interpolation=cv2.INTER_AREA,
    )
    if resized.mean() < 127:
        resized = 255 - resized
    return resized


def preprocess_query(img_bgr, target_size=None, debug=False):
    """返回 (cnn_vec, hog_vec, debug_img)"""
    if target_size is None:
        target_size = config.TEMPLATE_SIZE
    if img_bgr is None or img_bgr.size == 0:
        return None, None, None

    resized = _crop_and_normalize(img_bgr, target_size)
    if resized is None:
        return None, None, None

    load_templates()
    cnn_vec = _extractor.extract(resized)
    if cnn_vec is None:
        return None, None, None

    hog_feat = _HOG.compute(resized).flatten().astype(np.float32)
    hog_norm = np.linalg.norm(hog_feat)
    if hog_norm < 1e-6:
        return None, None, None
    hog_vec = hog_feat / hog_norm

    if debug:
        return cnn_vec, hog_vec, resized
    return cnn_vec, hog_vec, None


def recognize(cnn_vec, hog_vec, top_k=None, coarse_k=None,
              min_score=None, alpha=0.15):
    """
    两阶段匹配 + alpha 加权融合：
      阶段 1：DINOv2 粗排，取 Top-COARSE_K
      阶段 2：候选池里 min-max 归一化后做 alpha 融合，返回 Top-K
    """
    if top_k is None:
        top_k = config.TOP_K
    if coarse_k is None:
        coarse_k = config.COARSE_TOP_K
    if min_score is None:
        min_score = config.MIN_SCORE

    cnn_templates, hog_templates, codepoints = load_templates()

    # ---- 阶段 1：DINOv2 粗排 ----
    cnn_scores = cnn_templates @ cnn_vec
    k = min(coarse_k, len(cnn_scores))
    coarse_idx = np.argpartition(cnn_scores, -k)[-k:]

    # ---- 阶段 2：候选池里 CNN + HOG 融合 ----
    cnn_sub = cnn_scores[coarse_idx]
    hog_sub = hog_templates[coarse_idx] @ hog_vec

    cnn_min, cnn_max = cnn_sub.min(), cnn_sub.max()
    cnn_norm = (cnn_sub - cnn_min) / (cnn_max - cnn_min + 1e-9)

    hog_min, hog_max = hog_sub.min(), hog_sub.max()
    hog_norm = (hog_sub - hog_min) / (hog_max - hog_min + 1e-9)

    final_scores = alpha * cnn_norm + (1 - alpha) * hog_norm

    sorted_local = np.argsort(final_scores)[::-1]
    top_local = sorted_local[:top_k]

    results = []
    for local_i in top_local:
        global_i = coarse_idx[local_i]
        final_score = float(final_scores[local_i])
        hog_score = float(hog_sub[local_i])
        cnn_score = float(cnn_scores[global_i])
        if final_score < min_score:
            continue
        cp = int(codepoints[global_i])
        results.append({
            "codepoint": f"{cp:04X}",
            "score": final_score,
            "hog_score": hog_score,
            "cnn_score": cnn_score,
            "yedict_url": config.YEDICT_URL_TEMPLATE.format(
                codepoint=f"{cp:04X}"
            ),
        })
    return results


def _main():
    if len(sys.argv) < 2:
        print("用法: python src/matcher.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(f"[错误] 文件不存在: {img_path}")
        sys.exit(1)

    data = np.fromfile(img_path, dtype=np.uint8)
    img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"[错误] 无法读取图片: {img_path}")
        sys.exit(1)

    print(f"图片尺寸: {img_bgr.shape}")
    cnn_vec, hog_vec, dbg = preprocess_query(img_bgr, debug=True)
    if cnn_vec is None:
        print("[错误] 预处理失败")
        sys.exit(1)

    out_path = os.path.join(config.BASE_DIR, "data", "last_preprocessed.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, dbg)
    print(f"预处理结果已存至: {out_path}")

    results = recognize(cnn_vec, hog_vec)
    if not results:
        print("没有匹配到任何候选")
        sys.exit(1)

    print(f"\nTop {len(results)} 候选（两阶段 + alpha=0.15）:")
    for i, r in enumerate(results, 1):
        print(f"  {i:2d}. U+{r['codepoint']:>6s}  "
              f"融合 {r['score'] * 100:6.2f}%  "
              f"HOG {r['hog_score'] * 100:6.2f}%  "
              f"CNN {r['cnn_score'] * 100:6.2f}%  "
              f"{r['yedict_url']}")


if __name__ == "__main__":
    _main()