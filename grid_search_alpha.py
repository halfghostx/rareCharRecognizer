# grid_search_alpha.py
# -*- coding: utf-8 -*-
"""
Alpha 网格搜索：找出两阶段匹配中 CNN+HOG 加权融合的最优 alpha。
完全独立，不依赖 matcher.py 的当前状态。

用法：
    python grid_search_alpha.py
"""

import os
import re
import sys
import time
import numpy as np
import cv2

sys.path.insert(0, ".")
sys.path.insert(0, "src")
import config
from cnn_feature import CNNFeatureExtractor


# HOG 参数（必须和 build_templates.py 一致）
_HOG = cv2.HOGDescriptor(
    _winSize=(config.TEMPLATE_SIZE, config.TEMPLATE_SIZE),
    _blockSize=(16, 16),
    _blockStride=(8, 8),
    _cellSize=(8, 8),
    _nbins=12,
)


def crop_and_normalize(img_bgr, target_size):
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


def extract_features(img_bgr, extractor):
    resized = crop_and_normalize(img_bgr, config.TEMPLATE_SIZE)
    if resized is None:
        return None, None

    cnn_vec = extractor.extract(resized)
    if cnn_vec is None:
        return None, None

    hog_feat = _HOG.compute(resized).flatten().astype(np.float32)
    hog_norm = np.linalg.norm(hog_feat)
    if hog_norm < 1e-6:
        return None, None
    return cnn_vec, hog_feat / hog_norm


def parse_cp_from_filename(fname):
    m = re.search(r"[Uu]?([0-9A-Fa-f]{4,6})\.", fname)
    if not m:
        return None
    return int(m.group(1), 16)


def precompute(cnn_vec, hog_vec, cnn_t, hog_t, coarse_k):
    """预计算粗排 + HOG 分数（与 alpha 无关）"""
    cnn_scores = cnn_t @ cnn_vec
    k = min(coarse_k, len(cnn_scores))
    coarse_idx = np.argpartition(cnn_scores, -k)[-k:]
    cnn_sub = cnn_scores[coarse_idx]
    hog_scores = hog_t[coarse_idx] @ hog_vec
    return coarse_idx, cnn_sub, hog_scores


def evaluate_alpha(expect_cp, coarse_idx, cnn_sub, hog_scores,
                   codepoints, alpha, top_k=20):
    """给定 alpha，返回期望字符的排名（None = 未进 Top-K）"""
    cnn_min, cnn_max = cnn_sub.min(), cnn_sub.max()
    cnn_norm = (cnn_sub - cnn_min) / (cnn_max - cnn_min + 1e-9)

    hog_min, hog_max = hog_scores.min(), hog_scores.max()
    hog_norm = (hog_scores - hog_min) / (hog_max - hog_min + 1e-9)

    final = alpha * cnn_norm + (1 - alpha) * hog_norm

    k = min(top_k, len(final))
    top_local = np.argpartition(final, -k)[-k:]
    top_local = top_local[np.argsort(final[top_local])[::-1]]

    for rank, local_i in enumerate(top_local, 1):
        gi = coarse_idx[local_i]
        if int(codepoints[gi]) == expect_cp:
            return rank
    return None


def main():
    test_dir = os.path.join(config.BASE_DIR, "data", "test_batch")
    if not os.path.isdir(test_dir):
        print(f"[错误] 目录不存在: {test_dir}")
        sys.exit(1)

    files = sorted([
        f for f in os.listdir(test_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))
    ])
    if not files:
        print(f"[错误] {test_dir} 下没有图片")
        sys.exit(1)

    print("加载模板...")
    cnn_t = np.load(config.TEMPLATES_PATH)
    hog_t = np.load(config.HOG_TEMPLATES_PATH)
    codepoints = np.load(config.CODEPOINTS_PATH)
    print(f"  模板库: {len(codepoints)} 字，"
          f"CNN {cnn_t.shape[1]} 维，HOG {hog_t.shape[1]} 维")

    print("加载 CNN 模型...")
    extractor = CNNFeatureExtractor(config.CNN_MODEL_PATH)

    print("\n预计算测试图特征...")
    cache = []
    t0 = time.time()
    for fname in files:
        expect_cp = parse_cp_from_filename(fname)
        if expect_cp is None:
            continue
        img_path = os.path.join(test_dir, fname)
        data = np.fromfile(img_path, dtype=np.uint8)
        img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"  [跳过] 无法读取: {fname}")
            continue

        cnn_vec, hog_vec = extract_features(img_bgr, extractor)
        if cnn_vec is None:
            print(f"  [跳过] 预处理失败: {fname}")
            continue

        coarse_idx, cnn_sub, hog_scores = precompute(
            cnn_vec, hog_vec, cnn_t, hog_t, config.COARSE_TOP_K
        )
        cache.append({
            "fname": fname,
            "expect": expect_cp,
            "coarse_idx": coarse_idx,
            "cnn_sub": cnn_sub,
            "hog_scores": hog_scores,
        })

    print(f"  完成: {len(cache)} 张，用时 {time.time() - t0:.1f}s")

    alphas = [round(0.10 + 0.05 * i, 2) for i in range(7)]  # 0.10 ~ 0.40
    print(f"\nalpha 网格: {alphas}")
    print("=" * 88)

    results = {}

    for alpha in alphas:
        h1 = h3 = h5 = h10 = h20 = total = 0
        for item in cache:
            rank = evaluate_alpha(
                item["expect"], item["coarse_idx"],
                item["cnn_sub"], item["hog_scores"],
                codepoints, alpha, top_k=20,
            )
            total += 1
            if rank == 1:
                h1 += 1; h3 += 1; h5 += 1; h10 += 1; h20 += 1
            elif rank and rank <= 3:
                h3 += 1; h5 += 1; h10 += 1; h20 += 1
            elif rank and rank <= 5:
                h5 += 1; h10 += 1; h20 += 1
            elif rank and rank <= 10:
                h10 += 1; h20 += 1
            elif rank and rank <= 20:
                h20 += 1
        results[alpha] = {
            "h1": h1, "h3": h3, "h5": h5, "h10": h10,
            "h20": h20, "total": total,
        }

    # 打印表格
    print(f"\n{'alpha':>6s} | {'Top-1':>12s} | {'Top-3':>12s} | "
          f"{'Top-5':>12s} | {'Top-10':>12s} | {'Top-20':>12s}")
    print("-" * 88)

    best_alpha, best_score = None, -1
    for alpha in alphas:
        r = results[alpha]
        t = r["total"]
        s1 = r["h1"] / t * 100
        s3 = r["h3"] / t * 100
        s5 = r["h5"] / t * 100
        s10 = r["h10"] / t * 100
        s20 = r["h20"] / t * 100

        # 综合评分：Top-1 权重最大
        score = s1 * 10000 + s3 * 100 + s5 + s20 * 0.01
        marker = ""
        if score > best_score:
            best_score = score
            best_alpha = alpha
            marker = "  ← 当前最优"

        print(f"{alpha:>6.2f} | {r['h1']:>2d}/{t} {s1:>6.1f}% | "
              f"{r['h3']:>2d}/{t} {s3:>6.1f}% | "
              f"{r['h5']:>2d}/{t} {s5:>6.1f}% | "
              f"{r['h10']:>2d}/{t} {s10:>6.1f}% | "
              f"{r['h20']:>2d}/{t} {s20:>6.1f}%{marker}")

    print("=" * 88)

    r = results[best_alpha]
    t = r["total"]
    print(f"\n🏆 最优 alpha = {best_alpha}")
    print(f"   Top-1  : {r['h1']}/{t} ({r['h1']/t*100:.1f}%)")
    print(f"   Top-3  : {r['h3']}/{t} ({r['h3']/t*100:.1f}%)")
    print(f"   Top-5  : {r['h5']}/{t} ({r['h5']/t*100:.1f}%)")
    print(f"   Top-10 : {r['h10']}/{t} ({r['h10']/t*100:.1f}%)")
    print(f"   Top-20 : {r['h20']}/{t} ({r['h20']/t*100:.1f}%)")
    print(f"\n把 {best_alpha} 写进 matcher.py 的 recognize 默认值即可。")


if __name__ == "__main__":
    main()