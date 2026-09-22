# diagnose.py
import os
import sys
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, ".")
sys.path.insert(0, "src")
import config
from build_templates import render_glyph


# 1. 用模板端的方式渲染 U+4B6D
FONT_PATH = os.path.join(config.FONT_DIR, "TH-Tshyn-P0.ttf")
pil_font = ImageFont.truetype(FONT_PATH, config.FONT_SIZE)
template_img = render_glyph(pil_font, chr(0x4B6D), config.TEMPLATE_SIZE)

# 2. 读取查询端预处理结果
query_img = cv2.imread("data/last_preprocessed.png", cv2.IMREAD_GRAYSCALE)

if template_img is None:
    print("[错误] 模板渲染失败")
    sys.exit(1)
if query_img is None:
    print("[错误] 无法读取 data/last_preprocessed.png")
    sys.exit(1)

print(f"模板图 shape: {template_img.shape}, dtype: {template_img.dtype}")
print(f"查询图 shape: {query_img.shape}, dtype: {query_img.dtype}")

# 3. 像素级差异
if template_img.shape == query_img.shape:
    diff = np.abs(
        template_img.astype(np.float32) -
        query_img.astype(np.float32)
    )
    print(f"像素平均差异: {diff.mean():.2f}")
    print(f"像素最大差异: {diff.max():.0f}")

    # 像素级余弦相似度
    t = template_img.flatten().astype(np.float32)
    q = query_img.flatten().astype(np.float32)
    t = t - t.mean()
    q = q - q.mean()
    cos = (t @ q) / (np.linalg.norm(t) * np.linalg.norm(q) + 1e-9)
    print(f"像素级余弦相似度: {cos:.4f}")

# 4. 并排放大保存
combined = np.hstack([
    template_img,
    np.full((64, 4), 128, dtype=np.uint8),
    query_img,
])
combined = cv2.resize(
    combined, (combined.shape[1] * 6, combined.shape[0] * 6),
    interpolation=cv2.INTER_NEAREST,
)
cv2.imwrite("data/diagnose_compare.png", combined)
print("\n对比图已保存: data/diagnose_compare.png")
print("  左：模板渲染的 U+4B6D")
print("  右：你的截图的预处理结果")