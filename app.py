# app.py
# -*- coding: utf-8 -*-
"""
生僻字识别 Web 应用。
启动: python app.py
访问: http://127.0.0.1:5000
"""

import os
import io
import sys
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory, send_file
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, ".")
sys.path.insert(0, "src")
import config
from matcher import load_templates, preprocess_query, recognize

app = Flask(__name__, static_folder="static")

# ============================================================
# 启动时预热
# ============================================================

print("[启动] 加载模板库...")
load_templates()
print("[启动] 就绪")

# ============================================================
# 字形渲染（用于前端展示候选）
# ============================================================

_render_font = None


def get_render_font_path():
    """优先找 P0 字体文件用于渲染字形。"""
    if not os.path.isdir(config.FONT_DIR):
        return None
    candidates = []
    for f in sorted(os.listdir(config.FONT_DIR)):
        if not f.lower().endswith((".ttf", ".otf")):
            continue
        path = os.path.join(config.FONT_DIR, f)
        # P0 优先
        if "P0" in f.upper():
            return path
        candidates.append(path)
    return candidates[0] if candidates else None


def get_render_font(size=128):
    global _render_font
    if _render_font is None:
        path = get_render_font_path()
        if path is None:
            return None
        _render_font = ImageFont.truetype(path, size - 16)
    return _render_font


def render_glyph_png(cp, size=128):
    """用天珩字库渲染单字为 PNG，返回 BytesIO 或 None。"""
    font = get_render_font(size)
    if font is None:
        return None
    img = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(img)
    char = chr(cp)
    bbox = font.getbbox(char)
    if bbox is None:
        return None
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    if w <= 0 or h <= 0:
        return None
    x = (size - w) // 2 - bbox[0]
    y = (size - h) // 2 - bbox[1]
    draw.text((x, y), char, font=font, fill=0)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


# ============================================================
# 路由
# ============================================================

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/glyph/<cp_hex>")
def glyph(cp_hex):
    """返回码点对应字形的 PNG。"""
    try:
        cp = int(cp_hex, 16)
    except ValueError:
        return "invalid codepoint", 400
    buf = render_glyph_png(cp)
    if buf is None:
        return "glyph not available", 404
    return send_file(buf, mimetype="image/png")


@app.route("/recognize", methods=["POST"])
def api_recognize():
    """接收上传图片，返回 Top-20 候选。"""
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "未收到图片"}), 400

    data = file.read()
    if not data:
        return jsonify({"error": "图片为空"}), 400

    img_array = np.frombuffer(data, dtype=np.uint8)
    img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return jsonify({"error": "无法解析图片格式"}), 400

    cnn_vec, hog_vec, _ = preprocess_query(img_bgr, debug=False)
    if cnn_vec is None:
        return jsonify({
            "error": "无法从图片中提取文字区域，请上传更清晰的单字截图"
        }), 400

    results = recognize(cnn_vec, hog_vec, top_k=20, min_score=0.0)
    return jsonify({"results": results})


if __name__ == "__main__":
    print("[启动] 访问 http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)