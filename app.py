# app.py
# -*- coding: utf-8 -*-
"""
生僻字识别桌面应用。
双击 exe 后启动 Flask，自动打开浏览器。
"""

import os
import sys
import io
import threading
import webbrowser
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory, send_file
from PIL import Image, ImageDraw, ImageFont


def get_base_path():
    """获取资源根路径（打包后 vs 开发时）。"""
    if getattr(sys, "frozen", False):
        # PyInstaller 打包后
        return sys._MEIPASS
    else:
        # 开发时，项目根目录
        return os.path.dirname(os.path.abspath(__file__))


BASE_PATH = get_base_path()

# 让 config 和 matcher 能被正确导入
sys.path.insert(0, BASE_PATH)
sys.path.insert(0, os.path.join(BASE_PATH, "src"))

import config
from matcher import load_templates, preprocess_query, recognize


# ============================================================
# Flask 应用
# ============================================================

# static 文件夹打包后位置会变，用绝对路径
static_dir = os.path.join(BASE_PATH, "static")
app = Flask(__name__, static_folder=static_dir)


print("[启动] 加载模板库...")
load_templates()
print("[启动] 就绪")


_render_font = None


def get_render_font_path():
    """找 P0 字体文件用于渲染候选字形。"""
    font_dir = config.FONT_DIR
    if not os.path.isdir(font_dir):
        return None
    candidates = []
    for f in sorted(os.listdir(font_dir)):
        if not f.lower().endswith((".ttf", ".otf")):
            continue
        path = os.path.join(font_dir, f)
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


@app.route("/")
def index():
    return send_from_directory(static_dir, "index.html")


@app.route("/glyph/<cp_hex>")
def glyph(cp_hex):
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


def open_browser():
    """延迟 1.5 秒打开浏览器，给 Flask 启动留时间。"""
    import time
    time.sleep(1.5)
    webbrowser.open("http://127.0.0.1:5000")


if __name__ == "__main__":
    # 打包后不打印 Werkzeug 启动日志
    import logging
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.WARNING)

    # 自动打开浏览器
    threading.Thread(target=open_browser, daemon=True).start()

    print("[启动] 访问 http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)