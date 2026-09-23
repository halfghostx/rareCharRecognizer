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
    if getattr(sys, "frozen", False):
        return sys._MEIPASS
    else:
        return os.path.dirname(os.path.abspath(__file__))


BASE_PATH = get_base_path()

sys.path.insert(0, BASE_PATH)
sys.path.insert(0, os.path.join(BASE_PATH, "src"))

import config
from matcher import load_templates, preprocess_query, recognize


# ============================================================
# Flask 应用
# ============================================================

static_dir = os.path.join(BASE_PATH, "static")
app = Flask(__name__, static_folder=static_dir)


print("[启动] 加载模板库...")
load_templates()
print("[启动] 就绪")


# ============================================================
# 字形渲染（多字体回退 + cmap 检查）
# ============================================================

_render_fonts = {}   # (path, size) -> ImageFont
_font_cmaps = {}     # path -> set of codepoints


def _list_all_fonts():
    """按优先级返回所有字体文件路径：黑体 > 宋体 P0 > 宋体 P2 > 楷体 P0 > 楷体 P2。"""
    base = config.BASE_DIR
    priority_dirs = ["fonts_hei", "fonts_song", "fonts_kai"]
    result = []
    for d in priority_dirs:
        dir_path = os.path.join(base, "data", d)
        if not os.path.isdir(dir_path):
            continue
        for f in sorted(os.listdir(dir_path)):
            if f.lower().endswith((".ttf", ".otf")):
                result.append(os.path.join(dir_path, f))
    if not result:
        base2 = config.FONT_DIR
        if os.path.isdir(base2):
            for root, dirs, files in os.walk(base2):
                for f in files:
                    if f.lower().endswith((".ttf", ".otf")):
                        result.append(os.path.join(root, f))
    return result


def _get_cmap(path):
    """读取字体的 cmap，缓存到内存。"""
    if path not in _font_cmaps:
        try:
            from fontTools.ttLib import TTFont
            font = TTFont(path, fontNumber=0, lazy=True)
            cmap = font.getBestCmap()
            _font_cmaps[path] = set(cmap.keys())
            font.close()
        except Exception as e:
            print(f"[glyph] 读取 cmap 失败 {path}: {e}")
            _font_cmaps[path] = set()
    return _font_cmaps[path]


def _get_font(path, size):
    key = (path, size)
    if key not in _render_fonts:
        try:
            _render_fonts[key] = ImageFont.truetype(path, size - 16)
        except Exception:
            _render_fonts[key] = None
    return _render_fonts[key]

print("[启动] 预热字体 cmap...")
for p in _list_all_fonts():
    _get_cmap(p)
print("[启动] 字体 cmap 就绪")

def render_glyph_png(cp, size=128):
    """
    多字体回退：只有字体 cmap 里真正有该码点，才用它渲染。
    """
    char = chr(cp)
    for font_path in _list_all_fonts():
        # 关键：检查 cmap 里有没有这个码点
        if cp not in _get_cmap(font_path):
            continue

        font = _get_font(font_path, size)
        if font is None:
            continue

        bbox = font.getbbox(char)
        if bbox is None:
            continue
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w <= 0 or h <= 0:
            continue

        img = Image.new("L", (size, size), 255)
        draw = ImageDraw.Draw(img)
        x = (size - w) // 2 - bbox[0]
        y = (size - h) // 2 - bbox[1]
        draw.text((x, y), char, font=font, fill=0)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf
    return None


# ============================================================
# 路由
# ============================================================

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

    results = recognize(cnn_vec, hog_vec, top_k=50, min_score=0.0)
    return jsonify({"results": results})


def open_browser():
    import time
    time.sleep(1.5)
    webbrowser.open("http://127.0.0.1:5000")


if __name__ == "__main__":
    import logging
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.WARNING)

    threading.Thread(target=open_browser, daemon=True).start()

    print("[启动] 访问 http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)