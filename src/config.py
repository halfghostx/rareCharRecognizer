# config.py
# -*- coding: utf-8 -*-
"""
生僻字识别项目配置文件
"""

# config.py 顶部
import os
import sys


def _get_project_root():
    """获取项目根目录（兼容打包和开发两种模式）。"""
    if getattr(sys, "frozen", False):
        # 打包后：exe 所在目录
        return os.path.dirname(sys.executable)
    else:
        # 开发时：config.py 的上一级
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


BASE_DIR = _get_project_root()

# ============================================================
# 路径配置
# ============================================================

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FONT_DIR = os.path.join(BASE_DIR, "data", "fonts")

# 粗排：DINOv2 CNN 特征
TEMPLATES_PATH = os.path.join(BASE_DIR, "data", "templates.npy")
CODEPOINTS_PATH = os.path.join(BASE_DIR, "data", "codepoints.npy")

# 精排：HOG 特征
HOG_TEMPLATES_PATH = os.path.join(BASE_DIR, "data", "hog_templates.npy")


# ============================================================
# CNN 模型配置
# ============================================================

CNN_MODEL_PATH = os.path.join(BASE_DIR, "models", "dinov2.onnx")

SAMPLE_CODEPOINTS_FILE = os.path.join(
    BASE_DIR, "data", "sample_codepoints.txt"
)


# ============================================================
# 字形渲染配置
# ============================================================

TEMPLATE_SIZE = 64
FONT_SIZE = TEMPLATE_SIZE - 8


# ============================================================
# 平面配置
# ============================================================

PLANES = [
    {
        "file": "TH-Tshyn-P0.ttf",
        "range": (0x3400, 0x4DBF),
        "note": "扩展 A 区全量",
    },
]


# ============================================================
# 匹配配置
# ============================================================

TOP_K = 20
COARSE_TOP_K = 6000
MIN_SCORE = 0.1


# ============================================================
# 叶典网查询
# ============================================================

YEDICT_URL_TEMPLATE = "http://yedict.com/zscontent.asp?uni={codepoint}"


# ============================================================
# 自检
# ============================================================

if __name__ == "__main__":
    print("=== 配置概览 ===")
    print(f"字体目录     : {FONT_DIR}")
    print(f"CNN 模型     : {CNN_MODEL_PATH}")
    print(f"CNN 模板     : {TEMPLATES_PATH}")
    print(f"HOG 模板     : {HOG_TEMPLATES_PATH}")
    print(f"码点输出     : {CODEPOINTS_PATH}")
    print(f"模板尺寸     : {TEMPLATE_SIZE} × {TEMPLATE_SIZE}")
    print(f"粗排 Top-K   : {COARSE_TOP_K}")
    print(f"最终 Top-K   : {TOP_K}")
    print()
    print("启用的平面：")
    for p in PLANES:
        rng = p["range"]
        rng_str = f"0x{rng[0]:X} – 0x{rng[1]:X}" if rng else "不限"
        print(f"  {p['file']:24s} 范围: {rng_str}  ({p.get('note', '')})")