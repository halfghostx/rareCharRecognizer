# config.py
# -*- coding: utf-8 -*-
"""
生僻字识别项目配置文件

用户适配说明：
- 本项目不包含任何字体文件，请自行从天珩字库官网获取。
- 下载后将 TH-Tshyn-P0/P1/P2.ttf 放入 FONT_DIR 指定的目录。
- 不需要下载 P16 文件（私用区，无标准 Unicode 码点，叶典无法查询）。
- 如需切换字体或调整码点范围，只改本文件即可。
"""

import os

# ============================================================
# 路径配置
# ============================================================

# 项目根目录（config.py 所在目录的上一级）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 字体文件存放目录
FONT_DIR = os.path.join(BASE_DIR, "data", "fonts")

# 模板矩阵与码点表输出路径
TEMPLATES_PATH = os.path.join(BASE_DIR, "data", "templates.npy")
CODEPOINTS_PATH = os.path.join(BASE_DIR, "data", "codepoints.npy")


# ============================================================
# 字形渲染配置
# ============================================================

# 模板边长（像素）。渲染后的字形位图会被 resize 到这个尺寸，
# 再展平为 TEMPLATE_SIZE * TEMPLATE_SIZE 维向量。
#
# 取值权衡：
#   64  → 4096 维，15 万字约 2.5 GB (float32)，匹配 < 1s，推荐起点
#   48  → 2304 维，15 万字约 1.4 GB，精度略降
#   96  → 9216 维，15 万字约 5.5 GB，精度略升
#   128 → 16384 维，15 万字约 9.8 GB，内存吃紧
TEMPLATE_SIZE = 64

# 渲染时字体字号（略小于画布，留出边距便于居中）
FONT_SIZE = TEMPLATE_SIZE - 8


# ============================================================
# 平面配置
# ============================================================

# 每个平面包含：文件名、码点范围（None 表示不限制，取字体文件中全部字形）
#
# 关于码点范围：
#   - P0：基本区 + 扩展 A 区，字形密集，建议范围设为 None
#   - P1：扩展 G 区，码点范围 0x30000 – 0x3134F
#   - P2：包含扩展 B/C/D/E/F 区，第一版只跑扩展 B，范围 0x20000 – 0x2A6DF
#
# 第一版建议只启用 P2 的扩展 B 区（约 4.2 万字），跑通后再逐步放开其他平面。
# 想启用某个平面，把它从 PLANES 里注释掉/取消注释即可。

PLANES = [
    # --- 第一版只开这个：扩展 B 区 ---
    {
        "file": "TH-Tshyn-P2.ttf",
        "range": (0x20000, 0x2A6DF),
        "note": "扩展 B 区，约 4.2 万字",
    },

    # --- 以下平面后续按需启用 ---

    # {
    #     "file": "TH-Tshyn-P0.ttf",
    #     "range": None,
    #     "note": "基本区 + 扩展 A 区",
    # },

    # {
    #     "file": "TH-Tshyn-P1.ttf",
    #     "range": (0x30000, 0x3134F),
    #     "note": "扩展 G 区",
    # },

    # 如果后续想放开 P2 全部范围（B/C/D/E/F 区），改用下面这条：
    # {
    #     "file": "TH-Tshyn-P2.ttf",
    #     "range": None,
    #     "note": "扩展 B/C/D/E/F 区全部",
    # },
]


# ============================================================
# 匹配配置
# ============================================================

# 返回的候选数量。形近字场景下 Top-1 不一定准，给用户候选列表更实用。
TOP_K = 5

# 相似度低于此值的候选会被过滤（0~1，TM_CCOEFF_NORMED 归一化后范围）
MIN_SCORE = 0.3


# ============================================================
# 叶典网查询
# ============================================================

# 叶典网的码点查询 URL 模板。{codepoint} 会被替换为十六进制码点（不含 U+ 前缀）。
YEDICT_URL_TEMPLATE = "http://yedict.com/zscontent.asp?uni={codepoint}"


# ============================================================
# 自检：导入本模块时打印一次配置概览，方便确认
# ============================================================

if __name__ == "__main__":
    print("=== 配置概览 ===")
    print(f"字体目录   : {FONT_DIR}")
    print(f"模板输出   : {TEMPLATES_PATH}")
    print(f"码点输出   : {CODEPOINTS_PATH}")
    print(f"模板尺寸   : {TEMPLATE_SIZE} × {TEMPLATE_SIZE} "
          f"({TEMPLATE_SIZE ** 2} 维)")
    print(f"字体字号   : {FONT_SIZE}")
    print(f"返回候选数 : {TOP_K}")
    print(f"最低相似度 : {MIN_SCORE}")
    print()
    print("启用的平面：")
    for p in PLANES:
        rng = p["range"]
        rng_str = f"0x{rng[0]:X} – 0x{rng[1]:X}" if rng else "不限"
        print(f"  {p['file']:24s} 范围: {rng_str}  ({p.get('note', '')})")
    print()
    print("字体文件存在性检查：")
    for p in PLANES:
        path = os.path.join(FONT_DIR, p["file"])
        exists = "✓" if os.path.exists(path) else "✗ 缺失"
        print(f"  [{exists}] {path}")