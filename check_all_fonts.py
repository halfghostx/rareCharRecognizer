# check_all_fonts.py
import os
from fontTools.ttLib import TTFont

FONT_DIR = "data/fonts"
TARGETS = [0x054B2]  # 想查的码点，可以加多个

for fname in sorted(os.listdir(FONT_DIR)):
    if not fname.lower().endswith((".ttf", ".otf", ".ttc")):
        continue
    path = os.path.join(FONT_DIR, fname)
    try:
        font = TTFont(path)
        cmap = font.getBestCmap()
    except Exception as e:
        print(f"[读取失败] {fname}: {e}")
        continue

    print(f"\n{fname}: cmap 共 {len(cmap)} 个字形")
    for cp in TARGETS:
        if cp in cmap:
            print(f"  ✅ U+{cp:X} 存在，字形名: {cmap[cp]}")
        else:
            print(f"  ❌ U+{cp:X} 不存在")