from fontTools.ttLib import TTFont

font_path = "data/fonts/TH-Tshyn-P0.ttf"
target_cp = 0x34CE5

try:
    font = TTFont(font_path)
    cmap = font.getBestCmap()
    if target_cp in cmap:
        print(f"✅ U+{target_cp:X} 存在于 {font_path} 中")
        print(f"   对应字形名称: {cmap[target_cp]}")
    else:
        print(f"❌ U+{target_cp:X} 不存在于 {font_path} 中")
        print(f"   该字体文件共收录 {len(cmap)} 个字符")
        # 检查扩展A区的整体收录情况
        ext_a_count = sum(1 for cp in cmap if 0x3400 <= cp <= 0x4DBF)
        print(f"   其中扩展A区(0x3400-0x4DBF)收录了 {ext_a_count} 个字符")
except Exception as e:
    print(f"读取字体时出错: {e}")

# python -c "import numpy as np; c=np.load('data/codepoints.npy'); print('4B6D 在模板库中:', 0x4B6D in c)"