from paddleocr import PaddleOCR
import cv2
import numpy as np

# 初始化 OCR，指定使用服务端识别模型（识别能力更强）
# 注意：lang='ch' 表示中文，模型会自动下载
ocr = PaddleOCR(
    use_angle_cls=True,
    lang='ch',
    rec_model_dir='models/ppocr_rec', # 可选，指定模型保存目录
    show_log=True
)

# 读取你的截图
img_path = 'data/samples/Snipaste_2026-09-22_22-08-05.png'
img = cv2.imread(img_path)

# 执行识别
result = ocr.ocr(img, cls=True)

# 打印结果
for line in result:
    for word_info in line:
        text = word_info[1][0]
        confidence = word_info[1][1]
        print(f"识别结果: {text}, 置信度: {confidence:.4f}")
        # 打印每个字符的 Unicode 码点
        for char in text:
            print(f"  字符: {char}, 码点: U+{ord(char):04X}")