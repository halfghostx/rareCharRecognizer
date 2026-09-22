# test_onnx.py
import sys
import os
sys.path.insert(0, ".")
sys.path.insert(0, "src")   # 加上这行，找到 src/config.py
from cnn_feature import CNNFeatureExtractor
import config
import numpy as np

print(f"加载模型: {config.CNN_MODEL_PATH}")
extractor = CNNFeatureExtractor(config.CNN_MODEL_PATH)

# 造一张黑字白底的测试图：中间画一个简单方块
test_img = np.full((64, 64), 255, dtype=np.uint8)
test_img[20:44, 20:44] = 0

feat = extractor.extract(test_img)
if feat is None:
    print("[失败] 特征提取返回 None")
    sys.exit(1)

print(f"特征维度: {feat.shape}")
print(f"特征 L2 范数: {np.linalg.norm(feat):.4f}")
print(f"特征前 5 维: {feat[:5]}")
print("测试通过。")