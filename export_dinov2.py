# export_dinov2.py
# -*- coding: utf-8 -*-
"""
从本地权重加载 DINOv2 并导出为 ONNX。
不读 config.json，直接用 timm 内置配置 + 本地权重覆盖。
"""

import os
import torch
import timm

os.makedirs("models", exist_ok=True)

MODEL_NAME = "vit_base_patch14_dinov2.lvd142m"
LOCAL_DIR = "models/dinov2_local"
OUTPUT_PATH = "models/dinov2_base.onnx"


def find_weights(base_dir):
    """递归查找 model.safetensors"""
    for root, dirs, files in os.walk(base_dir):
        if "model.safetensors" in files:
            return os.path.join(root, "model.safetensors")
    return None


weights_path = find_weights(LOCAL_DIR)
if not weights_path:
    print(f"[错误] 在 {LOCAL_DIR} 中未找到 model.safetensors")
    raise SystemExit(1)

print(f"权重文件: {weights_path}")
print(f"构建模型: {MODEL_NAME}")

model = timm.create_model(
    MODEL_NAME,
    pretrained=True,
    num_classes=0,
    dynamic_img_size=True,
    pretrained_cfg_overlay={"file": weights_path},
)
model.eval()

# 打印输出维度
with torch.no_grad():
    dummy = torch.randn(1, 3, 224, 224)
    out = model(dummy)
    print(f"输出 shape: {out.shape}")

print(f"导出到: {OUTPUT_PATH}")
torch.onnx.export(
    model,
    dummy,
    OUTPUT_PATH,
    input_names=["pixel_values"],
    output_names=["features"],
    opset_version=18,
    do_constant_folding=True,
)

print("导出完成。文件大小:",
      f"{os.path.getsize(OUTPUT_PATH) / 1024 / 1024:.1f} MB")

print("导出完成。文件大小:",
      f"{os.path.getsize(OUTPUT_PATH) / 1024 / 1024:.1f} MB")