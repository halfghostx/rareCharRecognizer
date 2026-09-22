# cnn_feature.py
# -*- coding: utf-8 -*-
"""
DINOv2 CNN 特征提取器（timm 版本）。
输入: 黑字白底的灰度图 (numpy uint8)
输出: 归一化的特征向量
"""

import numpy as np
import cv2
import onnxruntime as ort


class CNNFeatureExtractor:
    def __init__(self, onnx_path):
        self.session = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        inp_shape = self.session.get_inputs()[0].shape
        out_shape = self.session.get_outputs()[0].shape
        print(f"[CNN] 模型: {onnx_path}")
        print(f"[CNN] 输入: {self.input_name} {inp_shape}")
        print(f"[CNN] 输出: {self.output_name} {out_shape}")

    def extract(self, gray_img):
        """
        gray_img: 单通道灰度图（黑字白底）
        返回: L2 归一化的特征向量 (float32)，失败返回 None
        """
        if gray_img is None or gray_img.size == 0:
            return None

        # 单通道 -> 三通道
        if len(gray_img.shape) == 2:
            img_rgb = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = gray_img

        # 缩放到 224×224
        img_resized = cv2.resize(
            img_rgb, (224, 224), interpolation=cv2.INTER_AREA
        )

        # [0,1] 归一化
        img_float = img_resized.astype(np.float32) / 255.0

        # ImageNet 归一化（DINOv2 使用标准 ImageNet 统计量）
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_norm = (img_float - mean) / std

        # HWC -> CHW -> NCHW
        img_tensor = np.transpose(img_norm, (2, 0, 1))[np.newaxis, ...]
        img_tensor = img_tensor.astype(np.float32)

        # 推理
        try:
            outputs = self.session.run(
                [self.output_name],
                {self.input_name: img_tensor},
            )[0]
        except Exception as e:
            print(f"[CNN] 推理失败: {e}")
            return None

        # timm 版 DINOv2 输出: (1, 384) 或 (1, 768)
        feature = outputs.flatten()

        # L2 归一化
        norm = np.linalg.norm(feature)
        if norm < 1e-6:
            return None
        return (feature / norm).astype(np.float32)