# download_dinov2.py
from modelscope import snapshot_download

# 模型ID
model_id = "timm/vit_base_patch14_dinov2.lvd142m"

# 下载到本地指定目录
model_dir = snapshot_download(
    model_id,
    cache_dir="./models/dinov2_local"  # 下载到这个文件夹
)

print(f"模型已下载到: {model_dir}")