# download_law_data.py
from modelscope import snapshot_download

dataset_dir = snapshot_download(
    'AI-ModelScope/DISC-Law-SFT',
    cache_dir='./raw_data',
    revision='master'
)
print(f"数据集已下载到: {dataset_dir}")