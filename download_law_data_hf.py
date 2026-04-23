# download_law_data_hf.py
from huggingface_hub import snapshot_download

dataset_dir = snapshot_download(
    repo_id="ShengbinYue/DISC-Law-SFT",
    repo_type="dataset",
    local_dir="./raw_data/DISC-Law-SFT"
)
print(f"数据集已下载到: {dataset_dir}")