from pathlib import Path
from huggingface_hub import snapshot_download
from lerobot.utils.constants import HF_LEROBOT_HOME

repo_id = "nc8304/so101_combined"
local_dir = HF_LEROBOT_HOME / repo_id

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
)
print(f"Download complete: {local_dir}")
