# 首先請確保安裝了：pip install huggingface_hub
from huggingface_hub import snapshot_download

# 設定模型 ID
repo_id = "zai-org/chatglm2-6b" 

# 開始下載
local_dir = snapshot_download(
    repo_id=repo_id,
    local_dir=f"./chatglm2-6b",      # 下載到本地的路徑
    local_dir_use_symlinks=False,  # 建議設為 False，直接存實體檔案
    revision="main"                # 指定分支，通常是 main
)

print(f"模型已成功下載至: {local_dir}")