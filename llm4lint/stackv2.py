import os
from pathlib import Path
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from smart_open import open
from datasets import load_dataset
import datasets.config


s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
download_path = ".."
datasets.config.DOWNLOADED_DATASETS_PATH = Path(download_path)

def download_contents(files):
    for file in files:
        s3_url = f"s3://softwareheritage/content/{file['blob_id']}"
        with open(s3_url, "rb", compression=".gz", transport_params={"client": s3}) as fin:
            file["content"] = fin.read().decode(file["src_encoding"])
    
    return {"files": files}

ds = load_dataset("bigcode/the-stack-v2-train-smol-ids", split="train", streaming=True, token=os.environ["HF_TOKEN"], cache_dir=download_path)
ds = ds.map(lambda row: download_contents(row["files"]))
download_root = "../stack-v2-smol"
Path(download_root).mkdir(exist_ok=True)
for row in ds:
    for file in row["files"]:
        blob_id = file["blob_id"]
        content = file["content"]
        data_path = Path(download_root, blob_id)
        with open(data_path, "w", encoding="utf-8") as f:
            f.write(content)
    break
