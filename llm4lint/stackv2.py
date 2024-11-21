import os
from pathlib import Path
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from smart_open import open
from datasets import load_dataset
import datasets.config


s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
DOWNLOAD_PATH = ".."
datasets.config.DOWNLOADED_DATASETS_PATH = Path(DOWNLOAD_PATH)

def download_contents(files):
    for file in files:
        s3_url = f"s3://softwareheritage/content/{file['blob_id']}"
        with open(s3_url, "rb", compression=".gz", transport_params={"client": s3}) as fin:
            file["content"] = fin.read().decode(file["src_encoding"])
    return {"files": files}

ds = load_dataset("bigcode/the-stack-v2-train-smol-ids", split="train", streaming=True, token=os.environ["HF_TOKEN"], cache_dir=DOWNLOAD_PATH)
ds = ds.map(lambda row: download_contents(row["files"]))

DOWNLOAD_ROOT = "../stack-v2-smol"
DATASET_SIZE = int(1e+3)
no_python_files = 0
for row in ds:
    repo_name = row["repo_name"]
    for file in row["files"]:
        blob_id = file["blob_id"]
        content = file["content"]
        if file["language"] == "Python":
            Path(DOWNLOAD_ROOT, repo_name).mkdir(exist_ok=True, parents=True)
            data_path = Path(DOWNLOAD_ROOT, repo_name, blob_id + ".py")
            with open(data_path, "w", encoding="utf-8") as f:
                f.write(content)
            no_python_files += 1
    if no_python_files == DATASET_SIZE:
        break
