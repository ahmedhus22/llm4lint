import os
from pathlib import Path
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from smart_open import open
from datasets import load_dataset
import datasets.config
from tqdm import tqdm


s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
DOWNLOAD_PATH = ".."
datasets.config.DOWNLOADED_DATASETS_PATH = Path(DOWNLOAD_PATH)

def download_contents(files, repo_name):
    # UNCOMMENT TO SKIP ENTIRE REPO
    # if Path(DOWNLOAD_ROOT, repo_name).exists():
    #     return {"files": None}
    for file in files:
        if Path(DOWNLOAD_ROOT, repo_name, file['blob_id'] + ".py").exists():
            continue
        if file["language"] != "Python":
            continue
        s3_url = f"s3://softwareheritage/content/{file['blob_id']}"
        with open(s3_url, "rb", compression=".gz", transport_params={"client": s3}) as fin:
            file_content: str = fin.read().decode(file["src_encoding"])
        no_of_lines = file_content.count("\n")
        if no_of_lines < 200:
            file["content"] = file_content
    return {"files": files}

ds = load_dataset("bigcode/the-stack-v2-train-smol-ids", split="train", streaming=True, token=os.environ["HF_TOKEN"], cache_dir=DOWNLOAD_PATH)
ds = ds.map(lambda row: download_contents(row["files"], row["repo_name"]))

DOWNLOAD_ROOT = "../stack-v2-smol"
DATASET_SIZE = int(4e+2)
no_python_files = 0
with tqdm(total=DATASET_SIZE) as pbar:
    for row in ds:
        repo_name = row["repo_name"]
        # UNCOMMENT TO SKIP ENTIRE REPO
        # if Path(DOWNLOAD_ROOT, repo_name).exists():
        #     continue
        for file in row["files"]:
            if Path(DOWNLOAD_ROOT, repo_name, file["blob_id"] + ".py").exists():
                continue
            try:
                content = file["content"]
            except KeyError:
                continue
            if file["language"] == "Python":
                blob_id = file["blob_id"]
                content = file["content"]
                Path(DOWNLOAD_ROOT, repo_name).mkdir(exist_ok=True, parents=True)
                data_path = Path(DOWNLOAD_ROOT, repo_name, blob_id + ".py")
                with open(data_path, "w", encoding="utf-8") as f:
                    f.write(content)
                no_python_files += 1
                pbar.update(1)
        if no_python_files >= DATASET_SIZE:
            break
