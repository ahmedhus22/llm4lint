from typing import Callable
import sys
from pathlib import Path
import subprocess
import pandas as pd

def lint_dataset(linter: Callable[[Path], str], dataset_path: Path = Path("../stack-v2-smol"), save_path: Path = Path("../datasets/dataset_pylint.csv")) -> pd.DataFrame:
    """performs linting on the entire dataset,
    and saves the output in a directory with same filenames of source codes
    """
    dataset = {"code":[], "label":[]}
    for repo_owner in dataset_path.iterdir():
        project = next(repo_owner.iterdir())
        for file in project.iterdir():
            label = linter(file)
            with open(file, "r", encoding="utf-8") as code:
                dataset["code"].append(code.read())
            dataset["label"].append(label)
            output_file = file.with_suffix(".json")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(label)
    df = pd.DataFrame(dataset)
    save_path.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(save_path, index=False, encoding="utf-8")
    return df

def linter_pylint(file: Path) -> str:
    """performs linting on file using pylint, returns output as json"""
    out = subprocess.run(['pylint', '--persistent=n', '--output-format=json', file],
                         capture_output=True).stdout
    return out.decode(encoding=sys.stdout.encoding)

lint_dataset(linter=linter_pylint)