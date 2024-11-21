from typing import Callable
import sys
from pathlib import Path
import subprocess
import json
import pandas as pd

def lint_dataset(
        linter: Callable[[Path], str],
        dataset_path: Path = Path("../stack-v2-smol"),
        save_path: Path = Path("../datasets"),
        save_raw: bool = False
    ) -> pd.DataFrame:
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
            # save json output if raw linter function is used.
            # This part is unnecessary though, no longer used
            if linter.__name__ == "linter_pylint_raw":
                output_file = file.with_suffix(".json")
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(label)
    df = pd.DataFrame(dataset)
    save_path.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(save_path / Path("dataset_pylint.csv"), index=False, encoding="utf-8")
    if save_raw:
        df["code"].to_csv(save_path / Path("code_raw.csv"), index=False, header=False, encoding="utf-8")
        df["label"].to_csv(save_path / Path("label_raw.csv"), index=False, header=False, encoding="utf-8")
    return df

def linter_pylint_raw(file: Path) -> str:
    """performs linting on file using pylint, returns output as json"""
    out = subprocess.run(['pylint', '--persistent=n', '--disable=import-error', '--output-format=json', file],
                         capture_output=True).stdout
    return out.decode(encoding=sys.stdout.encoding)

def linter_pylint(file: Path) -> str:
    """performs linting on file using pylint, returns source code issues messages"""
    raw_out = subprocess.run(['pylint', '--persistent=n', '--disable=import-error', '--output-format=json', file],
                         capture_output=True).stdout
    raw_out = raw_out.decode(encoding=sys.stdout.encoding)
    json_objs = json.loads(raw_out)
    messages = ""
    for obj in json_objs:
        message = str(obj["line"]) + " " + obj["message"] + "\n"
        messages += message
    return messages

lint_dataset(linter=linter_pylint)
