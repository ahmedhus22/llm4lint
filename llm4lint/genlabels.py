from typing import Callable, Union, List
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
        #files = project.glob("*.py")
        files = []
        for file in project.iterdir():
            files.append(file)
            with open(file, "r", encoding="utf-8") as code:
                dataset["code"].append(code.read())
        labels = linter(list(files))
        dataset["label"] += labels
        print(dataset["code"])
        print(dataset["label"])
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

def linter_pylint(file: Union[Path, List]) -> str:
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

def linter_pylint_project(files: List) -> List:
    """performs linting on all files using pylint, returns dictionary of source code issues messages"""
    raw_out = subprocess.run(['pylint', '--persistent=n', '--disable=import-error', '--output-format=json'] + files,
                         capture_output=True).stdout
    raw_out = raw_out.decode(encoding=sys.stdout.encoding)
    json_objs = json.loads(raw_out)
    messages = []
    module_messages = ""
    prev_module = ""
    for obj in json_objs:
        message = str(obj["line"]) + " " + obj["message"] + "\n"
        module_messages += message
        if obj["module"] != prev_module:
            messages.append(module_messages)
            module_messages = ""
        prev_module = obj["module"]
    messages.append(module_messages) # adding the last module_messages
    return messages

lint_dataset(linter=linter_pylint_project)
