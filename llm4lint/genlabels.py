from typing import Callable
import sys
from pathlib import Path
import subprocess

def lint_dataset(linter: Callable[[Path], str], dataset_path: Path = Path("../stack-v2-smol")):
    """performs linting on the entire dataset,
    and saves the output in a directory with same filenames of source codes
    """
    for repo_owner in dataset_path.iterdir():
        project = next(repo_owner.iterdir())
        for file in project.iterdir():
            label = linter(file)
            output_file = file.with_suffix(".txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(label)

def linter_pylint(file: Path) -> str:
    """performs linting on file using pylint, returns output as json"""
    out = subprocess.run(['pylint', '--persistent=n', '--output-format=json', file],
                         capture_output=True).stdout
    return out.decode(encoding=sys.stdout.encoding)
lint_dataset(linter=linter_pylint)
