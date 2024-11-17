from typing import Callable
import sys
from pathlib import Path
import subprocess

def lint_dataset(linter: Callable[[Path], bytes], dataset_path: Path = Path("../stack-v2-smol")):
    """performs linting on the entire dataset,
    and saves the output in a directory with same filenames of source codes
    """
    # LABEL_ROOT = "../labels"
    # Path(LABEL_ROOT).mkdir(exist_ok=True)
    for repo_owner in dataset_path.iterdir():
        project = next(repo_owner.iterdir())
        for file in project.iterdir():
            label = linter(file).decode(encoding=sys.stdout.encoding)
            output_file = file.with_suffix(".txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(label)
            break


linter = lambda file: subprocess.run(['pylint', '--persistent=n', '--output-format=json', file], capture_output=True).stdout
lint_dataset(linter)
