"""randomdata module provides functions to get random data for data augmentation,
    like random variable names"""
from typing import Set
import ast
import argparse
from pathlib import Path
import augmentation

def get_names(file: Path) -> Set[str]:
    """finds all name ids of code in given file"""
    with open(file, "r", encoding="utf-8") as f:
        code = f.read()
    tree = ast.parse(code)
    name_finder = augmentation.FindNames()
    name_finder.visit(tree)
    return list(name_finder.node_ids)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="file name to get data from")
    args = parser.parse_args()
    names = get_names(Path(args.file))
    print(names)

if __name__=="__main__":
    main()
