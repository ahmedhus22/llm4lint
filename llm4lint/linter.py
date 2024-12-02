"""Classes to use LLMs to perform Linting"""
from typing import List, Dict
#from abc import ABC, abstractmethod
from argparse import ArgumentParser
from pathlib import Path
from llm import LLM

class Linter:
    def __init__(self, llm: LLM):
        self.llm = llm
        self.lints = {} # key:filename, value:lint

    def get_lints(self, files: List[Path]) -> Dict[str, str]:
        """Prompt the LLM to perform linting"""
        for file in files:
            with open(file, "r", encoding="utf-8") as f:
                code = f.read()
                self.lints[str(file)] = self.llm.inference(code, text_stream=False)
        return self.lints
    
    def init_interactive(self, files: List[Path]) -> None:
        """Start interactive shell for analyzing code"""
        raise NotImplementedError
    
def main():
    parser = ArgumentParser()
    parser.add_argument("files", nargs="+", help="file names for linting")
    args = parser.parse_args()
    print("Analyzing files: " + ", ".join(args.files) + "...")
    cli_linter = Linter(LLM(None, model_path=Path("../models/lora_model")))
    cli_linter.get_lints(args.files)
    print(cli_linter.lints)

if __name__=="__main__":
    main()