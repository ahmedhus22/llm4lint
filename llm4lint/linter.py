"""classes to use LLMs to perform Linting,
THIS MODULE ONLY WORKS ON LORA ADAPTERS, THIS IS FOR DEBUGGING ONLY.
GGUF FORMAT IS USED FOR DEPLOYMENT"""
from typing import List, Dict
#from abc import ABC, abstractmethod
from argparse import ArgumentParser
from pathlib import Path
from llm import LLM, InstructLLM

class Linter:
    """Base Linter class handles LLM prompting and preprocess' src code files"""
    def __init__(self, llm: LLM):
        self.llm = llm
        self.lints = {} # key:filename, value:lint
        self.instructllm = None
        self.code = None

    def get_lints(self, files: List[Path]) -> Dict[str, str]:
        """Prompt the LLM(Base Model) to perform linting"""
        for file in files:
            with open(file, "r", encoding="utf-8") as src_file:
                code = ""
                for index, line in enumerate(src_file):
                    code += str(index + 1) + "   " + line
            self.lints[str(file)] = self.llm.inference(code, text_stream=False)
        return self.lints

    def init_shell(
            self,
            model_name: str,
            file: Path,
        ) -> None:
        """Start interactive shell for analyzing code"""
        if self.instructllm is None:
            self.instructllm = InstructLLM(model_name, model_path=None)
            with open(file, "r", encoding="utf-8") as src_file:
                self.code = src_file.read()
            #prompt = "Answer questions about this python program:\n" + code
        #prompt = "Analyze this Python program for issues (lint this program): "
        while True:
            user_msg = input(">>> ")
            if user_msg == "q" or user_msg == "exit":
                break
            prompt = "Answer the question about this python program:\n" "QUESTION: " + user_msg + "\n"
            prompt += self.code
            print(self.instructllm.inference(prompt, text_stream=False))

def main():
    parser = ArgumentParser()
    parser.add_argument("file", help="file name for linting")
    parser.add_argument("--interactive", action="store_true", help="start interactive shell enter 'q' or 'exit' to exit")
    parser.add_argument("--parameters", action="store_const", const="0.5B", help="-i <no_of_model_parameters> default=0.5B")
    args = parser.parse_args()
    # no_of_params = str.upper(args.parameters)
    # print(no_of_params)
    if args.interactive:
        linter = Linter(llm=None)
        linter.init_shell("unsloth/Qwen2.5-0.5B", args.file)
    else:
        print("Analyzing files: " + args.file)
        cli_linter = Linter(LLM(None, model_path=Path("../models/lora_model")))
        cli_linter.get_lints([args.file])
        print(cli_linter.lints[args.file])

if __name__=="__main__":
    main()
