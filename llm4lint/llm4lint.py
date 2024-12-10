import argparse
from typing import Iterator, List, Dict
from pathlib import Path
from ollama import chat, ChatResponse


class App:
    def __init__(self, model: str) -> None:
        self.model:str = model
        self.lint_prompt = "Perform linting on the given code. Specify output in format: <line_number> - <type>: <issue>\n"

    def _addcodelines(self, code: str) -> str:
        code_with_lnos = ""
        code_lines = code.split("\n")
        for index, line in enumerate(code_lines):
            code_with_lnos += str(index+1) + "   " + line + "\n"
        return code_with_lnos

    def _getcode(self, path: Path) -> str:
        with open(path, "r", encoding="utf-8") as f:
            code: str = f.read()
        code = self._addcodelines(code)
        return code

    def get_lints(self, file: Path) -> Iterator[ChatResponse]:
        """returns predicted tokens as a stream(iterable),
        chunk["message"]["content"]"""
        user_code = self._getcode(file)
        print(user_code + "\n" + "Analyzing...")
        stream = chat(
            model=self.model,
            messages=[{'role': 'user', 'content': self.lint_prompt + user_code}],
            stream=True,
        )
        return stream

    def init_shell(
            self,
            #model: str, # if different model needs to be selected
            file: Path
        ) -> None:
        """starts chat interface for interacting with model"""
        user_code = self._getcode(file)
        print(user_code)
        print("Enter 'q' or 'exit' to exit.")
        prompt = "Answer questions regarding the python code:\n" + user_code
        messages: List[Dict[str, str]] = [{'role': 'user', 'content': prompt}]
        while True:
            print()
            user_prompt = input(">>> ")
            if user_prompt == "q" or user_prompt == "exit":
                return
            messages.append({'role': 'user', 'content': user_prompt})
            stream = chat(
                model=self.model,
                messages=messages,
                stream=True,
            )
            assistant_message: str = ""
            for chunk in stream:
                assistant_message += chunk['message']['content']
                print(chunk['message']['content'], end='', flush=True)
            
            messages.append({'role': 'assistant', 'content': assistant_message})


def main():
    parser = argparse.ArgumentParser(
        description="A Local LLM Linter that you can finetune with your own data"
        "(visit https://github.com/ahmedhus22/llm4lint to get train script)"
    )
    parser.add_argument("filename", help="Python Source File to lint")
    #REMOVE THIS FEATURE. Use train.py script for augmentation
    #parser.add_argument("--examples", default=None, help="csv file with linting examples: cols=input, output")
    parser.add_argument("-i", "--interactive", action="store_true", help=App.init_shell.__doc__)
    args = parser.parse_args()
    cli_app = App("llm4lint7b")
    if args.interactive:
        cli_app.init_shell(args.filename)
    else:
        stream = cli_app.get_lints(args.filename)
        for chunk in stream:
            print(chunk['message']['content'], end='', flush=True)
        print()

if __name__=="__main__":
    main()
