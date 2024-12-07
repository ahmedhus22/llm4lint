import argparse
from typing import Iterator
from pathlib import Path
from ollama import chat, ChatResponse

class App:
    def __init__(self, model: str) -> None:
        self.model:str = model
        self.lint_prompt = "Perform linting on the given code. Specify output in format: <line_number> - <type>: <issue>"

    def get_lints(self, file: Path) -> Iterator[ChatResponse]:
        """returns predicted tokens as a stream(iterable),
        chunk["message"]["content"]"""
        with open(file, "r", encoding="utf-8") as src_file:
            user_code = src_file.read()
        stream = chat(
            model=self.model,
            messages=[{'role': 'user', 'content': self.lint_prompt + user_code}],
            stream=True,
        )
        return stream

    def init_shell(
            self,
            model: str,
            file: Path
        ) -> None:
        """starts chat interface for interacting with model"""
        raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(
        description="An LLM Linter that you can finetune with your own data"
    )
    parser.add_argument("filename", help="Python Source File to lint")
    parser.add_argument("--examples", default=None, help="csv file with linting examples: cols=input, output")
    parser.add_argument("--interactive", action="store_true", help=App.init_shell.__doc__)
    args = parser.parse_args()
    # cli_app = App()

if __name__=="__main__":
    main()

# for chunk in stream:
#   print(chunk['message']['content'], end='', flush=True)