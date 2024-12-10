# LLM4LINT
A Static Analysis Tool built by finetuning Qwen2.5 Coder using unsloth.

## Features:
- Linting of Python source code in traditional linting format and an interactive mode.
- You can also provide your own data to create a different model (use train.py script):
    - Specify your own examples in a csv file with input and output columns.
    output should be `<lineno> - <type>: <issue>`. With just 1 example multiple datapoints are created using augmentation.
    - Augmentation of inputs: 
        - Variable names are replaced with different names to make sure model does not memorize the code.
        - Additional code is added before and after your input example. (outputs are also adjusted to account for lineno changes).
- Dataset created using this script: https://huggingface.co/datasets/ahmedhus22/python-static-analysis-linting

## Usage:
llm4lint [-h] [-i] filename

- positional arguments:
  filename             Python Source File to lint

- options:
  - -h, --help           show this help message and exit
  - -i, --interactive    starts chat interface for interacting with model

## Installation (For Inference Only)
```
pip install llm4lint
```
- Download the fine-tuned model from [Hugging Face](https://huggingface.co/ahmedhus22/llm4lint-7B-Qwen2.5Coder/tree/main) .gguf and Modelfile

- Create ollama Model
```
`ollama create llm4lint7b -f <Modelfile-Path>`
```
Now, You can access the linter anywhere in terminal using
```
llm4lint <filename> [options]
```
