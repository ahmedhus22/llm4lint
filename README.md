# LLM4LINT
A Static Analysis Tool built by finetuning Qwen2.5 Coder using unsloth.

## Features:
- Linting of Python source code in traditional linting format and an interactive mode.
- You can also provide your own data to create a different model (use train.py script):
    - Specify your own examples in a csv file with input and output columns.
    output should be `\<lineno\> - \<type\>: \<issue\>`. With just 1 example multiple datapoints are created using augmentation.
    - Augmentation of inputs: 
        - Variable names are replaced with different names to make sure model does not memorize the code.
        - Additional code is added before and after your input example. (outputs are also adjusted to account for lineno changes).
- Dataset created using this script: [https://huggingface.co/datasets/ahmedhus22/python-static-analysis-linting]()

## Usage:
llm4lint.py [-h] [-i] filename

- positional arguments:
  filename             Python Source File to lint

- options:
  - -h, --help           show this help message and exit
  - -i, --interactive    starts chat interface for interacting with model

## Installation
- Install ollama python library, In a virtual environment: [https://github.com/ollama/ollama-python/tree/main](Ollama-Python).
- Clone this repo.
- Download the fine-tuned model from huggingface into `models/q4_gguf/`directory.
- Change first line of Modelfile to `FROM <model_download_path>`
- run `ollama create llm4lint7b -f ./models/q4_gguf/Modelfile`
- Add your virtual environment path to top of llm4lint.py file with #! line.


## Future Update: Docker Installation... or Maybe install script...
Check Deploy branch
