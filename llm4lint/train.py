from pathlib import Path
from llm import LLM, InstructLLM
from augmentation import augment_dataset
from datasets import load_dataset

def train(model_name, model_path: Path = None):
    qwen_base = LLM(
        model_name,
        model_path=model_path
    )
    qwen_base.train(Path("../datasets/examples.csv"))
    qwen_base.save(Path("../models/q4_gguf"), format="q4_gguf")

def main():
    AUGMENT_DATA = True
    if AUGMENT_DATA:
        augment_dataset(
            Path("../datasets/examples.csv"),
            save_path=Path("../datasets/examples_aug.csv"),
            save_format="csv"
        )
    train("unsloth/Qwen2.5-Coder-1.5B-bnb-4bit")

if __name__=="__main__":
    main()
