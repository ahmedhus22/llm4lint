import argparse
from pathlib import Path
from llm import LLM, InstructLLM
from augmentation import augment_dataset
from datasets import load_dataset

def train(model_name, model_path: Path = None):
    qwen_base = LLM(
        model_name,
        model_path=model_path,
    )
    qwen_base.train(Path("../datasets/dataset_combined.csv"))
    qwen_base.save(Path("../models/q4_gguf"), format="q4_gguf")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--augment_data", help="specify filename to perform augmentation")
    parser.add_argument("-p", "--parameters", choices=["1.5b", "3b", "7b", "14b"], help="Qwen2.5-Coder Model Parameters", required=True)
    args = parser.parse_args()
    if not args.augment_data is None:
        augment_dataset(
            Path("../datasets/examples.csv"),
            save_path=Path("../datasets/trainer_aug.csv"),
            save_format="csv"
        )
    train("unsloth/Qwen2.5-Coder-" + str.upper(args.parameters) + "-bnb-4bit")

if __name__=="__main__":
    main()
