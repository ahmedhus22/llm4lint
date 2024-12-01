"""abstractions for fine-tuning pre-trained LLMs"""
from pathlib import Path
from unsloth import (
    FastLanguageModel,
    is_bfloat16_supported
)
import torch
from datasets import load_dataset, Dataset
from trl import SFTTrainer
from transformers import TrainingArguments

class LLM:
    def __init__(
            self, 
            model_name: str,
            type: str = "base",
            path: Path = Path("../models/"),
            max_seq_length: int = 2048,
            HF_TOKEN: str = None
        ) -> None:
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        self.load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

        self.alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

        ### Instruction:
        {}

        ### Input:
        {}

        ### Response:
        {}"""

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            # Can select any from the below:
            # "unsloth/Qwen2.5-0.5B", "unsloth/Qwen2.5-1.5B", "unsloth/Qwen2.5-3B"
            # "unsloth/Qwen2.5-14B",  "unsloth/Qwen2.5-32B",  "unsloth/Qwen2.5-72B",
            # And also all Instruct versions and Math. Coding verisons!
            model_name = self.model_name,
            max_seq_length = self.max_seq_length,
            dtype = self.dtype,
            load_in_4bit = self.load_in_4bit,
            # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
        )

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",],
            lora_alpha = 16,
            lora_dropout = 0, # Supports any, but = 0 is optimized
            bias = "none",    # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
            random_state = 3407,
            use_rslora = False,  # We support rank stabilized LoRA
            loftq_config = None, # And LoftQ
        )

    def _formatting_prompts_func(self, examples):
        """formats batch of examples with columns 'code' and 'lable' into alpaca_promt format"""
        EOS_TOKEN = self.tokenizer.eos_token # Must add EOS_TOKEN
        instructions = ["Perform linting on the given code. Try to find all the source code issues, style issues, type errors, and fatal errors."]
        inputs       = examples["code"]
        outputs      = examples["label"]
        texts = []
        for input, output in zip(inputs, outputs):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = self.alpaca_prompt.format(instructions, input, output) + EOS_TOKEN
            texts.append(text)
        return { "text" : texts, }

    def preprocess(self, dataset_path: Path) -> Dataset:
        dataset = load_dataset("csv", data_files=dataset_path)["train"]
        dataset = dataset.map(self._formatting_prompts_func, batched=True)
        return dataset
    
    def train(self, train_dataset: Path):
        self.dataset = self.preprocess(train_dataset)
        trainer = SFTTrainer(
        model = self.model,
        tokenizer = self.tokenizer,
        train_dataset = self.dataset,
        dataset_text_field = "text",
        max_seq_length = self.max_seq_length,
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            # num_train_epochs = 1, # Set this for 1 full training run.
            max_steps = 60,
            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            report_to = "none", # Use this for WandB etc
            ),
        )

    def inference(self, input):
        raise NotImplementedError
    
    def save(self, path: Path):
        raise NotImplementedError
    
    def load(self, path: Path):
        raise NotImplementedError
