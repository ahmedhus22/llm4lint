"""abstractions for fine-tuning pre-trained LLMs"""
from typing import Optional
from pathlib import Path
from unsloth import (
    FastLanguageModel,
    is_bfloat16_supported,
    to_sharegpt,
    standardize_sharegpt,
    apply_chat_template
)
from unsloth.chat_templates import get_chat_template
import torch
from datasets import load_dataset, Dataset
from trl import SFTTrainer
from transformers import (
    TrainingArguments,
    TextStreamer
)

class LLM:
    def __init__(
            self, 
            model_name: str,
            model_path: Optional[Path] = Path("../models/lora_model"),
            max_seq_length: int = 2048,
            HF_TOKEN: str = None
        ) -> None:
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        self.load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
        self.model_path = model_path
        self.HF_TOKEN = HF_TOKEN
        self.alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

        if (not self.model_path is None) and self.model_path.exists():
            self.model, self.tokenizer = self.load(self.model_path)
        else:
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
        
        self.dataset = None # load it only if train method is called

    def _formatting_prompts_func(self, examples):
        """formats batch of examples with columns 'input' and 'output' into alpaca_promt format"""
        EOS_TOKEN = self.tokenizer.eos_token # Must add EOS_TOKEN
        instructions = "Perform linting on the given code. Specify output in format: <line_number> - <type>: <issue>."
        inputs       = examples["input"]
        outputs      = examples["output"]
        texts = []
        for input, output in zip(inputs, outputs):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = self.alpaca_prompt.format(instructions, input, output) + EOS_TOKEN
            texts.append(text)
        return { "text" : texts, }

    def preprocess(self, dataset_path: Path) -> Dataset:
        dataset = load_dataset("csv", data_files=str(dataset_path))["train"]
        dataset = to_sharegpt(
            dataset,
            merged_prompt = "Perform linting on the given code. Specify output in format: <line_number> - <type>: <issue>.\n [[\nYour input is:\n{input}]]",
            output_column_name = "output",
            conversation_extension = 1, # Select more to handle longer conversations
        )
        dataset = standardize_sharegpt(dataset)
        chat_template = """Below are some instructions that describe some tasks. Write responses that appropriately complete each request.

### Instruction:
Perform linting on the given code. Specify output in format: <line_number> - <type>: <issue>.
{INPUT}

### Response:
{OUTPUT}"""
        dataset = apply_chat_template(
            dataset,
            tokenizer = self.tokenizer,
            chat_template = chat_template,
            # default_system_message = "You are a helpful assistant", << [OPTIONAL]
        )
        return dataset
    
    def train(
            self, 
            train_dataset: Path,
            save_path: Path = Path("../models/lora_model")
        ):
        """fine-tune model 'self.model_name' or (continue training) loaded model with given dataset, 
        saves model and returns training stats"""
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
            num_train_epochs = 1, # Set this for 1 full training run.
            #max_steps = 60,
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
        trainer_stats = trainer.train()
        self.save(save_path)
        return trainer_stats

    def inference(self, input: str, text_stream:bool = True):
        FastLanguageModel.for_inference(self.model) # Enable native 2x faster inference
        inputs = self.tokenizer(
        [
            self.alpaca_prompt.format(
                "Perform linting on the given code. Try to find all the source code issues, style issues, type errors, and fatal errors.", # instruction
                input, # input
                "", # output - leave this blank for generation!
            )
        ], return_tensors = "pt").to("cuda")

        if text_stream:
            text_streamer = TextStreamer(self.tokenizer)
            _ = self.model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128) # TODO: THIS IS WRONG, TextStreamer STREAMS DIRECTLY TO STDOUT!!
        else:
            outputs = self.model.generate(**inputs, max_new_tokens = 64, use_cache = True)
            return self.tokenizer.batch_decode(outputs)[0]
    
    def save(self, path: Path, format: str = "lora") -> None:
        """saves the model in formats: 'lora', '16bit_merged' 'q4_gguf' or '16bit_gguf'"""
        if format == "lora":
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
        elif format == "q4_gguf":
            self.model.save_pretrained_gguf(path, self.tokenizer, quantization_method = "q4_k_m")
            # if not self.HF_TOKEN is None:
            #     self.model.push_to_hub_gguf("hf/model", self.tokenizer, quantization_method = "q4_k_m", token = self.HF_TOKEN)
        elif format == "16bit_merged":
            # gguf DOES NOT WORK use 16bit_merged then manually convert to gguf
            # based on wiki: https://github.com/unslothai/unsloth/wiki#manually-saving-to-gguf (use cmake instead)
            self.model.save_pretrained_merged(path, self.tokenizer, save_method = "merged_16bit")
        elif format == "16bit_gguf":
            self.model.save_pretrained_gguf(path, self.tokenizer, quantization_method = "f16")
    
    def load(self, path: Path):
        self.model_path = path
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = str(self.model_path),
            max_seq_length = self.max_seq_length,
            dtype = self.dtype,
            load_in_4bit = self.load_in_4bit,
        )
        #FastLanguageModel.for_inference(model) # Enable native 2x faster inference
        return model, tokenizer


class InstructLLM(LLM):
    def __init__(self, 
        model_name: str,
        model_path: Path | None = Path("../models/lora_model"),
        max_seq_length: int = 2048,
        HF_TOKEN: str = None
    ) -> None:
        super().__init__(model_name, model_path, max_seq_length, HF_TOKEN)
        if (self.model_path is None) or not self.model_path.exists():
            self.tokenizer = get_chat_template(
                self.tokenizer,
                chat_template = "llama-3", # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
                mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
            )

    def _formatting_prompts_chat(self, examples):
        convos = examples["conversations"]
        texts = [self.tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
        return { "text" : texts, }

    def preprocess(self, dataset_path: Path) -> Dataset:
        raise NotImplementedError
    
    def inference(self, input: str, text_stream: bool = True) -> str:
        FastLanguageModel.for_inference(self.model) # Enable native 2x faster inference
        messages = [
            {"from": "human", "value": input},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize = True,
            add_generation_prompt = True, # Must add for generation
            return_tensors = "pt",
        ).to("cuda")
        outputs = self.model.generate(input_ids = inputs, max_new_tokens = 64, use_cache = True)
        return self.tokenizer.batch_decode(outputs)[0]
