# Starndard Library dependencies
import os
from pathlib import Path
from typing import Callable

# ML dependencies
from peft import LoraConfig
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig

# Internal dependencies
from src.utils.interfaces import DatasetInterface, ModelInterface
from src.utils.extra import clean_string, get_src_path


# Global Variables
MODELS: list[dict[str, str]] = [
    {"name": "meta-llama/Llama-3.1-8B"},
    {"name": "mistralai/Mistral-7B-v0.3"},
    {"name": "Qwen/Qwen2.5-7B"},
    {"name": "ibm-granite/granite-3.0-8b-base"},
]


# Functions
def locate_save_path(dir_name: str) -> str:
    src_path: str = get_src_path()
    rel_path: str = f"data/explore-models/{dir_name}"
    dir_path: str = os.path.join(src_path, rel_path)
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    return dir_path


SAVE_PATH: Callable[[str], str] = lambda dir_name: f"../data/explore-models/{dir_name}"


def run20241122A() -> None:

    for info in MODELS:

        model_name: str = info["name"]
        model_path: str = locate_save_path(dir_name=clean_string(model_name))
        eval_steps: int = 10
        save_steps: int = 45
        warmup_steps: int = 25
        max_steps: int = 1  # 100
        dataset_name: str = "GAIR/lima"

        # Train config
        training_arguments: SFTConfig = SFTConfig(
            output_dir=model_path,
            eval_strategy="steps",
            do_eval=True,
            optim="paged_adamw_8bit",
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            per_device_eval_batch_size=2,
            log_level="debug",
            logging_steps=10,
            learning_rate=1e-4,
            eval_steps=eval_steps,
            max_steps=max_steps,
            save_steps=save_steps,
            warmup_steps=warmup_steps,
            lr_scheduler_type="linear",
            report_to="tensorboard",
            logging_dir=os.path.join(model_path, "logs"),
            max_seq_length=512,
        )

        # # Initialize SFTConfig
        # sft_config: SFTConfig = SFTConfig()

        # Low-Rank Adaptation (LoRA) configuration for efficient fine-tuning
        peft_config = LoraConfig(
            lora_alpha=16,  # Scaling factor for LoRA updates
            lora_dropout=0.05,  # Dropout rate applied to LoRA layers
            r=16,  # Rank of the LoRA decomposition
            bias="none",  # No bias is added to the LoRA layers
            task_type="CAUSAL_LM",  # Specify the task as causal language modeling
            target_modules=[  # Modules to apply LoRA to
                "k_proj",
                "q_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "down_proj",
                "up_proj",
            ],
        )

        dataset_interface: DatasetInterface = DatasetInterface(
            dataset_name=dataset_name, model_name=model_name
        )
        model_interface: ModelInterface = ModelInterface()
        model_interface.load_model(name=model_name)
        model_interface.load_PEFT_config(config=peft_config)
        model_interface.load_dataset(interface=dataset_interface)

        print(f"Training: {model_name}")
        model_interface.train(
            method=SFTTrainer,
            arguments=training_arguments,  # , method_config=sft_config
        )


if __name__ == "__main__":
    run20241122A()
