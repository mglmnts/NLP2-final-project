# Starndard Library dependencies
import os
from pathlib import Path
from typing import Callable

# ML dependencies
import torch
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# Internal dependencies
from src.utils.interfaces import DatasetInterface, ModelInterface
from src.utils.extra import clean_string, locate_data_path


# Global Variables
device: str = "cuda" if torch.cuda.is_available() else "cpu"
MODELS: list[dict[str, str]] = [
    {"name": "meta-llama/Llama-3.1-8B"},
    {"name": "mistralai/Mistral-7B-v0.3"},
    {"name": "Qwen/Qwen2.5-7B"},
    {"name": "ibm-granite/granite-3.0-8b-base"},
]


SAVE_PATH: Callable[[str], str] = lambda dir_name: f"../data/explore-models/{dir_name}"


def run20241122A() -> None:

    for info in MODELS:

        model_name: str = info["name"]
        model_path: str = locate_data_path(dir_name=clean_string(model_name))
        eval_steps: int = 10
        save_steps: int = 45
        warmup_steps: int = 25
        max_steps: int = 100
        dataset_name: str = "GAIR/lima"

        # Train config
        training_arguments: SFTConfig = SFTConfig(
            output_dir=model_path,
            eval_strategy="steps",
            do_eval=False,
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
        model_interface.cleanup_model()
        dataset_interface.cleanup_dataset()
        torch.cuda.empty_cache()
        # input()


if __name__ == "__main__":
    run20241122A()
