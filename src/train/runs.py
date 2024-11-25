# Starndard Library dependencies
import gc
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
    "meta-llama/Llama-3.1-8B",
    "mistralai/Mistral-7B-v0.3",
    "Qwen/Qwen2.5-7B",
    "ibm-granite/granite-3.0-8b-base",
]

DATASETS: list[str] = [
    "GAIR/lima",
    "databricks/databricks-dolly-15k",
    "tatsu-lab/alpaca",
    "argilla/ifeval-like-data",
]


def run_experiment_A(id="A") -> None:

    model_name: str = MODELS[1]
    dataset_name: str = DATASETS[3]
    rel_path: Path = Path("final-model-train")
    rel_path = rel_path / id / "runs"
    model_path: str = locate_data_path(rel_path=str(rel_path))

    # Training timing control
    eval_steps: int = 100
    save_steps: int = 100
    warmup_steps: int = 25
    max_steps: int = 3000

    # Train config
    training_arguments: SFTConfig = SFTConfig(
        output_dir=model_path,
        eval_strategy="steps",
        do_eval=True,
        optim="paged_adamw_8bit",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=4,
        log_level="debug",
        logging_steps=10,
        learning_rate=3e-5,
        eval_steps=eval_steps,
        max_steps=max_steps,
        save_steps=save_steps,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        report_to="tensorboard",
        logging_dir=str(Path(model_path) / "logs"),
        max_seq_length=512,
    )

    # Low-Rank Adaptation (LoRA) configuration for efficient fine-tuning
    peft_config = LoraConfig(
        r=16,  # Rank of the LoRA decomposition
        lora_alpha=32,  # Scaling factor for LoRA updates
        lora_dropout=0.2,  # Dropout rate applied to LoRA layers
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
    gc.collect()
    model_interface: ModelInterface = ModelInterface()
    model_interface.load_model(name=model_name)
    model_interface.load_PEFT_config(config=peft_config)
    model_interface.load_dataset(interface=dataset_interface)

    print(f"\n\n\nTraining: {model_name} on {dataset_name}\n")
    model_interface.train(
        method=SFTTrainer,
        arguments=training_arguments,  # , method_config=sft_config
    )

    # Clean up
    model_interface.cleanup_model()
    dataset_interface.cleanup_dataset()
    del model_interface
    del dataset_interface
    gc.collect()


if __name__ == "__main__":
    run_experiment_A()
