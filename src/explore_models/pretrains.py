# Starndard Library dependencies
from typing import Callable

# ML dependencies
from peft import LoraConfig
from transformers import TrainingArguments
from trl import SFTTrainer

# Internal dependencies
from src.utils.interfaces import DatasetInterface, ModelInterface
from src.utils.extra import clean_string


# Global Variables
MODELS: dict = {
    {"name": "meta-llama/Llama-3.1-8B"},
    {"name": "mistralai/Mistral-7B-v0.3"},
    {"name": "Qwen/Qwen2.5-7B"},
    {"name": "ibm-granite/granite-3.0-8b-base"},
}

# Functions
SAVE_PATH: Callable[[str], str] = lambda dir_name: f"../data/explore-models/{dir_name}"


def benchmark20241122A() -> None:

    for info in MODELS:

        model_name: str = info["name"]
        model_path: str = SAVE_PATH(dir_name=clean_string(model_name))
        eval_steps: int = 10
        save_steps: int = 20
        dataset_name: str = "GAIR/lima"

        # Train config
        training_arguments: TrainingArguments = TrainingArguments(
            output_dir=model_path,  # Director y for checkpoints and logs
            eval_strategy="steps",  # Evaluation strategy: evaluate every few steps
            do_eval=True,  # Enable evaluation during training
            optim="paged_adamw_8bit",  # Use 8-bit AdamW optimizer for memory efficiency
            per_device_train_batch_size=4,  # Batch size per device during training
            gradient_accumulation_steps=2,  # Accumulate gradients over multiple steps
            per_device_eval_batch_size=2,  # Batch size per device during evaluation
            log_level="debug",  # Set logging level to debug for detailed logs
            logging_steps=10,  # Log metrics every 10 steps
            learning_rate=1e-4,  # Initial learning rate
            eval_steps=eval_steps,  # Evaluate the model every 25 steps
            max_steps=100,  # Total number of training steps
            save_steps=save_steps,  # Save checkpoints every 25 steps
            warmup_steps=25,  # Number of warmup steps for learning rate scheduler
            lr_scheduler_type="linear",  # Use a linear learning rate scheduler
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
        model_interface.train(method=SFTTrainer, arguments=training_arguments)


if __name__ == "__main__":
    benchmark20241122A()
