# Standard Library dependencies
import os

# from pathlib import Path

# ML dependencies
import torch
from peft import (
    LoraConfig,
    LoHaConfig,
    LoKrConfig,
    AdaLoraConfig,
    XLoraConfig,
    IA3Config,
)
from trl import SFTTrainer, SFTConfig

# Internal dependencies
from src.utils.interfaces import DatasetInterface, ModelInterface
from src.utils.extra import clean_string, locate_data_path

# Global Variables
device: str = "cuda" if torch.cuda.is_available() else "cpu"

# # Define PEFT methods with their corresponding configuration classes and parameters
# PEFT_METHODS: list[dict] = [
#     {
#         "name": "LoRA",
#         "config_class": LoraConfig,
#         "params": {
#             "r": 16,
#             "lora_alpha": 16,
#             "lora_dropout": 0.05,
#             "bias": "none",
#             "task_type": "CAUSAL_LM",
#             "target_modules": [
#                 "k_proj",
#                 "q_proj",
#                 "v_proj",
#                 "o_proj",
#                 "gate_proj",
#                 "down_proj",
#                 "up_proj",
#             ],
#         },
#     },
#     {
#         "name": "LoHa",
#         "config_class": LoHaConfig,
#         "params": {
#             "r": 16,
#             "alpha": 16,
#             "rank_dropout": 0.05,
#             "target_modules": [
#                 "k_proj",
#                 "q_proj",
#                 "v_proj",
#                 "o_proj",
#                 "gate_proj",
#                 "down_proj",
#                 "up_proj",
#             ],
#         },
#     },
#     {
#         "name": "LoKr",
#         "config_class": LoKrConfig,
#         "params": {
#             "r": 16,
#             "alpha": 16,
#             "rank_dropout": 0.05,
#             "target_modules": [
#                 "k_proj",
#                 "q_proj",
#                 "v_proj",
#                 "o_proj",
#                 "gate_proj",
#                 "down_proj",
#                 "up_proj",
#             ],
#         },
#     },
#     {
#         "name": "AdaLoRA",
#         "config_class": AdaLoraConfig,
#         "params": {
#             "init_r": 12,
#             "r": 16,
#             "beta1": 0.85,
#             "beta2": 0.85,
#             "tinit": 200,
#             "tfinal": 1000,
#             "deltaT": 10,
#         },
#     },
#     {
#         "name": "X-LoRA",
#         "config_class": XLoraConfig,
#         "params": {
#             "r": 16,
#             "xlora_dropout": 0.05,
#         },
#     },
#     {
#         "name": "IA3",
#         "config_class": IA3Config,
#         "params": {
#             "target_modules": [
#                 "k_proj",
#                 "q_proj",
#                 "v_proj",
#                 "o_proj",
#                 "gate_proj",
#                 "down_proj",
#                 "up_proj",
#             ],
#         },
#     },
# ]

# Define PEFT methods with their corresponding configuration classes and parameters
PEFT_METHODS: list[dict] = [
    {
        "name": "LoRA",
        "config_class": LoraConfig,
        "params": {
            "r": 8,  # Reduced for lower computational cost, maintaining good adaptation capability
            "lora_alpha": 32,  # Increased to improve training stability
            "lora_dropout": 0.1,  # Dropout slightly increased to prevent overfitting
            "bias": "lora_only",  # Only the bias of the LoRA layers is updated
            "task_type": "CAUSAL_LM",
            "target_modules": [
                "k_proj",
                "q_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "down_proj",
                "up_proj",
            ],
        },
    },
    {
        "name": "LoHa",
        "config_class": LoHaConfig,
        "params": {
            "r": 8,  # Reduced for better efficiency
            "alpha": 32,  # Increased to provide better adjustment capability
            "rank_dropout": 0.1,  # Dropout increased to reduce the risk of overfitting
            "module_dropout": 0.05,  # Added to prevent all LoHa layers from always being active
            "target_modules": [
                "k_proj",
                "q_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "down_proj",
                "up_proj",
            ],
        },
    },
    {
        "name": "LoKr",
        "config_class": LoKrConfig,
        "params": {
            "r": 8,
            "alpha": 32,
            "rank_dropout": 0.1,
            "module_dropout": 0.05,  # Added to improve robustness
            "decompose_both": True,  # Enabled for a more effective decomposition
            "decompose_factor": 4,  # Factor for Kronecker product decomposition
            "target_modules": [
                "k_proj",
                "q_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "down_proj",
                "up_proj",
            ],
        },
    },
    {
        "name": "AdaLoRA",
        "config_class": AdaLoraConfig,
        "params": {
            "init_r": 8,  # Reduced to lower initial computational cost
            "r": 12,  # Increased progressively to improve adaptation capability
            "beta1": 0.9,  # Standard adjustment to stabilize convergence
            "beta2": 0.95,  # Adjusted to provide better training smoothness
            "tinit": 100,  # Reduced to start adaptation sooner
            "tfinal": 1200,  # Increased for greater adaptation towards the end of training
            "deltaT": 20,  # Longer interval for less frequent adjustments
        },
    },
    {
        "name": "X-LoRA",
        "config_class": XLoraConfig,
        "params": {
            "r": 8,  # Reduced to lower computational cost
            "xlora_dropout": 0.1,  # Increased to prevent overfitting
        },
    },
    {
        "name": "IA3",
        "config_class": IA3Config,
        "params": {
            "target_modules": [
                "k_proj",
                "q_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "down_proj",
                "up_proj",
            ],
            "feedforward_modules": "down_proj",  # Added to explicitly handle feedforward modules
        },
    },
]


def run20241122A() -> None:
    """
    Iterates over different Parameter-Efficient Fine-Tuning (PEFT) methods to train a
    specified model. For each PEFT method, it configures the training environment,
    loads the model and dataset, applies the PEFT configuration, and initiates the
    training process.

    Methods Tested:
        - LoRA
        - LoHa
        - LoKr
        - AdaLoRA
        - X-LoRA
        - IA3

    Raises:
        ValueError: If an unsupported PEFT method is encountered.
    """
    model_name: str = "mistralai/Mistral-7B-v0.3"
    dataset_name: str = "GAIR/lima"

    for method_info in PEFT_METHODS:
        method_name: str = method_info["name"]
        config_class = method_info["config_class"]
        config_params = method_info["params"]

        # Define unique directory for each PEFT method
        model_path: str = locate_data_path(
            section="explore-PEFTs",
            dir_name=clean_string(f"{model_name}_{method_name}"),
        )

        eval_steps: int = 10
        save_steps: int = 45
        warmup_steps: int = 25
        max_steps: int = 1  # 00

        # Training configuration
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

        # Initialize PEFT configuration using unpacked parameters
        try:
            peft_config = config_class(**config_params)
        except TypeError as e:
            raise ValueError(f"Error initializing {method_name} configuration: {e}")

        # Initialize dataset and model interfaces
        dataset_interface: DatasetInterface = DatasetInterface(
            dataset_name=dataset_name, model_name=model_name
        )
        model_interface: ModelInterface = ModelInterface()
        model_interface.load_model(name=model_name)
        model_interface.load_PEFT_config(config=peft_config)
        model_interface.load_dataset(interface=dataset_interface)

        print(f"Training with PEFT method: {method_name} on model: {model_name}")

        # Start training
        model_interface.train(
            method=SFTTrainer,
            arguments=training_arguments,
        )

        # Cleanup resources
        model_interface.cleanup_model()
        # dataset_interface.cleanup_dataset()
        torch.cuda.empty_cache()
        print(f"Completed training with PEFT method: {method_name}\n")


if __name__ == "__main__":
    run20241122A()
