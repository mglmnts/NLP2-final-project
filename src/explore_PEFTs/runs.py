# Standard Library dependencies
import gc
import time
from pathlib import Path
from typing import Type

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
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig

# Internal dependencies
from src.utils.interfaces import DatasetInterface, ModelInterface
from src.utils.extra import clean_string, locate_data_path

# Global Variables
device: str = "cuda" if torch.cuda.is_available() else "cpu"

# # Define PEFT methods with their corresponding configuration classes and parameters
MISTRAL_PEFT_METHODS: list[dict] = [
    {
        "name": "LoRA",
        "config_class": LoraConfig,
        "params": {
            "r": 16,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "bias": "none",
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
        "name": "AdaLoRA",
        "config_class": AdaLoraConfig,
        "params": {
            "init_r": 8,  # Reduced to lower initial computational cost
            "r": 12,  # Increased progressively to improve adaptation capability
            "beta1": 0.9,  # Standard adjustment to stabilize convergence
            "beta2": 0.95,  # Adjusted to provide better training smoothness
            "tinit": 100,  # Reduced to start adaptation sooner
            "tfinal": 1200,  # Increased for greater adaptation towards the training end
            "deltaT": 20,  # Longer interval for less frequent adjustments
            "target_modules": [  # Agregado para evitar el error
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
            "feedforward_modules": "down_proj",
        },
    },
]

# Define PEFT methods with their corresponding configuration classes and parameters
GENERIC_PEFT_METHODS: list[dict] = [
    {
        "name": "LoRA",
        "config_class": LoraConfig,
        "params": {
            "r": 8,  # small -> Low computational cost; high -> good learning capability
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
            "r": 1,  # Reduced for better efficiency
            "alpha": 32,  # Increased to provide better adjustment capability
            "rank_dropout": 0.2,  # Dropout increased to reduce the risk of overfitting
            "module_dropout": 0.1,  # To prevent LoHa layers from being always active
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
            "tfinal": 1200,  # Increased for greater adaptation towards the training end
            "deltaT": 20,  # Longer interval for less frequent adjustments
            "target_modules": [  # Agregado para evitar el error
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
        "name": "X-LoRA",
        "config_class": XLoraConfig,
        "params": {
            "hidden_size": 4096,  # Adjusted to match Mistral 7B's architecture
            "adapters": {},  # This can be populated with specific adapter paths
            "enable_softmax": True,  # Enable softmax for better classifier behavior
            "enable_softmax_topk": False,  # Use dense method for all LoRA experts
            "softmax_temperature": 1.0,  # Default sharpness for predictions
            "layerwise_scalings": True,  # Layerwise scaling for better granularity
            "top_k_lora": None,  # Do not sparsely select top-k adapters
            "xlora_depth": 2,  # Increase depth for better classifier capacity
            "xlora_size": 4096,  # Adjusted to align with Mistral's hidden size
            "xlora_dropout_p": 0.1,  # Lower for balance regularization and learning
            "use_trainable_adapters": True,  # Allow adapters to be updated in training
            "scaling_pass_value": 0.0,  # Default value
            "global_scaling_weight": 1.0,  # Default weight for scaling outputs
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
            "feedforward_modules": "down_proj",
        },
    },
]


FINAL_TESTS: list[dict] = [
    {
        "name": "LoRA1",
        "config_class": LoraConfig,
        "params": {
            "r": 16,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "bias": "none",
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
        "name": "LoRA2",
        "config_class": LoraConfig,
        "params": {
            "r": 32,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "bias": "none",
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
        "name": "LoRA3",
        "config_class": LoraConfig,
        "params": {
            "r": 16,
            "lora_alpha": 16,
            "lora_dropout": 0.15,
            "bias": "none",
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
        "name": "LoRA4",
        "config_class": LoraConfig,
        "params": {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "bias": "none",
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
        "name": "AdaLoRA1",
        "config_class": AdaLoraConfig,
        "params": {
            "init_r": 8,  # Reduced to lower initial computational cost
            "r": 12,  # Increased progressively to improve adaptation capability
            "beta1": 0.9,  # Standard adjustment to stabilize convergence
            "beta2": 0.95,  # Adjusted to provide better training smoothness
            "tinit": 100,  # Reduced to start adaptation sooner
            "tfinal": 1200,  # Increased for greater adaptation towards the training end
            "deltaT": 20,  # Longer interval for less frequent adjustments
            "target_modules": [  # Agregado para evitar el error
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
        "name": "AdaLoRA2",
        "config_class": AdaLoraConfig,
        "params": {
            "init_r": 8,  # Reduced to lower initial computational cost
            "r": 12,  # Increased progressively to improve adaptation capability
            "beta1": 0.85,  # Standard adjustment to stabilize convergence
            "beta2": 0.85,  # Adjusted to provide better training smoothness
            "tinit": 100,  # Reduced to start adaptation sooner
            "tfinal": 1200,  # Increased for greater adaptation towards the training end
            "deltaT": 1,  # Longer interval for less frequent adjustments
            "target_modules": [  # Agregado para evitar el error
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
        "name": "AdaLoRA3",
        "config_class": AdaLoraConfig,
        "params": {
            "init_r": 8,  # Reduced to lower initial computational cost
            "r": 12,  # Increased progressively to improve adaptation capability
            "beta1": 0.85,  # Standard adjustment to stabilize convergence
            "beta2": 0.85,  # Adjusted to provide better training smoothness
            "tinit": 100,  # Reduced to start adaptation sooner
            "tfinal": 1200,  # Increased for greater adaptation towards the training end
            "deltaT": 10,  # Longer interval for less frequent adjustments
            "target_modules": [  # Agregado para evitar el error
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
        "name": "AdaLoRA4",
        "config_class": AdaLoraConfig,
        "params": {
            "init_r": 8,  # Reduced to lower initial computational cost
            "r": 8,  # Increased progressively to improve adaptation capability
            "beta1": 0.85,  # Standard adjustment to stabilize convergence
            "beta2": 0.85,  # Adjusted to provide better training smoothness
            "tinit": 100,  # Reduced to start adaptation sooner
            "tfinal": 1200,  # Increased for greater adaptation towards the training end
            "deltaT": 10,  # Longer interval for less frequent adjustments
            "target_modules": [  # Agregado para evitar el error
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
        "name": "AdaLoRA5",
        "config_class": AdaLoraConfig,
        "params": {
            "init_r": 8,  # Reduced to lower initial computational cost
            "r": 16,  # Increased progressively to improve adaptation capability
            "beta1": 0.85,  # Standard adjustment to stabilize convergence
            "beta2": 0.85,  # Adjusted to provide better training smoothness
            "tinit": 100,  # Reduced to start adaptation sooner
            "tfinal": 2000,  # Increased for greater adaptation towards the training end
            "deltaT": 10,  # Longer interval for less frequent adjustments
            "target_modules": [  # Agregado para evitar el error
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
]


def run_experimet_A(id="A") -> None:
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
    dataset_name: str = "argilla/ifeval-like-data"
    PEFT_METHODS: list[dict] = MISTRAL_PEFT_METHODS

    for method_info in PEFT_METHODS:

        if "ia3" not in method_info["name"].lower():
            continue

        method_name: str = method_info["name"]
        config_class = method_info["config_class"]
        config_params = method_info["params"]

        # Define unique directory for each PEFT method
        rel_path: Path = Path("explore-PEFTs")
        rel_path = rel_path / id / "runs" / f"{clean_string(model_name)}-{method_name}"
        model_path: str = locate_data_path(rel_path=str(rel_path))

        # Training timing control
        eval_steps: int = 25
        save_steps: int = 45
        warmup_steps: int = 25
        max_steps: int = 300  # 00

        # Training configuration
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
            learning_rate=1e-4,
            eval_steps=eval_steps,
            max_steps=max_steps,
            save_steps=save_steps,
            warmup_steps=warmup_steps,
            lr_scheduler_type="linear",
            report_to="tensorboard",
            logging_dir=(Path(model_path) / "logs"),
            max_seq_length=512,
        )

        # Initialize PEFT configuration using unpacked parameters
        try:
            peft_config: Type[TrainingArguments] = config_class(**config_params)
        except TypeError as e:
            raise ValueError(f"Error initializing {method_name} configuration: {e}")

        dataset_interface: DatasetInterface = DatasetInterface(
            dataset_name=dataset_name, model_name=model_name
        )
        gc.collect()
        model_interface: ModelInterface = ModelInterface()
        model_interface.load_model(name=model_name)
        model_interface.load_PEFT_config(config=peft_config)
        model_interface.load_dataset(interface=dataset_interface)

        print(
            f"\n\n\nTraining with PEFT method: {method_name} on model: {model_name}\n"
        )
        t0 = time.time()
        model_interface.train(method=SFTTrainer, arguments=training_arguments)

        torch.cuda.empty_cache()
        print(torch.cuda.is_available())
        model_interface.cleanup_model()
        dataset_interface.cleanup_dataset()
        del model_interface
        del dataset_interface
        gc.collect()
        print(time.time() - t0)


def run_experimet_B(id="B") -> None:
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
    dataset_name: str = "argilla/ifeval-like-data"
    PEFT_METHODS: list[dict] = FINAL_TESTS

    for method_info in PEFT_METHODS:

        method_name: str = method_info["name"]
        config_class = method_info["config_class"]
        config_params = method_info["params"]

        # Define unique directory for each PEFT method
        rel_path: Path = Path("explore-PEFTs")
        rel_path = rel_path / id / "runs" / f"{clean_string(model_name)}-{method_name}"
        model_path: str = locate_data_path(rel_path=str(rel_path))

        # Training timing control
        eval_steps: int = 1000
        save_steps: int = 1000
        warmup_steps: int = 25
        max_steps: int = 75  # 00

        # Training configuration
        training_arguments: SFTConfig = SFTConfig(
            output_dir=model_path,
            eval_strategy="steps",
            do_eval=False,
            optim="paged_adamw_8bit",
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            per_device_eval_batch_size=4,
            log_level="debug",
            logging_steps=10,
            learning_rate=1e-4,
            eval_steps=eval_steps,
            max_steps=max_steps,
            save_steps=save_steps,
            warmup_steps=warmup_steps,
            lr_scheduler_type="linear",
            report_to="tensorboard",
            logging_dir=(Path(model_path) / "logs"),
            max_seq_length=512,
        )

        # Initialize PEFT configuration using unpacked parameters
        try:
            peft_config: Type[TrainingArguments] = config_class(**config_params)
        except TypeError as e:
            raise ValueError(f"Error initializing {method_name} configuration: {e}")

        dataset_interface: DatasetInterface = DatasetInterface(
            dataset_name=dataset_name, model_name=model_name
        )
        gc.collect()
        model_interface: ModelInterface = ModelInterface()
        model_interface.load_model(name=model_name)
        model_interface.load_PEFT_config(config=peft_config)
        model_interface.load_dataset(interface=dataset_interface)

        print(
            f"\n\n\nTraining with PEFT method: {method_name} on model: {model_name}\n"
        )
        model_interface.train(method=SFTTrainer, arguments=training_arguments)

        torch.cuda.empty_cache()
        print(torch.cuda.is_available())
        model_interface.cleanup_model()
        dataset_interface.cleanup_dataset()
        del model_interface
        del dataset_interface
        gc.collect()


if __name__ == "__main__":
    run_experimet_B()
