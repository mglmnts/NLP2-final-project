# Dependencias de la biblioteca estándar
import os
from pathlib import Path
from typing import Callable

# Dependencias de ML
import torch
from peft import IA3Config  # Importamos IA3Config en lugar de LoraConfig
from trl import SFTTrainer, SFTConfig

# Dependencias internas
from src.utils.interfaces import DatasetInterface, ModelInterface
from src.utils.extra import clean_string, locate_data_path

# Variables globales
device: str = "cuda" if torch.cuda.is_available() else "cpu"

# Modelo a utilizar
model_name: str = "mistralai/Mistral-7B-v0.3"

# Lista de datasets
DATASETS: list[dict[str, str]] = [
    {"name": "GAIR/lima"},
    {"name": "databricks/databricks-dolly-15k"},
    {"name": "tatsu-lab/alpaca"},
    {"name": "argilla/ifeval-like-data"},
]


def run_experiment() -> None:
    for info in DATASETS:
        dataset_name: str = info["name"]
        # Directorio de salida específico para cada dataset
        model_path: str = locate_data_path(
            section="explore-datasets",
            dir_name=clean_string(f"{model_name}_{dataset_name}"),
        )
        eval_steps: int = 10
        save_steps: int = 45
        warmup_steps: int = 25
        max_steps: int = 100

        # Configuración de entrenamiento
        training_arguments: SFTConfig = SFTConfig(
            output_dir=model_path,
            eval_strategy="steps",  # Activamos la evaluación cada cierto número de pasos
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

        # Configuración de IA3 para ajuste fino eficiente
        peft_config = IA3Config(
            target_modules=[
                "k_proj",
                "q_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "down_proj",
                "up_proj",
            ],
            feedforward_modules=None,  # Puedes especificar módulos de feedforward si aplica
            fan_in_fan_out=False,
            bias="none",
            task_type="CAUSAL_LM",
        )

        dataset_interface: DatasetInterface = DatasetInterface(
            dataset_name=dataset_name, model_name=model_name
        )
        model_interface: ModelInterface = ModelInterface()
        model_interface.load_model(name=model_name)
        model_interface.load_PEFT_config(config=peft_config)
        model_interface.load_dataset(interface=dataset_interface)

        print(f"Entrenando en el dataset: {dataset_name}")
        model_interface.train(
            method=SFTTrainer,
            arguments=training_arguments,
        )

        # Limpieza
        model_interface.cleanup_model()
        dataset_interface.cleanup_dataset()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run_experiment()
