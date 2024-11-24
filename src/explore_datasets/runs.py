# Standard Library dependencies
import gc
from pathlib import Path

# ML dependencies
import torch
from peft import IA3Config
from trl import SFTTrainer, SFTConfig

# Internal dependencies
from src.utils.interfaces import DatasetInterface, ModelInterface
from src.utils.extra import clean_string, locate_data_path

# Global variables
device: str = "cuda" if torch.cuda.is_available() else "cpu"

# Model to use for dataset benchmarking
model_name: str = "mistralai/Mistral-7B-v0.3"

# Lista de datasets
DATASETS: list[dict[str, str]] = [
    # {"name": "GAIR/lima"},
    {"name": "databricks/databricks-dolly-15k"},
    {"name": "tatsu-lab/alpaca"},
    {"name": "argilla/ifeval-like-data"},
]


def run_experiment_A(id="A") -> None:
    for info in DATASETS:
        dataset_name: str = info["name"]
        # Directorio de salida específico para cada dataset
        rel_path: Path = Path("explore-datasets")
        clean_model_name: str = clean_string(model_name)
        clean_dataset_name: str = clean_string(model_name)
        rel_path = rel_path / id / "runs" / f"{clean_model_name}-{clean_dataset_name}"
        model_path: str = locate_data_path(rel_path=str(rel_path))

        # Training timing control
        eval_steps: int = 20  # 10
        save_steps: int = 20  # 45
        warmup_steps: int = 25
        max_steps: int = 100

        # Configuración de entrenamiento
        training_arguments: SFTConfig = SFTConfig(
            output_dir=model_path,
            eval_strategy="steps",  # Activate evaluation each "save_steps"
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

        # IA3 config
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
            feedforward_modules=None,  # specify feedforward modules
            fan_in_fan_out=False,
            task_type="CAUSAL_LM",
        )

        dataset_interface: DatasetInterface = DatasetInterface(
            dataset_name=dataset_name, model_name=model_name
        )
        model_interface: ModelInterface = ModelInterface()
        model_interface.load_model(name=model_name)
        model_interface.load_PEFT_config(config=peft_config)
        model_interface.load_dataset(interface=dataset_interface)

        print(f"\n\n\nTraining with Dataset: {dataset_name}\n")
        model_interface.train(
            method=SFTTrainer,
            arguments=training_arguments,
        )

        # Limpieza
        model_interface.cleanup_model()
        dataset_interface.cleanup_dataset()
        del model_interface
        del dataset_interface
        gc.collect()


if __name__ == "__main__":
    run_experiment_A()
