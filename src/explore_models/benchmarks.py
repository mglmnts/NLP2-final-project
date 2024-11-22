# Standard Library dependencies
import os
import json

# ML dependencies
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

# Internal dependencies:
from src.utils.interfaces import DatasetInterface, ModelInterface
from src.utils.benchmarks import PerformanceBenchmark
from src.utils.extra import clean_string, locate_data_path


MODELS: list[dict[str, str]] = [
    {"name": "meta-llama/Llama-3.1-8B"},
    {"name": "mistralai/Mistral-7B-v0.3"},
    {"name": "Qwen/Qwen2.5-7B"},
    {"name": "ibm-granite/granite-3.0-8b-base"},
]


def execute_performance_benchmark() -> None:
    DATASET_NAME: str = "GAIR/lima"
    checkpoint_dirs: list[str] = [clean_string(model["name"]) for model in MODELS]

    for dir_name in checkpoint_dirs:
        data_path: str = locate_data_path(section="explore-models", dir_name=dir_name)

        for file in os.listdir(data_path):
            if ".pt" in file:
                checkpoint_path: str = file

        model_interface: ModelInterface
        model_interface = ModelInterface.from_checkpoint(
            checkpoint_path=checkpoint_path
        )
        model: str = model_interface.model
        model_name: str = model_interface.name
        dataset_interface: DatasetInterface
        dataset_interface = DatasetInterface(
            dataset_name=DATASET_NAME, model_name=model_name
        )
        tokenizer: Dataset = dataset_interface.tokenizer
        dataset_test: Dataset = dataset_interface.test

        benchmark = PerformanceBenchmark(model, tokenizer, dataset_test)
        results: dict = benchmark.run_benchmark()

        # save benchmark results
        os.makedirs(data_path, exist_ok=True)
        json_path: str = os.path.join(dir_name, "benchmark_results.json")
        with open(json_path, "w") as json_file:
            json.dump(results, json_file, indent=4)

        # display results
        print(results)
        print()

        return None


def execute_ifeval_benchmark() -> None:

    pass

    return None


if __name__ == "__main__":

    execute_performance_benchmark()
