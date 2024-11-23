# Standard Library dependencies
import os
import json

# ML dependencies
import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

# Internal dependencies:
from src.utils.interfaces import DatasetInterface, ModelInterface
from src.utils.benchmarks import PerformanceBenchmark
from src.utils.extra import clean_string, locate_data_path

from datasets import load_dataset
import json
from tqdm import tqdm


MODELS: list[dict[str, str]] = [
    {"name": "meta-llama/Llama-3.1-8B"},
    {"name": "mistralai/Mistral-7B-v0.3"},
    {"name": "Qwen/Qwen2.5-7B"},
    {"name": "ibm-granite/granite-3.0-8b-base"},
]


device: str = "cuda" if torch.cuda.is_available() else "cpu"


def execute_performance_benchmark() -> None:
    DATASET_NAME: str = "GAIR/lima"
    for model in MODELS:
        CHECKPOINTS_PATH: str = locate_data_path(
            "explore-models", clean_string(model["name"])
        )
        for dir_name in os.listdir(CHECKPOINTS_PATH):
            if dir_name.startswith("checkpoint-"):
                checkpoint_path: str = os.path.join(CHECKPOINTS_PATH, dir_name)

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
        data_path: str = locate_data_path("explore-models", "benchmarks")
        os.makedirs(data_path, exist_ok=True)
        json_path: str = os.path.join(dir_name, "benchmark_results.json")
        with open(json_path, "w") as json_file:
            json.dump(results, json_file, indent=4)

        # display results
        print(results)
        print()


def execute_ifeval_benchmark() -> None:
    DATASET_NAME: str = "GAIR/lima"
    for model in MODELS:
        CHECKPOINTS_PATH: str = locate_data_path(
            "explore-models", clean_string(model["name"])
        )
        for dir_name in os.listdir(CHECKPOINTS_PATH):
            if dir_name.startswith("checkpoint-"):
                checkpoint_path: str = os.path.join(CHECKPOINTS_PATH, dir_name)

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
        # Step 1: Load tokenizer
        tokenizer: Dataset = dataset_interface.tokenizer
        # Step 2: Load the google/IFEval dataset
        dataset: Dataset = load_dataset("google/IFEval")
        # Step 3: Generate predictions on the dataset
        output_file = "model_responses.jsonl"
        with open(output_file, "w", encoding="utf-8") as f_out:
            for sample in tqdm(
                dataset["train"]
            ):  # Use 'validation' or 'train' split if 'test' is not available
                input_text = sample[
                    "prompt"
                ]  # Adjust the field name based on the dataset's structure

                # Prepare the input prompt
                prompt: str = input_text

                # Tokenize input
                inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

                # Generate output
                outputs = model.generate(
                    inputs,
                    max_length=256,
                    eos_token_id=tokenizer.eos_token_id,
                )

                # Decode output
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Since the model may include the prompt in its output, we extract the
                # generated response
                response = generated_text[len(prompt) :]

                # Prepare the JSON object
                json_obj = {"prompt": prompt, "response": response}

                # Write the JSON object to file
                f_out.write(json.dumps(json_obj) + "\n")


if __name__ == "__main__":

    execute_performance_benchmark()
