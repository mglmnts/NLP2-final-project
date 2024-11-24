# Standard Library dependencies
import gc
import os
import json
from pathlib import Path

# ML dependencies
import torch
from datasets import Dataset, load_dataset

# Other dependencies
from tqdm import tqdm

# Internal dependencies:
from src.utils.interfaces import DatasetInterface, ModelInterface
from src.utils.benchmarks import PerformanceBenchmark
from src.utils.extra import (
    load_model_tokenizer,
    clean_string,
    locate_data_path,
    get_dataset_subset,
)

from src.ifeval.evaluation_main import main as ifeval_main


MODELS: list[dict[str, str]] = [
    {"name": "mistralai/Mistral-7B-v0.3"},
    {"name": "meta-llama/Llama-3.1-8B"},
    {"name": "Qwen/Qwen2.5-7B"},
    {"name": "ibm-granite/granite-3.0-8b-base"},
]


device: str = "cuda" if torch.cuda.is_available() else "cpu"


def execute_performance_benchmark(id: str = "A") -> None:
    DATASET_NAME: str = "GAIR/lima"
    for model in MODELS:
        torch.cuda.empty_cache()
        CHECKPOINTS_PATH: str = locate_data_path(
            f"explore-PEFTs/{id}/runs/{clean_string(model['name'])}"
        )
        checkpoint_path: str = None
        for dir_name in os.listdir(CHECKPOINTS_PATH):
            if dir_name.startswith("checkpoint-"):
                checkpoint_path: str = os.path.join(CHECKPOINTS_PATH, dir_name)
        if checkpoint_path:
            model_interface: ModelInterface
            model_interface = ModelInterface.from_checkpoint(
                checkpoint_path=checkpoint_path
            )
            model: str = model_interface.model
            model_name: str = model_interface.name

            tokenizer: Dataset = load_model_tokenizer(model_name=model_name)
            dataset: Dataset = load_dataset(path=DATASET_NAME)
            benchmark = PerformanceBenchmark(model, tokenizer, dataset["test"])
            results: dict = benchmark.run_benchmark()

            # save benchmark results
            rel_path: str = Path("explore-PEFTs") / id
            data_path: str = locate_data_path(rel_path=rel_path)
            os.makedirs(data_path, exist_ok=True)
            json_path: str = os.path.join(
                dir_name, f"{clean_string(model_name)}-results.jsonl"
            )
            with open(json_path, "w") as json_file:
                json.dump(results, json_file, indent=4)

            # display results
            print(results)
            print()
            model_interface.cleanup_model()
            del model
            del tokenizer
            del model_interface
            torch.cuda.empty_cache()
            gc.collect()


def execute_ifeval_response(id: str ="A") -> None:
    DATASET_NAME: str = "google/IFEval"
    for model in MODELS:
        torch.cuda.empty_cache()
        CHECKPOINTS_PATH: str = locate_data_path(
            f"explore-PEFTs/{clean_string(model['name'])}"
        )
        checkpoint_path: str = None
        for dir_name in os.listdir(CHECKPOINTS_PATH):
            if dir_name.startswith("checkpoint-"):
                checkpoint_path: str = os.path.join(CHECKPOINTS_PATH, dir_name)

        if checkpoint_path:
            # Step 0. Load model
            torch.cuda.empty_cache()
            model_interface: ModelInterface
            model_interface = ModelInterface.from_checkpoint(
                checkpoint_path=checkpoint_path
            )
            model: str = model_interface.model
            model_name: str = model_interface.name

            # Step 1: Load tokenizer
            tokenizer: Dataset = load_model_tokenizer(model_name=model_name)
            # Step 2: Load the google/IFEval dataset
            dataset: Dataset = load_dataset(path=DATASET_NAME)
            dataset = get_dataset_subset(dataset["train"], prop=0.002, shuffle=False)

            # Step 3: Generate predictions on the dataset
            output_file: Path = Path(locate_data_path(f"explore-PEFTs/{id}/ifeval"))
            clean_model_name: str = clean_string(model_name)
            clean_dataset_name: str = clean_string(DATASET_NAME)
            file_name: str = f"{clean_model_name}-{clean_dataset_name}-responses.jsonl"
            file_path: str = str(output_file / file_name)
            with open(file_path, "w", encoding="utf-8") as f_out:
                for sample in tqdm(
                    dataset
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
                        attention_mask=inputs["attention_mask"],
                        max_length=256,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                    # Decode output
                    generated_text = tokenizer.decode(
                        outputs[0], skip_special_tokens=True
                    )

                    # Since the model may include the prompt in its output, we extract the
                    # generated response
                    response = generated_text[len(prompt) :]

                    # Prepare the JSON object
                    json_obj = {"prompt": prompt, "response": response}

                    # Write the JSON object to file
                    f_out.write(json.dumps(json_obj) + "\n")

            # Cleanup
            model_interface.cleanup_model()
            del model
            del tokenizer
            del model_interface
            torch.cuda.empty_cache()
            gc.collect()


def execute_ifeval_evaluation(id: str ="A") -> None:
    DATASET_NAME: str = "google/IFEval"
    input_file = str(Path(locate_data_path("datasets")) / "ifeval.jsonl")
    ifeval_folder: Path = Path(locate_data_path("explore-PEFTs")) / id / "ifeval"
    for model in MODELS:
        clean_model_name: str = clean_string(model["name"])
        clean_dataset_name: str = clean_string(DATASET_NAME)
        clean_mixed_name: str = f"{clean_model_name}-{clean_dataset_name}"
        responses_data: str = str(ifeval_folder / f"{clean_mixed_name}-responses.jsonl")
        output_dir: str = str(ifeval_folder / f"{clean_mixed_name}-results")
        ifeval_main(input_file, responses_data, output_dir)


if __name__ == "__main__":

    # execute_performance_benchmark()
    # execute_ifeval_response()
    # execute_ifeval_evaluation()
    pass
