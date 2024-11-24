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
    get_src_path,
    locate_data_path,
    get_dataset_subset,
    ensure_punkt_available,
)
from src.ifeval.evaluation_main import main as ifeval_main

MODELS: list[dict[str, str]] = [
    {"name": "meta-llama/Llama-3.1-8B"},
    {"name": "mistralai/Mistral-7B-v0.3"},
    {"name": "Qwen/Qwen2.5-7B"},
    {"name": "ibm-granite/granite-3.0-8b-base"},
]


device: str = "cuda" if torch.cuda.is_available() else "cpu"


def execute_performance_benchmark(id: str = "A") -> None:
    DATASET_NAME: str = "argilla/ifeval-like-data"

    for model_info in MODELS:
        model_interface: ModelInterface
        model_interface = ModelInterface.load_model(name=model_info["name"])
        model: str = model_interface.model
        model_name: str = model_interface.name

        tokenizer: Dataset = load_model_tokenizer(model_name=model_name)
        dataset: Dataset = load_dataset(path=DATASET_NAME)
        benchmark = PerformanceBenchmark(model, tokenizer, dataset["test"])
        results: dict = benchmark.run_benchmark()

        # create json filename
        model_name: str = model_info["name"]
        clean_model_name: str = clean_string(model_name)
        file_name: str = f"{clean_model_name}-pretrained-results.jsonl"

        # save benchmark results
        rel_path: str = f"explore-base-models/{id}/performance-benchmarks"
        output_file: Path = Path(locate_data_path(rel_path=rel_path))
        file_path: str = str(output_file / file_name)
        with open(file_path, "w") as json_file:
            json.dump(obj=results, fp=json_file, indent=4)

        # display results
        print(results)
        print()
        model_interface.cleanup_model()
        del model
        del tokenizer
        del model_interface
        torch.cuda.empty_cache()
        gc.collect()


def execute_ifeval_response(id: str = "A") -> None:
    DATASET_NAME: str = "google/IFEval"

    for model_info in MODELS:
        model_interface: ModelInterface
        model_interface = ModelInterface.load_model(name=model_info["name"])
        model: str = model_interface.model
        model_name: str = model_interface.name

        tokenizer: Dataset = load_model_tokenizer(model_name=model_name)
        dataset: Dataset = load_dataset(path=DATASET_NAME)
        dataset = get_dataset_subset(dataset["train"], prop=0.4, shuffle=False)

        # create json filename
        model_name: str = model_info["name"]
        clean_model_name: str = clean_string(model_name)
        file_name: str = f"{clean_model_name}-pretrained-responses.jsonl"

        # save benchmark results
        rel_path: str = f"explore-base-models/{id}/ifeval"
        output_file: Path = Path(locate_data_path(rel_path=rel_path))
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
                    # attention_mask=inputs["attention_mask"],
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





def execute_ifeval_evaluation(id: str = "A") -> None:
    input_file = str(Path(locate_data_path("datasets")) / "ifeval.jsonl")
    ifeval_folder: Path = Path(locate_data_path("explore-base-models")) / id / "ifeval"
    for model_info in MODELS:
        filename: str = f"{clean_string(model_info["name"])}-pretrained"
        responses_data: str = str(ifeval_folder / f"{filename}-responses.jsonl")
        output_dir: str = str(ifeval_folder / f"{filename}-results")
        ifeval_main(input_file, responses_data, output_dir)


if __name__ == "__main__":

    # do not remove this
    ensure_punkt_available()

    # BENCHMARKS
    # execute_performance_benchmark()
    # execute_ifeval_response()
    # execute_ifeval_evaluation()
    pass