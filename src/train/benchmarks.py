# Standard Library dependencies
import gc
import os
import json
from pathlib import Path
from typing import Optional, Union

# ML dependencies
import torch
from torch._tensor import Tensor
from datasets import Dataset, load_dataset
from transformers.tokenization_utils_base import BatchEncoding
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

# Other dependencies
from tqdm import tqdm

# Internal dependencies:
from src.utils.interfaces import DatasetInterface, ModelInterface
from src.utils.benchmarks import PerformanceBenchmark
from src.utils.extra import (
    load_model_tokenizer,
    locate_data_path,
    get_dataset_subset,
    ensure_punkt_available,
)
from src.ifeval.evaluation_main import main as ifeval_main


device: str = "cuda" if torch.cuda.is_available() else "cpu"


def execute_performance_benchmark(
    id: str = "A", checkpoint_dir_name: Optional[str] = None
) -> None:
    DATASET_NAME: str = "argilla/ifeval-like-data"
    dir_name: Union[str, None] = checkpoint_dir_name  # force checkpoint directory name

    runs_path: str = locate_data_path(f"final-model-train/{id}/runs")
    files: list = list(sorted(os.listdir(runs_path)))
    checkpoints: list = [file for file in files if file.startswith("checkpoint-")]

    if dir_name is None and len(checkpoints) > 0:
        dir_name = checkpoints[-1]
        assert isinstance(dir_name, str)

    if dir_name is not None:
        assert isinstance(dir_name, str)
        checkpoint_path: str = os.path.join(runs_path, dir_name)
        model_interface: ModelInterface
        model_interface = ModelInterface.from_checkpoint(
            checkpoint_path=checkpoint_path
        )
        model: str = model_interface.model
        model_name: str = model_interface.name

        tokenizer: PreTrainedTokenizerFast = load_model_tokenizer(model_name=model_name)
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
                print(f"Pad Token ID set to EOS Token ID: {tokenizer.pad_token_id}")
            elif tokenizer.bos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.bos_token_id
                print(f"Pad Token ID set to BOS Token ID: {tokenizer.pad_token_id}")
            else:
                tokenizer.pad_token_id = 0  # Default value
                print(
                    f"Pad Token ID set to default value: {tokenizer.pad_token_id}"
                )

        # Verify and convert pad_token_id to integer if it's a string
        if isinstance(tokenizer.pad_token_id, str):
            ids =  tokenizer.convert_tokens_to_ids(tokenizer.pad_token_id)
            tokenizer.pad_token_id = ids
            print(f"Converted pad_token_id to integer: {tokenizer.pad_token_id}")

        dataset: Dataset = load_dataset(path=DATASET_NAME)
        benchmark = PerformanceBenchmark(model, tokenizer, dataset["test"])
        results: dict = benchmark.run_benchmark()

        # save benchmark results
        rel_path: str = f"final-model-train/{id}/performance-benchmarks"
        output_file: Path = Path(locate_data_path(rel_path=rel_path))
        file_name: str = f"{dir_name}-results.jsonl"
        file_path: str = str(output_file / file_name)
        with open(file_path, "w") as json_file:
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


def execute_ifeval_response(
    id: str = "A", checkpoint_dir_name: Optional[str] = None
) -> None:

    DATASET_NAME: str = "google/IFEval"
    dir_name: Union[str, None] = checkpoint_dir_name  # force checkpoint directory name

    runs_path: str = locate_data_path(f"final-model-train/{id}/runs")
    files: list = list(sorted(os.listdir(runs_path)))
    checkpoints: list = [file for file in files if file.startswith("checkpoint-")]

    if dir_name is None and len(checkpoints) > 0:
        dir_name = checkpoints[-1]
        assert isinstance(dir_name, str)

    if dir_name is not None:
        assert isinstance(dir_name, str)
        checkpoint_path: str: str = os.path.join(runs_path, dir_name)

        # Step 0. Load model
        torch.cuda.empty_cache()
        model_interface: ModelInterface
        model_interface = ModelInterface.from_checkpoint(
            checkpoint_path=checkpoint_path
        )
        model: str = model_interface.model
        model_name: str = model_interface.name

        # Step 1: Load tokenizer
        tokenizer: PreTrainedTokenizerFast = load_model_tokenizer(model_name=model_name)
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
                print(f"Pad Token ID set to EOS Token ID: {tokenizer.pad_token_id}")
            elif tokenizer.bos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.bos_token_id
                print(f"Pad Token ID set to BOS Token ID: {tokenizer.pad_token_id}")
            else:
                tokenizer.pad_token_id = 0  # Default value
                print(
                    f"Pad Token ID set to default value: {tokenizer.pad_token_id}"
                )

        # Verify and convert pad_token_id to integer if it's a string
        if isinstance(tokenizer.pad_token_id, str):
            ids =  tokenizer.convert_tokens_to_ids(tokenizer.pad_token_id)
            tokenizer.pad_token_id = ids
            print(f"Converted pad_token_id to integer: {tokenizer.pad_token_id}")

        # Step 2: Load the google/IFEval dataset
        dataset: Dataset = load_dataset(path=DATASET_NAME)
        dataset = get_dataset_subset(dataset["train"], prop=0.4, shuffle=False)

        # Step 3: Generate predictions on the dataset
        output_file: Path = Path(locate_data_path(f"final-model-train/{id}/ifeval"))
        file_name: str = f"{dir_name}-responses.jsonl"
        file_path: str = str(output_file / file_name)
        with open(file_path, "w", encoding="utf-8") as f_out:
            for sample in tqdm(
                dataset
            ):  # Use 'validation' or 'train' split if 'test' is not available
                # Adjust the field name ("prompt") based on the dataset's structure
                input_text: str = sample["prompt"]
                max_lenght: int = 256
                prompt: str = input_text[:max_lenght]  # prepare the input prompt

                # Tokenize input
                # inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
                inputs: BatchEncoding = tokenizer(
                    prompt,
                    truncation=True,
                    max_length=max_lenght,
                    return_tensors="pt",
                )
                input_ids: Tensor = inputs["input_ids"]
                att_mask: Tensor = inputs["attention_mask"]

                outputs = model.generate(
                    input_ids=input_ids.to(dtype=torch.long, device=device),
                    attention_mask=att_mask.to(dtype=torch.long, device=device),
                    max_new_tokens=max_lenght,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )

                # Decode output
                generated_text: str = tokenizer.decode(
                    outputs[0], skip_special_tokens=True
                )

                # Since the model may include the prompt in its output, we extract the
                # generated response
                response: str = generated_text[len(prompt) :]

                # Prepare the JSON object
                json_obj: dict = {"prompt": prompt, "response": response}

                # Write the JSON object to file
                f_out.write(json.dumps(json_obj) + "\n")

        # Cleanup
        model_interface.cleanup_model()
        del model
        del tokenizer
        del model_interface
        torch.cuda.empty_cache()
        gc.collect()


def execute_ifeval_evaluation(
    id: str = "A", checkpoint_dir_name: Optional[str] = None
) -> None:
    dir_name: Union[str, None] = checkpoint_dir_name  # force checkpoint directory name

    input_file = str(Path(locate_data_path("datasets")) / "ifeval.jsonl")
    ifeval_folder: Path = Path(locate_data_path("final-model-train")) / id / "ifeval"
    runs_path: str = locate_data_path(f"final-model-train/{id}/runs")
    files: list = list(sorted(os.listdir(runs_path)))
    checkpoints: list = [file for file in files if file.startswith("checkpoint-")]

    if dir_name is None and len(checkpoints) > 0:
        dir_name: str = checkpoints[-1]
        assert isinstance(dir_name, str)

    if dir_name is not None:
        assert isinstance(dir_name, str)
        responses_data: str = str(ifeval_folder / f"{dir_name}-responses.jsonl")
        output_dir: str = str(ifeval_folder / f"{dir_name}-results")
        ifeval_main(input_file, responses_data, output_dir)


if __name__ == "__main__":

    # do not remove this
    ensure_punkt_available()

    # BENCHMARKS
    # execute_performance_benchmark()
    execute_ifeval_response()
    execute_ifeval_evaluation()
    pass
