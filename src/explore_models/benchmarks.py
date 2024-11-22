# Standard Library dependencies
import os
from typing import Union

# ML dependencies
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

# Internal dependencies:
from src.utils.interfaces import DatasetInterface, ModelInterface
from src.utils.benchmarks import PerformanceBenchmark


CHECKPOINTS_PATH: str = "../data/explore-models"
DATASET_NAME: str = "GAIR/lima"


if __name__ == "__main__":

    for dir_name in os.listdir(CHECKPOINTS_PATH):

        checkpoint: str = "tal"  # el checkpoint que más accuracy tiene

        # Instantiate interfaces
        model_interface: ModelInterface = ModelInterface.from_checkpoint(checkpoint)
        dataset_interface: DatasetInterface = DatasetInterface(
            dataset_name=DATASET_NAME, model_name=model_interface.name
        )

        # Load model tockenizer and dataset
        model: Union[AutoModelForCausalLM, PeftModel] = ModelInterface.model
        tokenizer: PreTrainedTokenizerFast = DatasetInterface.tokenizer
        test_dataset: Dataset = DatasetInterface.test

        # Run benchmark
        benchmark: PerformanceBenchmark
        benchmark = PerformanceBenchmark(model, tokenizer, test_dataset)
        results: dict = benchmark.run_benchmark()
        print(results)
