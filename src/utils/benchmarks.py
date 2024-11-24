# ML dependencies
from torch._tensor import Tensor
from evaluate.module import EvaluationModule
from datasets import Dataset
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
)
from transformers.tokenization_utils_base import BatchEncoding
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from evaluate import load

# Standard Library dependencies
import time
from typing import Union


accuracy_score: EvaluationModule = load("accuracy")


class PerformanceBenchmark:
    """
    A class to benchmark the performance of a model on a given dataset.

    Attributes:
    -----------
    model : transformers.PreTrainedModel
        The model to be benchmarked.
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer associated with the model.
    dataset : datasets.Dataset
        The dataset on which the model's performance will be evaluated.
    """

    def __init__(self, model, tokenizer, dataset) -> None:
        """
        Initializes the PerformanceBenchmark with the provided model, tokenizer, and
        dataset.

        Parameters:
        -----------
        model : transformers.PreTrainedModel
            The model to be benchmarked.
        tokenizer : transformers.PreTrainedTokenizer
            The tokenizer for encoding the inputs for the model.
        dataset : datasets.Dataset
            The dataset on which the model's performance will be evaluated.
        """
        self.model: Union[AutoModelForCausalLM, PeftModel] = model
        self.tokenizer: PreTrainedTokenizerFast = tokenizer
        self.dataset: Dataset = dataset

    def compute_parameters(self) -> dict[str, int]:
        """
        Computes the total number of parameters and the number of trainable parameters.

        Returns:
        --------
        dict :
            A dictionary containing:
            - `total_params`: The total number of parameters in the model.
            - `trainable_params`: The number of trainable parameters in the model.
        """
        total_params: int = sum(
            p.numel() for p in self.model.parameters()
        )  # Total parameters
        trainable_params: int = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )  # Trainable parameters

        return {"total_params": total_params, "trainable_params": trainable_params}

    def compute_size(self) -> dict[str, Union[int, float]]:
        """
        Computes the size of the model in terms of the number of parameters
        and memory usage in megabytes (MB).

        Returns:
        --------
        dict :
            A dictionary containing the number of parameters (`num_params`) and
            the model size in MB (`model_size_mb`).
        """
        num_params: int = sum(p.numel() for p in self.model.parameters())
        model_size_mb: float = sum(
            p.element_size() * p.nelement() for p in self.model.parameters()
        ) / (1024**2)

        return {"num_params": num_params, "model_size_mb": model_size_mb}

    def time_pipeline(self) -> dict[str, float]:
        """
        Measures the total time and average time taken by the model to process
        the dataset.

        This method will use the tokenizer to encode the inputs before passing them
        to the model.

        Returns:
        --------
        dict :
            A dictionary containing the total processing time in seconds
            (`total_time_sec`) and the average time per example
            (`avg_time_per_example_sec`).
        """
        start_time: float = time.time()

        for example in self.dataset:
            key: str
            example_keys: list[str] = list(example.keys())
            if "conversations" in example_keys:
                key = "converstions"
            if "instruction" in example_keys:
                key = "instruction"
            inputs: Tensor = example[key]
            # Tokenize the input
            tokenized_input: BatchEncoding = self.tokenizer(
                inputs, return_tensors="pt"
            ).to(self.model.device)
            _ = self.model.generate(**tokenized_input, max_new_tokens=10)

        end_time: float = time.time()
        total_time: float = end_time - start_time
        avg_time_per_example: float = (
            total_time / len(self.dataset) if len(self.dataset) > 0 else float("inf")
        )

        return {
            "total_time_sec": total_time,
            "avg_time_per_example_sec": avg_time_per_example,
        }

    def compute_latency(self) -> dict[str, float]:
        """
        Computes the average latency of the model, defined as the time taken
        to process a single example from the dataset.

        Returns:
        --------
        dict :
            A dictionary containing the average latency in seconds (`avg_latency_sec`).
        """
        latencies: list[float] = []

        for example in self.dataset:
            key: str
            example_keys: list[str] = list(example.keys())
            if "conversations" in example_keys:
                key = "converstions"
            if "instruction" in example_keys:
                key = "instruction"
            inputs: Tensor = example[key]
            # Tokenize the input
            tokenized_input: BatchEncoding = self.tokenizer(
                inputs, return_tensors="pt"
            ).to(self.model.device)

            start_time: float = time.time()
            _ = self.model.generate(**tokenized_input, max_new_tokens=10)
            end_time: float = time.time()

            latencies.append(end_time - start_time)

        avg_latency = (
            sum(latencies) / len(latencies) if len(latencies) > 0 else float("inf")
        )
        return {"avg_latency_sec": avg_latency}

    def compute_throughput(self) -> dict[str, Union[float, int]]:
        """
        Computes the throughput of the model, defined as the number of examples
        processed per second.

        Returns:
        --------
        dict :
            A dictionary containing the throughput in examples per second
            (`throughput_examples_per_sec`).
        """
        start_time: float = time.time()

        for example in self.dataset:
            key: str
            example_keys: list[str] = list(example.keys())
            if "conversations" in example_keys:
                key = "converstions"
            if "instruction" in example_keys:
                key = "instruction"
            inputs: Tensor = example[key]
            # Tokenize the input
            tokenized_input: BatchEncoding = self.tokenizer(
                inputs, return_tensors="pt"
            ).to(self.model.device)
            _ = self.model.generate(**tokenized_input, max_new_tokens=10)

        end_time: float = time.time()
        total_time: float = end_time - start_time
        throughput: float = len(self.dataset) / total_time if total_time > 0 else 0

        return {"throughput_examples_per_sec": throughput}

    def run_benchmark(self) -> dict:
        """
        Runs all the benchmark metrics (size, time, latency, throughput, and FLOPs)
        and returns the results.

        Returns:
        --------
        dict :
            A dictionary containing all the computed metrics for the model.
            Includes size, parameters, time, latency, throughput, and FLOPs estimates.
        """
        metrics: dict = {}
        metrics["Size"] = self.compute_size()
        metrics["Parameters"] = self.compute_parameters()
        metrics["Time"] = self.time_pipeline()
        metrics["Latency"] = self.compute_latency()
        metrics["Throughput"] = self.compute_throughput()
        return metrics
