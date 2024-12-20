# Standard Library dependencies
import os
import gc

import shutil
from typing import Optional, Type, Union

# ML dependencies
import torch
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from peft.config import PeftConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)
from transformers.tokenization_utils_base import BatchEncoding
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

# Internal dependencies
from src.utils.extra import load_model_tokenizer
from src.utils.filter_ifeval_data import filter_datset


# Custom Types
PEFTType = Union[LoraConfig]

# Global Variables
device: str = "cuda" if torch.cuda.is_available() else "cpu"
compute_dtype: torch.dtype = getattr(torch, "bfloat16")  # set computation dtype


class DatasetInterface:

    def __init__(
        self,
        dataset_name: str,
        model_name: Optional[str] = None,
        auto_split: Optional[bool] = True,
    ) -> None:
        """
        Initializes the DatasetInterface with a specified dataset and optional model.

        Args:
            dataset_name (Optional[str]):
                The name or path of the dataset to load.
            model_name (Optional[str]):
                The name or path of the model for tokenization.
            dataset (Optional[Union[Dataset, DatasetDict]]):
                A pre-loaded dataset to use.

        Raises:
            ValueError: If neither `dataset_name` nor `dataset` is provided.
        """

        self._dataset_name: str = dataset_name
        self._dataset: Union[Dataset, DatasetDict]
        self._tokenizer: Union[PreTrainedTokenizerFast, None] = None
        self._tokenized_dataset: Union[Dataset, DatasetDict]

        if self._dataset_name == "merge":
            self._dataset = None
            auto_split = False
        elif self._dataset_name == "argilla/ifeval-like-data":
            # dataset_name: str = "argilla-warehouse/ifeval-like-data"
            self._dataset = load_dataset(path=dataset_name, split="train")
            # self._dataset = filter_datset(self._dataset)
        else:
            self._dataset = load_dataset(path=dataset_name)
        # if self._dataset_name == "argilla/ifeval-like-data":
        #     assert False, self._dataset.keys()

        if auto_split:
            self.dataset_split()

        if model_name is not None:
            self.set_model(model_name=model_name)

        return None

    @property
    def tokenizer(self) -> PreTrainedTokenizerFast:
        """
        Returns the tokenizer associated with the dataset.

        Returns:
            PreTrainedTokenizerFast: The tokenizer instance.
        """
        return self._tokenizer

    @property
    def raw_dataset(self) -> Dataset:
        """
        Provides access to the raw loaded dataset.

        Returns:
            Dataset: The raw dataset.
        """
        raw_dataset: Dataset = self._dataset
        return raw_dataset

    @property
    def train(self) -> Dataset:
        """
        Retrieves the training split of the tokenized dataset.

        Returns:
            Dataset: The training dataset.
        """
        train_dataset: Dataset = self._tokenized_dataset["train"]
        return train_dataset

    @property
    def test(self) -> Dataset:
        """
        Retrieves the test split of the tokenized dataset.

        Returns:
            Dataset: The test dataset.
        """
        test_dataset: Dataset = self._tokenized_dataset["test"]
        return test_dataset

    def set_model(self, model_name: str) -> None:
        """
        Defines and initializes the tokenizer and tokenizes the dataset.

        Args:
            model_name (str): The name or path of the model to use for tokenization.
        """
        self._tokenizer = load_model_tokenizer(model_name=model_name)

        # tokenize the dataset
        self._tokenized_dataset = self._dataset.map(
            function=self._format_conversation, batched=True
        )

        # # remove any columns not needed for training (e.g., original text fields)
        # rm_columns: list[str] = ["conversations", "source"]
        # self._tokenized_dataset = self._tokenized_dataset.remove_columns(rm_columns)
        # # ensure the format is PyTorch-friendly
        # format_columns: list[str] = ["input_ids", "attention_mask"]
        # self._tokenized_dataset.set_format(type="torch", columns=format_columns)

        # Remove any columns not needed for training (e.g., original text fields)
        rm_columns: list[str] = [
            "conversations",
            "source",
            "instruction",
            "response",
            "output",
        ]
        for col in rm_columns:
            if col in self._tokenized_dataset.column_names:
                self._tokenized_dataset = self._tokenized_dataset.remove_columns(col)

        # Ensure the format is PyTorch-friendly
        format_columns: list[str] = ["input_ids", "attention_mask"]
        self._tokenized_dataset.set_format(type="torch", columns=format_columns)

        return None

    def cleanup_dataset(self) -> None:
        """
        Cleans up the dataset from memory and clears GPU cache.
        """
        if self._dataset is not None:
            del self._dataset
            self._dataset = None
        torch.cuda.empty_cache()
        gc.collect()
        return None

    @classmethod
    def merge(
        cls,
        dataset_paths: list[str],
        sample_proportions: Optional[list[float]] = None,
        shuffle: Optional[bool] = True,
        model_name: Optional[Union[AutoModelForCausalLM, PeftModel]] = None,
    ) -> "DatasetInterface":
        """
        Merges multiple datasets into a single DatasetInterface instance.

        Args:
            dataset_paths (list[str]):
                A list of dataset names or paths to load.
            sample_proportions (Optional[list[float]]):
                Proportions for sampling each dataset. If None, equal proportions are
                used.
            shuffle (Optional[bool]):
                Whether to shuffle the merged dataset. Defaultsto True.
            model_name (Optional[str]):
                The name or path of the model for tokenization.

        Returns:
            DatasetInterface: An instance containing the merged dataset.

        Raises:
            AssertionError:
                If `sample_proportions` contain values outside [0, 1] or if their length
                doesn't match `dataset_paths`.
        """

        # Ensure dataset_paths is not empty
        assert len(dataset_paths) > 0, "No dataset paths provided"

        # Set equal proportions if sample_proportions is None
        if sample_proportions is None:
            sample_proportions = [1.0 / len(dataset_paths)] * len(dataset_paths)

        # Normalize sample_proportions to sum to 1
        total_proportion: float = sum(sample_proportions)
        sample_proportions = [p / total_proportion for p in sample_proportions]

        # Ensure sample_proportions has the same length as dataset_paths
        assert len(sample_proportions) == len(
            dataset_paths
        ), "Length of sample_proportions must match length of dataset_paths"

        # Load and sample datasets
        datasets: list[Dataset] = []
        dataset: Dataset
        for path, proportion in zip(dataset_paths, sample_proportions):
            # # Load dataset
            # dataset: DatasetInterface = load_dataset(path)
            if path == "argilla/ifeval-like-data":
                dataset = load_dataset(path=path, split="train")
                # dataset = filter_datset(dataset)
            else:
                dataset = load_dataset(path=path)

            # Use 'train' split if available
            if isinstance(dataset, DatasetDict) and "train" in dataset:
                dataset = dataset["train"]
            elif isinstance(dataset, DatasetDict):
                # Merge all splits if 'train' is not available
                dataset = concatenate_datasets(
                    [dataset[split] for split in dataset.keys()]
                )
            # Sample the dataset
            num_samples: int = len(dataset)
            num_to_sample = int(num_samples * proportion)
            sampled_dataset: Dataset
            if proportion < 1.0:
                sampled_dataset = dataset.shuffle(seed=42).select(range(num_to_sample))
            else:
                sampled_dataset = dataset
            datasets.append(sampled_dataset)

        # Concatenate the sampled datasets
        merged_dataset: Dataset = concatenate_datasets(datasets)

        # Shuffle the merged dataset if required
        if shuffle:
            merged_dataset = merged_dataset.shuffle(seed=42)

        # Initialize a new DatasetInterface instance with the merged dataset
        dataset_interface: "DatasetInterface"
        dataset_interface = cls(dataset_name="merge", model_name=model_name)
        dataset_interface._dataset = merged_dataset
        dataset_interface.dataset_split()

        return dataset_interface

    def dataset_split(
        self,
        test_proportion: Optional[float] = None,
        test_samples: Optional[int] = None,
        seed: Optional[int] = 42,
    ) -> DatasetDict:

        assert test_proportion is None or test_samples is None

        train_samples: int
        new_dataset: DatasetDict
        if isinstance(self._dataset, Dataset):
            train_samples = len(self._dataset)
            new_dataset = DatasetDict({"train": self._dataset})
        if isinstance(self._dataset, DatasetDict):
            assert "train" in self._dataset.keys(), "no train split in dataset"
            if "test" in self._dataset.keys():
                return None
            train_samples = len(self._dataset["train"])
            new_dataset = self._dataset

        _test_proportion: float
        if test_proportion is None and test_samples is None:
            _test_proportion = min(500, train_samples // 4) / train_samples

        elif test_proportion is None:
            assert test_samples <= train_samples
            _test_proportion = test_samples / train_samples

        elif test_samples is None:
            assert test_proportion >= 0 and test_proportion <= 1
            _test_proportion = test_proportion

        self._dataset = new_dataset["train"].train_test_split(
            test_size=_test_proportion, seed=seed
        )
        print(f"Split {_test_proportion*100:.4f}% of 'train' into 'test'.")

        return None

    def _format_conversation(
        self, examples: dict[str, Union[list[str], list[list[str]]]]
    ) -> BatchEncoding:
        """
        Formats and tokenizes conversation data.

        Args:
            examples (Dict[str, Union[List[str], List[List[str]]]]):
                A dictionary where the key 'conversations' maps to a list of
                conversations. Each conversation can be a single string or a list of
                sentences.

        Returns:
            BatchEncoding:
            The tokenized conversations with input_ids and attention_mask.
        """
        joined_conversations: list[str]
        # lima
        if all(substr in examples.keys() for substr in ["conversations"]):
            joined_conversations = [
                " ".join(conv) if isinstance(conv, list) else conv
                for conv in examples["conversations"]
            ]

        # dolly, ifeval-like
        if all(substr in examples.keys() for substr in ["instruction", "response"]):
            joined_conversations = [
                f"{pair[0]} {pair[1]}"
                for pair in zip(examples["instruction"], examples["response"])
            ]

        # alpaca
        if all(substr in examples.keys() for substr in ["instruction", "output"]):
            joined_conversations = [
                f"{pair[0]} {pair[1]}"
                for pair in zip(examples["instruction"], examples["output"])
            ]

        # Tokenize the joined conversations
        tokenization: BatchEncoding = self._tokenizer(
            joined_conversations,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt",
        )

        return tokenization


class ModelInterface:
    """
    A class to interface with machine learning models, handling loading, PEFT
    configurations, dataset integration, training, and cleanup.
    """

    def __init__(self) -> None:
        """
        Initializes the ModelInterface with default configurations for model
        quantization and sets up initial attributes.
        """

        self._bnb_config: BitsAndBytesConfig = BitsAndBytesConfig(
            load_in_4bit=True,  # Enable loading the model in 4-bit precision
            bnb_4bit_quant_type="nf4",  # Specify quantization type as Normal Float 4
            bnb_4bit_compute_dtype=compute_dtype,  # Set computation data type
            bnb_4bit_use_double_quant=True,  # Double quantization for better accuracy
            # load_in_8bit_fp32_cpu_offload=True,  # Enable CPU offloading for FP32
        )

        self._name: str
        self._model: Optional[Union[AutoModelForCausalLM, PeftModel]] = None
        self._peft_config: Optional[PeftConfig] = None
        self._dataset: Optional[DatasetInterface] = None

        return None

    @property
    def name(self) -> str:
        """
        Returns the name of the loaded model.

        Returns:
            str: The name of the model.
        """
        return self._name

    @property
    def model(self) -> Union[AutoModelForCausalLM, PeftModel]:
        """
        Provides access to the loaded model.

        Returns:
            Union[AutoModelForCausalLM, PeftModel]: The loaded model instance.
        """
        return self._model

    def load_model(self, name: str) -> None:
        """
        Loads a pre-trained causal language model with quantization based on the
        provided name or path.

        Args:
            name (str): The name or path of the pre-trained model to load.

        Raises:
            ValueError: If the model cannot be loaded.
        """
        device_map: str = "auto"  # "balanced_low_0"  # auto
        self._model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=name,
            quantization_config=self._bnb_config,  # Apply quantization configuration
            device_map=device_map,  # Automatically map layers to devices
        )
        self._name = name
        self._model = prepare_model_for_kbit_training(self._model)
        self._model.to(device=device)
        return None

    def load_PEFT_config(self, config: PeftConfig) -> None:
        """
        Loads a PEFT (Parameter-Efficient Fine-Tuning) configuration.

        Args:
            config (PeftConfig): The PEFT configuration to load.

        Raises:
            AssertionError: If the provided config is not an instance of PeftConfig.
        """
        assert isinstance(config, PeftConfig)
        self._peft_config = config
        return None

    def load_dataset(self, interface: DatasetInterface) -> None:
        """
        Integrates a DatasetInterface instance with the model and sets the model's
        padding token ID.

        Args:
            interface (DatasetInterface): The dataset interface to load.

        Raises:
            AssertionError:
                If the provided interface is not an instance of DatasetInterface.
            AttributeError:
                If the model has not been loaded prior to setting the dataset.
        """
        assert isinstance(interface, DatasetInterface)
        self._dataset = interface
        # Set the model's padding token ID
        self._model.config.pad_token_id = self._dataset.tokenizer.pad_token_id
        return None

    def train(
        self,
        method: Type[Trainer],
        arguments: Type[TrainingArguments],
    ) -> None:
        """
        Trains the model using the specified Trainer class and training arguments.

        Args:
            method (Type[Trainer]): The Trainer class to use for training.
            arguments (Type[TrainingArguments]): The training arguments.

        Raises:
            AssertionError:
                If the provided method or arguments are not of the correct type, or if
                the model, PEFT config, or dataset are not properly loaded.
        """
        assert isinstance(method, type) and issubclass(method, Trainer)
        assert issubclass(type(self._model), (PreTrainedModel, PeftModel)), type(
            self._model
        )
        assert issubclass(type(self._peft_config), PeftConfig)
        assert isinstance(self._dataset, DatasetInterface)
        assert issubclass(type(arguments), TrainingArguments)

        trainer: Trainer = method(
            model=self._model,
            train_dataset=self._dataset.train,
            eval_dataset=self._dataset.test,
            peft_config=self._peft_config,
            tokenizer=self._dataset.tokenizer,
            args=arguments,
        )
        trainer.train()

        del trainer
        gc.collect()

        return None

    def cleanup_model(self) -> None:
        """
        Cleans up the model from GPU memory and clears any offloaded files.
        """
        if self._model is not None:
            # Ensure all CUDA operations are finished
            torch.cuda.synchronize()
            del self._model
            self._model = None
            torch.cuda.empty_cache()
            gc.collect()

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str) -> "ModelInterface":
        """
        Class method to load the model and its tokenizer from a saved checkpoint.

        Args:
            checkpoint_path (str): The path to the checkpoint directory.

        Returns:
            ModelInterface: An instance of ModelInterface with the model loaded.

        Raises:
            ValueError: If the provided checkpoint path is not a valid directory.
            Exception: If loading the model or tokenizer fails.
        """

        # Create a new instance
        torch.cuda.empty_cache()
        instance: ModelInterface = cls()

        # Validate checkpoint path
        if not os.path.isdir(checkpoint_path):
            raise ValueError(
                f"Checkpoint path '{checkpoint_path}' is not a valid directory."
            )

        # Load the PEFT config to get the base model name
        peft_config: PeftConfig = PeftConfig.from_pretrained(checkpoint_path)
        base_model_name: str = peft_config.base_model_name_or_path
        assert isinstance(base_model_name, str)

        # Load the base model with quantization configurations
        model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            offload_folder="offload_dir",
            quantization_config=instance._bnb_config,  # Use the quantization config
        )

        # Prepare the model for k-bit training
        model = prepare_model_for_kbit_training(model)

        # Load the LoRA adapter weights
        model = PeftModel.from_pretrained(model, checkpoint_path)

        # to device
        model = model.to(device)

        # Assign model and tokenizer to the instance
        instance._model = model
        instance._name = base_model_name
        instance._tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        return instance
