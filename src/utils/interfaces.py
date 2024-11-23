# Standard Library dependencies
import os
import gc
from typing import Optional, Type, Union

# ML dependencies
import torch
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from peft.config import PeftConfig
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


# from evaluate import load


# Custom Types
PEFTType = Union[LoraConfig]

# Global Variables
device: str = "cuda" if torch.cuda.is_available() else "cpu"
compute_dtype = getattr(torch, "bfloat16")  # Set computation data type to bfloat16
# compute_dtype = getattr(torch, "float16")  # Set computation data type to bfloat16


class DatasetInterface:

    def __init__(
        self, dataset_name: str, model_name: Union[AutoModelForCausalLM, PeftModel]
    ) -> None:

        self._dataset_name: str = dataset_name

        self._dataset: Union[Dataset, DatasetDict]
        self._tokenizer: PreTrainedTokenizerFast
        self._tokenized_dataset: Union[Dataset, DatasetDict]

        self._dataset = load_dataset(path=dataset_name)
        self._tokenizer = self._load_model_tokenizer(model_name=model_name)

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

    @property
    def tokenizer(self) -> PreTrainedTokenizerFast:
        return self._tokenizer

    @property
    def train(self) -> Dataset:
        train_dataset: Dataset = self._tokenized_dataset["train"]
        return train_dataset

    @property
    def test(self) -> Dataset:
        test_dataset: Dataset = self._tokenized_dataset["test"]
        return test_dataset

    def _load_model_tokenizer(self, model_name: str) -> PreTrainedTokenizerFast:
        # Load the specific tokenizer for the specified model
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            add_eos_token=True,  # Add end-of-sequence token to the tokenizer
            use_fast=True,  # Use the fast tokenizer implementation
            padding_side="left",  # Pad sequences on the left side
        )
        assert isinstance(tokenizer, PreTrainedTokenizerFast)
        tokenizer.pad_token = tokenizer.eos_token  # Set padding token to EOS token

        return tokenizer

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
        if all(substr in self._dataset_name.lower() for substr in ["lima"]):
            joined_conversations = [
                " ".join(conv) if isinstance(conv, list) else conv
                for conv in examples["conversations"]
            ]
        if all(substr in self._dataset_name.lower() for substr in ["dolly"]):

            joined_conversations = [
                f"{pair[0]} {pair[1]}"
                for pair in zip(examples["instruction"], examples["response"])
            ]

        if all(substr in self._dataset_name.lower() for substr in ["alpaca"]):

            joined_conversations = [
                f"{pair[0]} {pair[1]}"
                for pair in zip(examples["instruction"], examples["output"])
            ]

        if all(substr in self._dataset_name.lower() for substr in ["ifeval", "like"]):

            joined_conversations = [
                f"{pair[0]} {pair[1]}"
                for pair in zip(examples["instruction"], examples["response"])
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

    def cleanup_dataset(self) -> None:
        if self._dataset is not None:
            del self._dataset
            self._dataset = None
        torch.cuda.empty_cache()
        gc.collect()


class ModelInterface:
    def __init__(self) -> None:

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
        # self._trainer: Optional[TrainerType] = None

        return None

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> Union[AutoModelForCausalLM, PeftModel]:
        return self._model

    def load_model(self, name: str) -> None:
        device_map: str = "auto"  # "balanced_low_0"  # auto
        self._model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=name,
            quantization_config=self._bnb_config,  # Apply quantization configuration
            device_map=device_map,  # Automatically map layers to devices
        )
        self._name = name
        self._model = prepare_model_for_kbit_training(self._model)
        return None

    def load_PEFT_config(self, config: PeftConfig) -> None:
        assert isinstance(config, PeftConfig)
        self._peft_config = config
        return None

    def load_dataset(self, interface: DatasetInterface) -> None:
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
        assert isinstance(method, type) and issubclass(method, Trainer)
        assert issubclass(type(self._model), (PreTrainedModel, PeftModel))
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

        return None

    # def cleanup_model(self) -> None:
    #     """
    #     Cleans up the model from GPU memory.
    #     """
    #     if self._model is not None:
    #         # Move the model to CPU
    #         self._model.to("cpu")
    #         # Delete the model object
    #         del self._model
    #         self._model = None
    #         # Clear GPU cache
    #         torch.cuda.empty_cache()
    #         # Run garbage collection
    #         gc.collect()

    def cleanup_model(self) -> None:
        """
        Cleans up the model from GPU memory.
        """
        if self._model is not None:
            # Elimina el modelo sin moverlo manualmente
            del self._model
            self._model = None
            # Limpia la caché de GPU
            torch.cuda.empty_cache()
            # Ejecuta la recolección de basura
            gc.collect()

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str) -> "ModelInterface":
        """
        Class method to load the model and its configuration from a saved checkpoint.

        Args:
            checkpoint_path (str): The path to the checkpoint directory.

        Returns:
            ModelInterface: An instance of ModelInterface with the model loaded.
        """

        # Create a new instance
        torch.cuda.empty_cache()
        instance: ModelInterface = cls()

        # Validate checkpoint path
        if not os.path.isdir(checkpoint_path):
            raise ValueError(
                f"Checkpoint path '{checkpoint_path}' is not a valid directory."
            )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            device_map="auto",
            offload_folder="offload_dir",
        )
        instance._model = model
        instance._name = model.config.model_type
        instance._tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

        return instance
