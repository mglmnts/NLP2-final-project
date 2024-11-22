# Standard Library dependencies
import os
from typing import Optional, Type, Union

# ML dependencies
import torch
from datasets import Dataset, DatasetDict, load_dataset
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

# from evaluate.module import EvaluationModule
from trl import SFTConfig  # PPOTrainer, PPOv2Trainer, DPOTrainer, SFTTrainer

# from evaluate import load


# Custom Types
PEFTType = Union[LoraConfig]

# Global Variables
device: str = "cuda" if torch.cuda.is_available() else "cpu"
compute_dtype = getattr(torch, "bfloat16")  # Set computation data type to bfloat16


class DatasetInterface:

    def __init__(
        self, dataset_name: str, model_name: Union[AutoModelForCausalLM, PeftModel]
    ) -> None:

        self._dataset: Union[Dataset, DatasetDict]
        self._tokenizer: PreTrainedTokenizerFast
        self._tokenized_dataset: Union[Dataset, DatasetDict]

        self._dataset = load_dataset(path=dataset_name)
        self._tokenizer = self._load_model_tokenizer(model_name=model_name)

        # tokenize the dataset
        self._tokenized_dataset = self._dataset.map(
            function=self._format_conversation, batched=True
        )
        # remove any columns not needed for training (e.g., original text fields)
        rm_columns: list[str] = ["conversations", "source"]
        self._tokenized_dataset = self._tokenized_dataset.remove_columns(rm_columns)
        # ensure the format is PyTorch-friendly
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
        # ???
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
        # Join the list into a single string if it's a list of sentences
        joined_conversations: list[str] = [
            " ".join(conv) if isinstance(conv, list) else conv
            for conv in examples["conversations"]
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
    def __init__(self) -> None:

        self._bnb_config: BitsAndBytesConfig = BitsAndBytesConfig(
            load_in_4bit=True,  # Enable loading the model in 4-bit precision
            bnb_4bit_quant_type="nf4",  # Specify quantization type as Normal Float 4
            bnb_4bit_compute_dtype=compute_dtype,  # Set computation data type
            bnb_4bit_use_double_quant=True,  # Double quantization for better accuracy
            load_in_8bit_fp32_cpu_offload=True,  # Enable CPU offloading for FP32
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
        device_map: str = "balanced_low_0"  # auto
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

    def from_checkpoint(self, cls, checkpoint_path: str) -> "ModelInterface":
        """
        Class method to load the model and its configuration from a saved checkpoint.

        Args:
            checkpoint_path (str): The path to the checkpoint directory.

        Returns:
            ModelInterface: An instance of ModelInterface with the model loaded.
        """

        # Create a new instance
        instance: ModelInterface = cls()

        # Validate checkpoint path
        if not os.path.isdir(checkpoint_path):
            raise ValueError(
                f"Checkpoint path '{checkpoint_path}' is not a valid directory."
            )

        # Load model configuration to retrieve model name
        try:
            config: AutoConfig = AutoConfig.from_pretrained(checkpoint_path)
            model_name: str = config._name_or_path
            instance._name = model_name
        except Exception as e:
            raise ValueError(
                f"Failed to load configuration from '{checkpoint_path}': {e}"
            )

        # Load the base model
        try:
            base_model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                quantization_config=instance._bnb_config,  # quantization configuration
                device_map="auto",  # Automatically map layers to devices
            )
            instance._model = base_model
        except Exception as e:
            raise ValueError(f"Failed to load model from '{checkpoint_path}': {e}")

        # Check for PEFT configuration
        peft_config_path: str = os.path.join(checkpoint_path, "adapter_config.json")
        if os.path.isfile(peft_config_path):
            try:
                # Load the PEFT configuration
                peft_config: PeftConfig = PeftConfig.from_pretrained(checkpoint_path)
                instance._peft_config = peft_config

                # Wrap the base model with PeftModel
                peft_model: PeftModel = PeftModel.from_pretrained(
                    base_model,
                    checkpoint_path,
                )
                instance._model = peft_model
            except Exception as e:
                raise ValueError(
                    f"Failed to load PEFT configuration from '{peft_config_path}': {e}"
                )

        return instance
