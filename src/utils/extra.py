# Standard Library dependencies
import os
import regex as re
from pathlib import Path
from typing import Optional, Union

# ML dependencies
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


def load_model_tokenizer(model_name: str) -> PreTrainedTokenizerFast:
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


def clean_string(string: str) -> str:
    """
    Converts the input string to lowercase and removes all characters
    except for lowercase letters (a-z), digits (0-9), and hyphens (-).

    Args:
        string (str): The original string to be cleaned.

    Returns:
        str: The cleaned string containing only lowercase letters, digits, and hyphens.
    """
    # Convert the string to lowercase
    lowered: str = string.lower()

    # Use regex to remove characters that are not lowercase letters, digits, or hyphens
    clean_str: str = re.sub(r"[^a-z0-9\-]", "", lowered)

    return clean_str


def get_src_path() -> Path:
    """
    Traverses up the directory tree from the current file's location
    to find the 'src' directory.

    Returns:
        Path: The absolute path to the 'src' directory.

    Raises:
        FileNotFoundError: If the 'src' directory is not found in the hierarchy.
    """
    current_file: Path = Path(__file__).resolve()
    for parent in current_file.parents:
        if parent.name == "src":
            return parent
    raise FileNotFoundError("The 'src' directory was not found in the path hierarchy.")


def locate_data_path(rel_path: str) -> str:
    """
    Locates the data path for the specified directory name within the
    'data/explore-models' structure relative to the 'src' directory.
    Creates the directory if it does not already exist.

    Args:
        dir_name (str): The name of the directory to locate or create.

    Returns:
        str: The absolute path to the specified directory.

    Raises:
        FileNotFoundError: If the 'src' directory cannot be located.
    """
    src_path: str = get_src_path()
    dir_path: str = os.path.join(src_path.parent, "data", rel_path)
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    return dir_path


from datasets import Dataset


def get_dataset_subset(
    dataset: Dataset, prop: float, shuffle: Optional[bool] = True
) -> Dataset:
    """
    Returns a random subset of the IFEval dataset.

    Args:
        dataset (Dataset): The original IFEval dataset.
        prop (float): The proportion of the dataset to include in the subset (between 0 and 1].
        shuffle (bool, optional): Whether to shuffle the dataset before sampling. Defaults to True.

    Returns:
        Dataset: A subset of the IFEval dataset containing the specified proportion of data.
    """
    assert 0 < prop <= 1, "Proportion must be between 0 and 1."

    # Optionally shuffle the dataset
    if shuffle:
        dataset = dataset.shuffle(seed=42)

    # Calculate the number of samples to select
    num_samples = int(len(dataset) * prop)

    # Select the subset
    subset = dataset.select(range(num_samples))

    return subset
