# Standard Library dependencies
import os
import regex as re
from pathlib import Path


def clean_string(string: str) -> str:
    """
    Converts the input string to lowercase and removes all characters
    except for letters (a-z) and hyphens (-).

    Args:
        string (str): The original string to be cleaned.

    Returns:
        str: The cleaned string containing only lowercase letters and hyphens.
    """
    # Convert the string to lowercase
    lowered: str = string.lower()

    # Use regex to remove characters that are not letters or hyphens
    clean_str: str = re.sub(r"[^a-z\-]", "", lowered)

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


def locate_data_path(section: str, dir_name: str) -> str:
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
    rel_path: str = f"data/{section}/{dir_name}"
    dir_path: str = os.path.join(src_path.parent, rel_path)
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    return dir_path
