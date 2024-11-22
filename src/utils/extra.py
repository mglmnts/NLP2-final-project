# Standard Library dependencies
import regex as re


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
