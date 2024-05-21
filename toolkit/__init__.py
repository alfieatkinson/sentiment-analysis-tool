import datetime
import platform
import sys
import os

from dotenv import load_dotenv

from .analysis import *
from .data import *
from .ui import *
from .messages import *
from .config import *

def version() -> str:
    """
    Returns the version of the application.

    Returns:
        str: The version of the application.
    """
    return 'v1.1'

def time() -> str:
    """
    Returns the current date and time in the format 'YYYY-MM-DD HH:MM:SS'.

    Returns:
        str: Current date and time.
    """
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def year() -> str:
    """
    Returns the current year as a string.

    Returns:
        str: Current year.
    """
    return str(datetime.now().year)

def platform() -> str:
    """
    Returns the name of the operating system and its release version.

    Returns:
        str: Name of the operating system and its release version.
    """
    return platform.system() + " " + platform.release()

def path() -> list[str]:
    """
    Returns a list of strings representing the search path for modules.

    Returns:
        list[str]: List of strings representing the search path for modules.
    """
    return sys.path

def get_dir() -> str:
    """
    Returns the directory of the current file's parent directory.

    Returns:
        str: Directory path of the current file's parent directory.
    """
    return os.path.dirname(os.path.realpath(__file__)) + '/../'


load_dotenv()

print("Initialising...")