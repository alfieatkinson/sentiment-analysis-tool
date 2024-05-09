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
    return 'v0.2'

def time() -> str:
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def year() -> str:
    return str(datetime.datetime.now().year)

def platform() -> str:
    return platform.system() + " " + platform.release()

def path() -> list[str]:
    return sys.path

def get_dir() -> str:
    return os.path.dirname(os.path.realpath(__file__)) + '/../'

load_dotenv()

print("Initialising...")