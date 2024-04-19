import datetime
import platform
import sys
import os

from .analysis import *
from .data import *
from .ui import *

def version():
    return "v0.1"

def time():
    return datetime.datetime.now().strftime("%Y-%m-%d @ %H:%M:%S.%f")

def year():
    return str(datetime.datetime.now().year)

def platform():
    return platform.system() + " " + platform.release()

def path():
    return sys.path

def get_dir():
    return os.path.dirname(os.path.realpath(__file__)) + "/../"

print("Initialising...")