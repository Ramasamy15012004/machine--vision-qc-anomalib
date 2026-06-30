# mvs_sdk_path.py
import sys

# Centralized location for the MVS SDK python import path
MVS_PY_PATH = r"C:/Program Files (x86)/MVS/Development/Samples/Python/MvImport"

def add_sdk_to_path():
    """Dynamically adds the MVS SDK Python wrapper directory to the system path."""
    if MVS_PY_PATH not in sys.path:
        sys.path.append(MVS_PY_PATH)
