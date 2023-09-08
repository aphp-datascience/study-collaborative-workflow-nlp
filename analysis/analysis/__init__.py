import os
from pathlib import Path

CSE = os.environ.get("CSE", "")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / "data"
EXPORT_DIR = None
# DATA_DIR = BASE_DIR.parent / CSE / "data"
# EXPORT_DIR = BASE_DIR.parent / CSE / "export"
