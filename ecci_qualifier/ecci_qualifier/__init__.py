from pathlib import Path
from typing import Optional

from confit.config import Config

from edsml.modules import registry

BASE_DIR = Path(__file__).parent

AVAILABLE_CONFIGS = {
    "inference",
    "inference_camembert_eds",
    "training",
    "training_camembert_eds",
}


def get_config(which: Optional[str] = None):
    if which not in AVAILABLE_CONFIGS:
        raise ValueError(f"The `which` parameter should be in {AVAILABLE_CONFIGS}")

    return Config.from_disk(BASE_DIR / "data" / f"{which}.cfg")
