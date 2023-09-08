import warnings

from .utils import annotation_setters, span_getters

# Remove warnings for inference via SpaCy

warnings.filterwarnings("ignore", module="lightning_lite")
warnings.filterwarnings("ignore", module="pytorch_lightning")
