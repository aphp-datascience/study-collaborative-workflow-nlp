import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from .registry import registry


@registry.callback("PytorchLightningCallback")
def BaseCallback(name=None, **kwargs) -> Callback:
    """
    Returns a PyTorch Lightning existing callback

    Parameters
    ----------
    name : str
        The name of the callback

    Returns
    -------
    Callback
    """
    return getattr(pl.callbacks, name)(**kwargs)
