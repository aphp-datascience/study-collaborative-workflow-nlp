from typing import Any, Dict, List, Optional

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback

from .registry import registry


@registry.trainer("PytorchLightningTrainer")
def PytorchLightningTrainer(
    cpu: Dict[str, Any] = dict(),
    gpu: Dict[str, Any] = dict(),
    auto_lr_find: bool = False,
    auto_scale_batch_size: bool = False,
    log_every_n_steps: int = 10,
    max_epochs: Optional[int] = None,
    max_steps: Optional[int] = None,
    batch_size: Optional[int] = None,
    n_rows: Optional[int] = None,
    callbacks: List[Callback] = [],
    logger: Any = None,
):
    """
    Trainer instance to handle training and inference

    Parameters
    ----------
    cpu : Dict[str, Any], optional
        Parameters used when training/infering on CPU, by default dict()
    gpu : Dict[str, Any], optional
        Parameters used when training/infering on GPU, by default dict()

    Following params are given directly to the Trainer.__init__

    auto_lr_find : bool, optional
    auto_scale_batch_size : bool, optional
    log_every_n_steps : int, optional
    max_epochs : Optional[int], optional
    max_steps : Optional[int], optional
    batch_size : Optional[int], optional
    n_rows : Optional[int], optional
    callbacks : List[Callback], optional
    logger : Any, optional

    """

    assert not (
        max_epochs and max_steps
    ), "Both `max_epochs` and ``max_steps`were provided to the Trainer."

    if max_epochs:
        max_steps = int(max_epochs * (n_rows / batch_size))

    trainer_params = dict(
        max_steps=max_steps,
        auto_lr_find=auto_lr_find,
        auto_scale_batch_size=auto_scale_batch_size,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=log_every_n_steps,
    )

    hardware_params = gpu if torch.cuda.device_count() > 0 else cpu
    trainer_params.update(hardware_params)

    trainer = Trainer(**trainer_params)

    return trainer
