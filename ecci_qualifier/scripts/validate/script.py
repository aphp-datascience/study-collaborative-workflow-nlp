import os
from pathlib import Path
from typing import Optional, Union

import pytorch_lightning as pl
import torch
from confit import Cli

from edsml.modules.data import BaseDataModule
from edsml.modules.helpers.checkpoints import ckpt_from_config
from edsml.modules.helpers.misc import set_proxy

app = Cli(pretty_exceptions_show_locals=False)


@app.command(name="script")
def main(
    name: str,
    model: pl.LightningModule,
    trainer: pl.Trainer,
    data: BaseDataModule,
    seed: int,
    ckpt_version: Optional[Union[Path, str]] = "last",
):
    torch.cuda.empty_cache()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    set_proxy()
    pl.seed_everything(seed)

    ckpt_dir = trainer.checkpoint_callback.dirpath
    ckpt_path = ckpt_from_config(
        config_name=name,
        ckpt_dir=ckpt_dir,
        version=ckpt_version,
    )
    trainer.validate(model, datamodule=data, ckpt_path=ckpt_path)


if __name__ == "__main__":
    app()
