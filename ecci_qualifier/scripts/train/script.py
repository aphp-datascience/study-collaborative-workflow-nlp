import os

import pytorch_lightning as pl
import torch
from confit import Cli

from edsml.modules.data import BaseDataModule
from edsml.modules.helpers.misc import flatten_dict, set_proxy

app = Cli(pretty_exceptions_show_locals=False)


@app.command(name="script")
def main(
    name: str,
    model: pl.LightningModule,
    trainer: pl.Trainer,
    data: BaseDataModule,
    seed: int,
    config_meta=None,
):
    torch.cuda.empty_cache()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    set_proxy()

    for k, v in model.classification_head.state_dict().items():
        print(k, v.sum())

    trainer.logger.log_hyperparams(flatten_dict(config_meta["unresolved_config"]))

    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    pl.seed_everything(42)
    app()
