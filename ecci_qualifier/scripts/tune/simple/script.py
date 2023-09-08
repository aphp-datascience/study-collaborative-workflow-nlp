import os
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
import typer
from tuner import Tuner

from edsml.modules.helpers.misc import set_proxy

app = typer.Typer(pretty_exceptions_show_locals=False)

tune_params = [
    {"classification_head.@classification_head": ["simple"]},
    {
        "optimizer_params.transformer.lr": [6e-5, 4e-5, 2e-5],
    },
    {
        "optimizer_params.head.lr": [1e-3, 4e-4, 7e-4, 1e-4],
    },
]


def experience_naming(modifs):
    shortnames = {
        "classification_head.@classification_head": "head",
        "optimizer_params.transformer.lr": "t_lr",
        "optimizer_params.head.lr": "h_lr",
    }

    shortname = "-".join(
        [f"{shortname}:{modifs[name]}" for name, shortname in shortnames.items()]
    )

    return shortname


@app.command()
def tune(
    base_conf: Path,
    name: Optional[str] = None,
    save_n_max: int = 5,
    seed: int = 42,
):
    torch.cuda.empty_cache()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    set_proxy()

    name = name if name is not None else base_conf.stem

    tuner = Tuner(
        tune_params=tune_params,
        base_conf=base_conf,
        name=name,
        experience_naming=experience_naming,
        save_n_max=save_n_max,
    )

    for experience_name, modifs, conf in tuner.iterate_over_tuned_config(resolve=False):
        pl.seed_everything(conf["script"]["seed"])

        dirpath = conf["checkpoint"]["dirpath"]
        monitor = conf["checkpoint"]["monitor"]
        monitor_slug = "".join(x if x.isalnum() else "_" for x in monitor)

        conf["checkpoint"]["filename"] = (
            f"{experience_name}" f"-{monitor_slug}:" "{" f"{monitor}" ":.4f}"
        )
        conf["logger"]["name"] = experience_name

        conf.to_disk(Path(dirpath) / f"{experience_name}.cfg")

        conf_resolved = conf.resolve()

        data = conf_resolved["data"]
        model = conf_resolved["model"]
        trainer = conf_resolved["trainer"]

        for k, v in model.classification_head.state_dict().items():
            print(k, v.sum())

        trainer.fit(model, datamodule=data)

        best_score = trainer.checkpoint_callback.best_model_score.cpu().item()

        tuner.ckpt_handler(trainer, conf)

        trainer.logger.log_hyperparams(modifs, {monitor: best_score})


if __name__ == "__main__":
    app()
