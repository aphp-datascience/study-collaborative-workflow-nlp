import os
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
import typer
from tuner import Tuner

from ecci_qualifier import BASE_DIR
from edsml.modules.helpers.misc import set_proxy

app = typer.Typer(pretty_exceptions_show_locals=False)

CSE = ["cse180032", "cse200055", "cse200093"]
tune_params = [
    {
        "script.name": CSE,
        "dataset_loader.data": [
            f"/export/home/share/datascientists/ecci/final/{cse}_entities.csv"
            for cse in CSE
        ],
        "dataset_loader.save_path": [
            f"/data/scratch/tpetitjean/ML/datasets/ecci_{cse}" for cse in CSE
        ],
    },
]


def experience_naming(modifs):
    shortnames = {
        "script.name": "cse",
    }

    shortname = "-".join(
        [f"{shortname}:{modifs[name]}" for name, shortname in shortnames.items()]
    )

    return shortname


@app.command()
def tune(
    model: Optional[str] = None,
    name: Optional[str] = None,
    save_n_max: int = 5,
):
    assert model in {"camembert-base", "camembert-eds"}

    if model == "camembert-base":
        base_conf = (
            BASE_DIR.parent / "scripts" / "configs" / "training_camembert_base.cfg"
        )

    elif model == "camembert-eds":
        base_conf = (
            BASE_DIR.parent / "scripts" / "configs" / "training_camembert_eds.cfg"
        )

    print("CONF: ", base_conf)

    model_type = model.split("-")[-1]

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
            f"{model_type}/{experience_name}"
            f"-{monitor_slug}:"
            "{"
            f"{monitor}"
            ":.4f}"
        )

        conf.to_disk(Path(dirpath) / model_type / f"{experience_name}.cfg")

        conf_resolved = conf.resolve()

        data = conf_resolved["data"]
        model = conf_resolved["model"]
        trainer = conf_resolved["trainer"]

        trainer.fit(model, datamodule=data)

        best_score = trainer.checkpoint_callback.best_model_score.cpu().item()

        tuner.ckpt_handler(trainer, conf)

        trainer.logger.log_hyperparams(modifs, {monitor: best_score})


if __name__ == "__main__":
    app()
