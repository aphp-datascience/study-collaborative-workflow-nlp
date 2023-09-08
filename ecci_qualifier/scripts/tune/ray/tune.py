import os
from pathlib import Path

import torch
from edstoolbox import SlurmApp
from helpers.dataset import QualifierDataModule
from helpers.modules import QualifierModelNoisyLabels
from helpers.utils import PathParse, set_proxy
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from ray import air, tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler
from transformers import CamembertTokenizerFast

app = SlurmApp()


def train(tune, not_tune, name):
    # FIXME union of 2 dicts !
    conf = {**tune, **not_tune}

    if "loss_name" in conf:
        alpha = conf.pop("alpha")
        beta = conf.pop("beta")
        loss_name = conf.pop("loss_name")
        conf["loss"] = dict(alpha=alpha, beta=beta, loss_name=loss_name)

    # print(conf)
    path_parser = PathParse(scratch_as_home=conf["scratch_as_home"])
    hf_datasets_cache = path_parser.expand(conf["hf_datasets_cache"])
    dataset_path = path_parser.expand(conf["dataset_path"])
    model_path = path_parser.expand(conf["model_path"])

    torch.cuda.empty_cache()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_DATASETS_CACHE"] = str(hf_datasets_cache)

    set_proxy()
    data = QualifierDataModule(
        dataset_path,
        batch_size=conf["batch_size"],
        max_seq_length=128,
        tokenizer=CamembertTokenizerFast.from_pretrained(
            model_path,
        ),
    )

    model = QualifierModelNoisyLabels(
        model_path,
        dropout=conf["dropout"],
        bert_lr=float(conf["bert_lr"]),
        head_lr=float(conf["head_lr"]),
        batch_size=conf["batch_size"],
        num_epoch=conf["n_epochs"],
        warmup_rate=conf["warmup_rate"],
        num_classes=conf["num_classes"],
        classification_head=conf["classification_head"],
        loss=conf["loss"],
    )

    # Ray
    ray_callback = TuneReportCallback(
        metrics={
            "loss": "valid/loss",
            "f1": "valid/f1",
            "precision": "valid/precision",
        },
        on="validation_end",
    )

    # Initialize the logger
    logger = TensorBoardLogger(
        save_dir=Path.home() / "tensorboard_data",
        name=name,
        default_hp_metric=False,
        version=".",
    )

    # Trainer
    trainer = Trainer(
        gpus=1,
        auto_select_gpus=True,
        strategy="dp",
        max_epochs=conf["n_epochs"],
        auto_lr_find=False,
        auto_scale_batch_size=False,
        callbacks=[
            ray_callback,
        ],
        logger=logger,
        log_every_n_steps=10,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    trainer.fit(model, datamodule=data)


@app.submit
def main(config):
    seed_everything(config["tune_conf"]["seed"])
    if (config["tune_conf"]["name"] == "tune_LR") or (
        config["tune_conf"]["name"] == "DEBUG"
    ):
        config_tune = dict(
            bert_lr=tune.choice(config["tune"]["bert_lr"]),
            head_lr=tune.choice(config["tune"]["head_lr"]),
        )
    else:
        config_tune = dict(
            alpha=tune.choice(config["tune"]["loss"]["alpha"]),
            beta=tune.choice(config["tune"]["loss"]["beta"]),
            loss_name=tune.choice(config["tune"]["loss"]["loss_name"]),
        )

    scheduler = ASHAScheduler(
        max_t=config["not_tune"]["n_epochs"],  # time_attr = 'time_total_s'
        grace_period=config["tune_conf"]["grace_period"],
        reduction_factor=2,
        metric=config["tune_conf"]["metric"],
        mode=config["tune_conf"]["mode"],
    )

    reporter = CLIReporter(
        parameter_columns=list(config_tune.keys()),
        metric_columns=["loss", "f1", "training_iteration", "precision"],
    )

    resources_per_trial = {"cpu": 2, "gpu": 1}

    train_fn_with_parameters = tune.with_parameters(
        train,
        not_tune=config["not_tune"],
        name=config["tune_conf"]["name"],
    )

    tuner = tune.Tuner(
        tune.with_resources(train_fn_with_parameters, resources=resources_per_trial),
        tune_config=tune.TuneConfig(
            # metric=config["tune_conf"]["metric"],
            # mode=config["tune_conf"]["mode"],
            scheduler=scheduler,
            num_samples=config["tune_conf"]["num_samples"],
            max_concurrent_trials=4,
            reuse_actors=False,
        ),
        run_config=air.RunConfig(
            name=config["tune_conf"]["name"],
            progress_reporter=reporter,
            # local_dir = Path.home() / "tensorboard_data",
            checkpoint_config=air.config.CheckpointConfig(
                num_to_keep=None, checkpoint_at_end=False
            ),
            sync_config=tune.SyncConfig(syncer=None),  # Disable syncing
        ),
        param_space=config_tune,
    )
    results = tuner.fit()

    print(
        "Best hyperparameters found were: ",
        results.best_config(
            metric=config["tune_conf"]["metric"],
            mode=config["tune_conf"]["mode"],
        ),
    )


if __name__ == "__main__":
    app.run()
