import os
from collections import ChainMap
from itertools import product
from operator import itemgetter
from pathlib import Path
from typing import Any, Dict, List, Optional

from confit.config import Config
from torch.utils.data import DataLoader


class Tuner:
    def __init__(
        self,
        tune_params: List[Dict[str, Any]],
        base_conf: Config,
        name: Optional[str] = None,
        experience_naming: Optional[str] = None,
        save_n_max: int = 10,
    ):
        self.tune_params = list(self.build_params(tune_params))
        self.base_conf = (
            base_conf
            if isinstance(base_conf, Config)
            else Config.from_disk(Path(base_conf))
        )
        self.base_conf_str = self.base_conf.to_str()

        self.experience_naming = (
            experience_naming if experience_naming is not None else lambda d: ""
        )

        self.name = name

        self.ckpt_handler = CheckpointHandling(
            save_n_max=save_n_max,
        )

    def build_params(self, tune_params: List[Dict[str, Any]]):
        all_params = []
        for params in tune_params:
            p = []
            for values in zip(*params.values()):
                p.append({k: v for k, v in zip(params.keys(), values)})
            all_params.append(p)

        for params in product(*all_params):
            yield params

    def iterate_over_tuned_config(
        self,
        resolve: bool = True,
    ):
        for modifs in self.tune_params:
            conf = Config.from_str(self.base_conf_str)  # copy all
            for modif in modifs:
                for path, value in modif.items():
                    copy = Config(**conf)
                    *path, last = path.split(".")
                    for bit in path:
                        copy = copy.setdefault(bit, Config())
                    copy[last] = value

            conf = conf.resolve() if resolve else conf
            modifs = dict(ChainMap(*modifs))  # merging dict

            yield (
                f"{self.name}-{self.experience_naming(modifs)}",
                modifs,
                conf,
            )


class CheckpointHandling:
    def __init__(self, save_n_max: int):
        self.save_n_max = save_n_max
        self.ckpts = []
        self.trainer = None
        self.model = None

    def __call__(
        self,
        trainer,
        conf,
    ):
        if not self.trainer:
            self.trainer = trainer

        path = trainer.checkpoint_callback.best_model_path
        score = trainer.checkpoint_callback.best_model_score.cpu().item()

        ckpt = dict(
            path=path,
            score=score,
            conf=conf,
        )

        self.ckpts.append(ckpt)
        self.ckpts.sort(key=itemgetter("score"), reverse=True)

        self.ckpts, to_remove = (
            self.ckpts[: self.save_n_max],
            self.ckpts[self.save_n_max :],
        )

        for ckpt in to_remove:
            os.remove(ckpt["path"])

    def best_to_cpu(self):
        self.ckpts.sort(key=itemgetter("score"))
        best = self.ckpts[0]
        conf = best["conf"]

        print(f"SAVING {best} VIA CPU")

        # Using a CPU-only trainer for final checkpointing
        conf["trainer"]["gpu"] = conf["trainer"]["cpu"]

        # Empty dataloader to bind model and trainer during predict
        empty_dl = DataLoader([])

        conf = conf.resolve()

        model = conf["model"]  # .load_from_checkpoint(ckpt_path)
        trainer = conf["trainer"]
        trainer.checkpoint_callback.filename = str(Path(best["path"]).stem) + "cpu.ckpt"

        trainer.predict(model, dataloaders=empty_dl, ckpt_path=best["path"])

        trainer.save_checkpoint(
            trainer.checkpoint_callback.filename,
            weights_only=True,
        )
