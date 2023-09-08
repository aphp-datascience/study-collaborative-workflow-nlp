from pathlib import Path
from typing import Callable, Optional

from confit.config import Config
from loguru import logger
from spacy.language import Language

from ecci_qualifier import BASE_DIR

from .model import SpacyEcciQualifierModel

DEFAULT_ECCI_QUALIFIER_CONFIG = dict(
    preprocess_size=50000,
    batch_size=32,
    config_path=str(BASE_DIR / "data/inference_camembert_eds.cfg"),
    ckpt_path=str(BASE_DIR / "data/inference_camembert_eds.ckpt"),
    span_getters={
        "@span_getters": "spans-with-context",
        "n_before": 1,
        "n_after": 1,
        "return_type": "text",
        "mode": "sentence",
        "attr": "TEXT",
        "ignore_excluded": True,
        "with_ents": True,
        "with_spangroups": True,
        "output_keys": {"text": "text", "span": "span"},
    },
    annotation_setters={
        "@annotation_setters": "set-all",
    },
    use_better_transformer=True,
)


@Language.factory("eds.ecci-qualifier", default_config=DEFAULT_ECCI_QUALIFIER_CONFIG)
def create_component(
    nlp: Language,
    name: str,
    config_path: Path,
    ckpt_path: Optional[Path],
    span_getters: Callable,
    annotation_setters: Callable,
    batch_size: int,
    preprocess_size: int,
    use_better_transformer: bool,
):
    conf = Config.from_disk(config_path)

    ckpt_path = ckpt_path or BASE_DIR / conf["model"]["checkpoint_path"]

    conf["model"]["checkpoint_path"] = str(ckpt_path)

    logger.info(f"Using checkpoint at {conf['model']['checkpoint_path']}")

    conf = conf.resolve()

    model = conf["model"]
    trainer = conf["trainer"]
    datamodule = conf["data"]

    return SpacyEcciQualifierModel(
        nlp,
        model_parts={
            "model": model,
            "trainer": trainer,
            "datamodule": datamodule,
        },
        span_getters=span_getters,
        annotation_setters=annotation_setters,
        preprocess_size=preprocess_size,
        batch_size=batch_size,
        use_better_transformer=use_better_transformer,
    )
