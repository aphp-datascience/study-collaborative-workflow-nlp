from pathlib import Path
from typing import Optional

import pandas as pd
from typing_extensions import Literal

from analysis.gold_data import omop2docs
from analysis.helpers import RARE_COMORBS
from ecci.config import COMORB_CONFIG
from ecci.ner import Extractor
from ecci.ner.pipeline import get_nlp


def run(
    notes: pd.DataFrame,
    qualifiers: Literal["rule_based", "ml", "no_qual"],
    ckpt_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Run the full SpaCY pipeline on the provided notes

    Parameters
    ----------
    notes : pd.DataFrame
        With `note_text`, `note_id` and `visit_occurrence_id` columns
    qualifiers : Literal["rule_based", "ml", "no_qual"]
        Which qualification pipe to use.
    ckpt_path: Optional[Path]
        Path to the PyTorch checkpoint to use
    Returns
    -------
    pd.DataFrame
        DataFrame with 1 row per extracted entity
    """

    assert qualifiers in {"rule_based", "ml", "no_qual"}

    if qualifiers == "rule_based":
        nlp = get_nlp(qualifiers=True)

    elif qualifiers in {"ml", "no_qual"}:
        nlp = get_nlp(qualifiers=False)
        if qualifiers == "ml":
            config = {} if ckpt_path is None else {"ckpt_path": ckpt_path}
            nlp.add_pipe("eds.ecci-qualifier", config=config)

    extract = Extractor(
        notes,
        qualifiers=(qualifiers == "rule_based"),
        nlp=nlp,
    )
    extract.run()

    df = extract.results

    # Keep only 0 - 1 - 2
    df.label_value = df.label_value.replace(
        {True: "1", False: "0"},
    ).str[0]

    if qualifiers == "rule_based":
        df["to_keep"] = ~(df.negation | df.family | df.hypothesis)

    elif qualifiers == "ml":
        df["to_keep"] = df.to_keep & (
            df.negation is not True
        )  # noqa: E712  # comorb-specific negation

    elif qualifiers == "no_qual":
        df["to_keep"] = True

    # Add visit_occurrence_id

    df = df.merge(
        notes[["note_id", "visit_occurrence_id"]],
        on="note_id",
        how="inner",
    )

    df = df[
        [
            "offset_begin",
            "offset_end",
            "label_value",
            "label_name",
            "to_keep",
            "note_id",
            "visit_occurrence_id",
        ]
    ]

    return clean_upsampled(df)


def run_qualifier_only(
    gold,
    notes,
    ckpt_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Run the qualifier pipeline on the gold entities

    Parameters
    ----------
    gold: pd.DataFrame
        Gold entities
    notes : pd.DataFrame
        With `note_text`, `note_id` and `visit_occurrence_id` columns
    ckpt_path: Optional[Path]
        Path to the PyTorch checkpoint to use

    Returns
    -------
    pd.DataFrame
        DataFrame with 1 row per extracted entity
    """

    # Mapping from label name to pipe name
    mapping = pd.DataFrame(COMORB_CONFIG)

    gold = (
        gold.rename(
            columns=dict(
                to_keep="gold_to_keep",
            )
        )
        .merge(mapping[["label_name", "pipe_name"]], on="label_name")
        .rename(
            columns=dict(
                label_name="old_label_name",
                pipe_name="label_name",
            )
        )
    )

    # Keeping an index for final merge
    gold["span_id"] = list(range(len(gold)))

    nlp = get_nlp(qualifiers=True, no_pipes=True)

    docs = omop2docs(
        notes[notes.note_id.isin(gold.note_id.unique())],
        gold,
        nlp,
        extensions=["gold_to_keep", "span_id"],
        doc_extensions=["cse", "patient_type"],
    )

    config = {} if ckpt_path is None else {"ckpt_path": ckpt_path}
    nlp.add_pipe("eds.ecci-qualifier", config=config)

    list_docs = list(nlp.pipe(docs))
    list_docs = [
        {
            "span_id": span.kb_id,
            "to_keep": span._.to_keep,
            "negation": span._.negation,
            "hypothesis": span._.hypothesis,
            "family": span._.family,
        }
        for doc in list_docs
        for span_type, spans in doc.spans.items()
        for span in spans
        if span_type != "pollutions"
    ]

    results = (
        pd.DataFrame(list_docs)
        .drop_duplicates()
        .merge(
            gold,
            on="span_id",
            validate="one_to_one",
        )
        .drop(["span_id", "label_name"], axis=1)
        .rename(
            columns=dict(
                old_label_name="label_name",
            )
        )
    )

    return clean_upsampled(results)


def clean_upsampled(data: pd.DataFrame):
    """
    On upsampled snippets, remove extractions different from the
    upsampled comorbidity
    """
    for rare_comorb in RARE_COMORBS:
        data = data[
            ~(
                data.note_id.str.contains(rare_comorb, regex=False)
                & (data.label_name != rare_comorb)
            )
        ]

    return data
