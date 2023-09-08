from typing import List

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from tqdm import tqdm

from ecci import DATA_DIR
from ecci.io.connectors import SparkData
from ecci.ner import Extractor


def get_validation_dataset(
    spark,
    patient_type: str,
    fetch_notes: bool = False,
):
    """
    Extract all notes from validation dataset
    """

    if fetch_notes:
        S = SparkData(
            spark=spark,
            patient_type="inpatient",
            dataset_type="test",
        )
        note_type = "CR:CRH-HOSPI" if (patient_type == "inpatient") else "CR:CR-CONS"

        notes = S.orbis_note.filter(F.col("note_class_source_value") == note_type)

        notes_pd = notes.toPandas()
        notes_pd = notes_pd[notes_pd.note_text.notna()]
        notes_pd = notes_pd[notes_pd.note_text.str.len() > 500]
        notes_pd = notes_pd.sample(min(50000, len(notes_pd)))

        notes_pd.to_pickle(DATA_DIR / f"validation_dataset_notes_{patient_type}.pickle")

    notes_pd = pd.read_pickle(
        DATA_DIR / f"validation_dataset_notes_{patient_type}.pickle"
    )

    entity_getter = Extractor(
        notes=notes_pd,
        qualifiers=True,
    )
    entity_getter.run()
    entities = entity_getter.results

    return entities


def prepare_for_annotation(
    note_nlp: pd.DataFrame,
    notes: pd.DataFrame,
    kept_comorbs: List[str],
    snippet_length: int,
    max_notes: int,
    patient_type: str,
    seed: int,
):
    all_ents, all_notes = [], []

    note_type = "CR-CONS" if (patient_type == "outpatient") else "CR-HOSP"

    for comorb in kept_comorbs:
        ents = note_nlp[note_nlp.label_name == comorb]
        ents_notes = notes[notes.note_id.isin(ents.note_id.unique())]

        for _, note in tqdm(
            ents_notes.sample(
                min(len(ents_notes), max_notes), random_state=seed
            ).iterrows(),
            total=len(ents_notes),
        ):
            note_text = note.note_text
            note_id = note.note_id

            original_ents = ents[ents.note_id == note_id]
            original_ents = original_ents[original_ents.label_name == comorb]

            if not original_ents.empty:
                new_ents, new_note_text = concat_snippets(
                    original_ents, note_text, snippet_length
                )

                new_ents["new_note_id"] = f"{comorb}-{patient_type}-{note_id}"

                all_ents.append(new_ents)
                all_notes.append(
                    dict(
                        note_id=note_id,
                        note_text=new_note_text,
                        new_note_id=f"{comorb}-{patient_type}-{note_id}",
                        title=f"{comorb} - {note_type}",
                    )
                )

    all_notes = pd.DataFrame(all_notes)
    all_ents = pd.concat(all_ents)

    return (
        all_ents.rename(columns=dict(note_id="old_note_id", new_note_id="note_id")),
        all_notes.rename(columns=dict(note_id="old_note_id", new_note_id="note_id")),
    )


def concat_snippets(
    original_ents: pd.DataFrame,
    note_text: str,
    snippet_length: int,
):
    """
    From a text and a DataFrame of entities, constructs a 'concatenated' text
    composed of snippets of text around entities
    """

    SEP = "\n\n" + 30 * "#" + "\n\n"
    SEP_LENGTH = len(SEP)

    if original_ents.empty:
        return pd.DataFrame(), note_text
    original_ents["tmp_idx"] = np.arange(len(original_ents))

    ents = original_ents.copy()
    ents["note_text"] = note_text

    # distance between ent and next ent
    ents["delta"] = (
        ents["offset_begin"] - ents.shift(1, fill_value=-np.inf)["offset_end"]
    )

    ents["far_from_before"] = ents["delta"] > snippet_length

    # grouping "close" entities together
    n_groups = ents.far_from_before.sum()
    ents.loc[ents.far_from_before, "far_from_before"] = list(range(1, n_groups + 1))
    ents["far_from_before"] = (
        ents["far_from_before"].replace(False, np.nan).ffill().astype(int)
    )
    ents.rename(columns=dict(far_from_before="snippet_num"), inplace=True)

    ents["note_length"] = ents.note_text.str.len()

    # Building one snippet per entity "group"

    snippets = ents.groupby("snippet_num", as_index=False).agg(
        snippet_start=("offset_begin", lambda r: r.min() - snippet_length / 2),
        snippet_end=("offset_end", lambda r: r.max() + snippet_length / 2),
        snippet_text=("note_text", "first"),
        note_length=("note_length", "first"),
    )

    number_cols = snippets.select_dtypes(include="number").columns

    snippets[number_cols] = snippets[number_cols].astype(int)

    snippets["snippet_start"] = np.maximum(0, snippets["snippet_start"])
    snippets["snippet_end"] = snippets[["snippet_end", "note_length"]].min(axis=1)

    snippets["snippet_text"] = snippets.apply(
        lambda r: r.snippet_text[r.snippet_start : r.snippet_end], axis=1
    )

    # Adding separator
    snippets["snippet_text"] = snippets["snippet_text"] + SEP
    snippets["snippet_end"] = snippets["snippet_end"] + SEP_LENGTH
    snippets["snippet_length"] = snippets["snippet_end"] - snippets["snippet_start"]

    # Computing offsets and final concatenated snippet
    snippets["snippet_offset"] = snippets["snippet_length"].cumsum().shift().fillna(0)
    snippets["concat_snippet"] = snippets["snippet_text"].sum()

    # Getting span relative to the new concatenated snippet

    ents = ents.merge(snippets, on=["snippet_num", "note_length"])

    ents["in_snippet_start"] = (
        ents["offset_begin"] + ents["snippet_offset"] - ents["snippet_start"]
    ).astype(int)
    ents["in_snippet_end"] = (
        ents["offset_end"] + ents["snippet_offset"] - ents["snippet_start"]
    ).astype(int)

    # Sanity check

    ents["new_lexical_variant"] = ents.apply(
        lambda r: r.concat_snippet[r.in_snippet_start : r.in_snippet_end], axis=1
    )

    original_ents = original_ents.rename(
        columns=dict(
            offset_begin="original_offset_begin",
            offset_end="original_end",
        )
    )

    ents = ents[
        ["in_snippet_start", "in_snippet_end", "new_lexical_variant", "tmp_idx"]
    ].rename(
        columns=dict(
            in_snippet_start="offset_begin",
            in_snippet_end="offset_end",
        )
    )

    original_ents = original_ents.merge(ents, on="tmp_idx").drop("tmp_idx", axis=1)

    assert all(
        original_ents["new_lexical_variant"].str.normalize("NFKD")
        == original_ents["lexical_variant"].str.normalize("NFKD")
    )

    return (
        original_ents.drop("new_lexical_variant", axis=1),
        snippets["concat_snippet"].iloc[0],
    )
