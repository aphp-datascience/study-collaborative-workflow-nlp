from typing import List, Optional

import pandas as pd
from spacy.language import Language
from spacy.tokens import Doc, Span

from analysis.helpers import add_notes_infos


def format_annotations(
    gold: pd.DataFrame,
    notes: pd.DataFrame,
):
    """
    Format and clean gold annotations
    """

    # Remove placeholders used on notes with no entities
    gold = gold[gold.gold_label_name.notna()]

    # Selecting columns of interest
    gold = gold[
        [
            "note_id",
            "visit_occurrence_id",
            "person_id",
            "offset_begin",
            "offset_end",
            "cse",
            "patient_type",
        ]
        + [c for c in gold.columns if "gold" in c]
    ].copy()

    gold = add_notes_infos(gold, notes)

    # Keep only 0 - 1 - 2
    gold.gold_label_value = gold.gold_label_value.replace({True: "1", False: "0"}).str[
        0
    ]

    gold.gold_label_value = gold.gold_label_value.str[0]

    # Removing incorrect extractions:
    gold = gold[gold.gold_label_value != "0"]

    # Merging qualifiers values
    gold["to_keep"] = ~(
        gold.gold_negation_value | gold.gold_family_value | gold.gold_hypothesis_value
    )

    gold.rename(
        columns=dict(gold_label_value="label_value", gold_label_name="label_name"),
        inplace=True,
    )

    return gold[
        [
            "offset_begin",
            "offset_end",
            "label_value",
            "label_name",
            "to_keep",
            "note_id",
            "visit_occurrence_id",
            "snippet",
            "lexical_variant",
            "cse",
            "patient_type",
        ]
    ]


def omop2docs(
    note: pd.DataFrame,
    note_nlp: pd.DataFrame,
    nlp: Language,
    extensions: Optional[List[str]] = None,
    doc_extensions: Optional[List[str]] = None,
    id_col: str = "span_id",
) -> List[Doc]:
    """
    Transforms an OMOP-formatted pair of dataframes into a list of documents.

    Parameters
    ----------
    note : pd.DataFrame
        The OMOP `note` table.
    note_nlp : pd.DataFrame
        The OMOP `note_nlp` table
    nlp : Language
        spaCy language object.
    extensions : Optional[List[str]], optional
        Extensions to keep, by default None
    doc_extensions : Optional[List[str]], optional
        Document level extensions to keep, by default None
    id_col: str
        Name of the `gold` column storing the ID of the span
    Returns
    -------
    List[Doc] :
        List of spaCy documents
    """

    note = note.copy()
    note_nlp = note_nlp.copy()

    extensions = extensions or []
    doc_extensions = doc_extensions or []

    for e in extensions:
        if not Span.has_extension(e):
            Span.set_extension(e, default=None)

    for e in doc_extensions:
        if not Doc.has_extension(e):
            Doc.set_extension(e, default=None)

    def row2ent(row):
        d = dict(
            start_char=row.offset_begin,
            end_char=row.offset_end,
            label=row.get("label_name"),
            extensions={ext: row.get(ext) for ext in extensions},
        )

        return d

    # Create entities
    note_nlp["ents"] = note_nlp.apply(row2ent, axis=1)

    note_nlp = note_nlp.groupby("note_id", as_index=False)["ents"].agg(list)

    note = note.merge(note_nlp, on=["note_id"], how="left")

    # Generate documents
    note["doc"] = note.note_text.apply(nlp)

    # Process documents
    for _, row in note.iterrows():
        doc = row.doc
        doc._.note_id = row.note_id
        for e in doc_extensions:
            setattr(doc._, e, row.get(e))

        if not isinstance(row.ents, list):
            continue

        for ent in row.ents:
            span = doc.char_span(
                ent["start_char"],
                ent["end_char"],
                ent["label"],
                alignment_mode="expand",
                kb_id=int(ent["extensions"].get(id_col)),
            )

            for k, v in ent["extensions"].items():
                setattr(span._, k, v)

            if span.label_ not in doc.spans:
                doc.spans[span.label_] = [span]
            else:
                doc.spans[span.label_].append(span)

    return list(note.doc)
