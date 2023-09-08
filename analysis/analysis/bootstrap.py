from functools import reduce

import numpy as np
import pandas as pd

rng = np.random.default_rng(seed=42)


def sample_rows(row: pd.Series):
    """
    row should contain "TP", "FN" and "FP", with values being
    the corresponding count.

    Returns a Series with resampled TP, FN and FP.

    For instance, if row = pd.Series(dict(TP=10, FP=2, FN=4)),
    we could return pd.Series(dict(TP=11, FP=2, FN=3))
    """

    copied_row = row.copy()

    choices = ["TP", "FN", "FP"]
    p = row[choices].to_numpy()
    total = p.sum()
    p = p / total

    sampled_choices = rng.choice(
        choices,
        size=int(total),
        replace=True,
        p=p.astype("float"),
    )

    sampled_choices_list, sampled_choices_count = np.unique(
        sampled_choices, return_counts=True
    )
    copied_row[sampled_choices_list] = sampled_choices_count

    return copied_row


def ci_from_list_of_dfs(dfs):
    # Elementwise concatenation of values
    concatenated = reduce(
        lambda x, y: x + y,
        [df.applymap(lambda y: [y]) for df in dfs],
    )

    # CI compute and formatting
    # concatenated = concatenated.applymap(
    #     lambda list_metrics: f"({np.quantile(list_metrics, q=0.025):.2f} - {np.quantile(list_metrics, q=0.975):.2f})"
    # )

    return concatenated
