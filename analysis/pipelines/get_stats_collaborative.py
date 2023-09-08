import typer

from analysis import data
from analysis.collaborative import (
    get_collaborative_matrix,
    get_metrics_with_ci_same_train_val_subcohort,
)

BOOTSTRAP_ITER = 200

app = typer.Typer(pretty_exceptions_enable=False)


def step_1(chosen_metric: str = "F1"):
    """
    Effects of collaborative / Compare EDS and BASE camemBERT models
    """

    for mode in ["full_on_notes"]:  # , "qualifier_on_gold"]:
        # Compare EDS vs BERT
        matrix = get_collaborative_matrix(
            chosen_metric, mode=mode, bootstrap_iter=BOOTSTRAP_ITER, compare="eds-base"
        )
        data.save(
            matrix,
            f"collaborative_matrix_{mode}_eds_vs_base",
            f"Collaborative matrix (EDS vs BASE) ({mode})",
            export=True,
            model="compare",
            bbox_inches="tight",
        )

        # same_train_val_metrics = get_metrics_with_ci_same_train_val_subcohort(
        #     mode=mode,
        #     bootstrap_iter=BOOTSTRAP_ITER,
        # )
        # data.save(
        #     same_train_val_metrics,
        #     f"same_train_val_metrics_{mode}_eds_vs_base",
        #     f"Metrics of model trained and validated on same cohort ({mode})",
        #     export=True,
        #     model="compare",
        # )


@app.command()
def main():
    step_1(chosen_metric="F1")


if __name__ == "__main__":
    app()
