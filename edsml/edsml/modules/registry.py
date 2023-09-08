import os

from confit import Registry, set_default_registry

user = os.environ.get("USER", "ML")


@set_default_registry
class MLRegistry:
    factory = Registry((f"{user}_registry", "factory"), entry_points=True)
    data = Registry((f"{user}_registry", "data"), entry_points=True)
    model = Registry((f"{user}_registry", "model"), entry_points=True)
    loss = Registry((f"{user}_registry", "loss"), entry_points=True)
    trainer = Registry((f"{user}_registry", "trainer"), entry_points=True)
    metric = Registry((f"{user}_registry", "metric"), entry_points=True)
    classification_head = Registry(
        (f"{user}_registry", "classification_head"), entry_points=True
    )
    callback = Registry((f"{user}_registry", "checkpoint"), entry_points=True)
    logger = Registry((f"{user}_registry", "logger"), entry_points=True)
    task = Registry((f"{user}_registry", "task"), entry_points=True)

    _catalogue = dict(
        factory=factory,
        data=data,
        model=model,
        loss=loss,
        trainer=trainer,
        classification_head=classification_head,
        metric=metric,
        callback=callback,
        logger=logger,
        task=task,
    )


registry = MLRegistry
