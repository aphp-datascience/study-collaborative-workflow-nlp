from typing import Any

from edsnlp.utils.extensions import rgetattr


def rsetattr(obj: Any, attr: str, value: Any):
    """
    Get attribute recursively.
    For instance, if `attr=a.b.c`, then under the hood,
    setattr(obj.a.b, "c", value) is executed

    Parameters
    ----------
    obj : Any
        An object
    attr : str
        The name of the attribute to set. Can contain dots.
    value: Any
        The value to set
    """
    splitted = attr.split(".", maxsplit=1)
    last = splitted.pop(-1)
    attr_obj = rgetattr(obj, ".".join(splitted)) if splitted else obj

    setattr(attr_obj, last, value)
