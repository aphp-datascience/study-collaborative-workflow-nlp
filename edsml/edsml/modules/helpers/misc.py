import os

import torch
from confit.config import Reference


def set_proxy():
    proxy = "http://proxym-inter.aphp.fr:8080"
    os.environ["http_proxy"] = proxy
    os.environ["HTTP_PROXY"] = proxy
    os.environ["https_proxy"] = proxy
    os.environ["HTTPS_PROXY"] = proxy


def flatten_dict(d, parent_key="", sep="/"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            v = str(v) if isinstance(v, Reference) else v
            items.append((new_key, v))
    return dict(items)


def shift(x, dim, n, pad=0):
    shape = list(x.shape)
    shape[dim] = abs(n)

    slices = [slice(None)] * x.ndim
    slices[dim] = slice(n, None) if n >= 0 else slice(None, n)
    pad = torch.full(shape, fill_value=pad, dtype=x.dtype, device=x.device)
    x = torch.cat(
        ([pad] if n > 0 else []) + [x] + ([pad] if n < 0 else []), dim=dim
    ).roll(dims=dim, shifts=n)
    return x[tuple(slices)]
