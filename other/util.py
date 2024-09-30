from collections.abc import Iterable

from transformers import pipeline as pl
from transformers import Pipeline

import torch


device = "cuda" if torch.cuda.is_available() else "cpu"


def pipeline(*args, **kwargs) -> Pipeline:
    """Simple pipeline overwrite to add the current device."""
    if "device" in kwargs:
        return pl(*args, **kwargs)
    else:
        return pl(*args, device=device, **kwargs)


def pretty_print(obj):
    """Print an iterable in multiple lines."""
    if isinstance(obj, dict):
        print(obj)
        for k, v in obj.items():
            print(f"{k}: {v}")
    elif isinstance(obj, Iterable):
        for e in obj:
            print(e)
    else:
        print(obj)
