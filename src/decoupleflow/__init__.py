from importlib import import_module

from .DecoupleFlow import DecoupleFlow

__version__ = "0.1.0"
version = __version__
__all__ = ["DecoupleFlow", "augment_fn"]


def __getattr__(name):
    if name == "augment_fn":
        return import_module(".augment_fn", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
