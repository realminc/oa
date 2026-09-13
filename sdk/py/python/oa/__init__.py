"""Python bindings for OA's Rust semantic API."""

from importlib.metadata import version

from . import audio, core, image, matrix, ml, runtime
from ._native import Audio, Engine, Event, Image, Matrix

__all__ = [
	"Audio", "Engine", "Event", "Image", "Matrix", "audio", "core", "image",
	"matrix", "ml", "runtime",
]
__version__ = version("oapython")
