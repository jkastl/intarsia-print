"""Local web UI for the intarsia-print pipeline: image -> Gemini poster art
-> quantized levels -> STL, with every step reviewable and re-runnable.
"""

from .app import app, main

__all__ = ["app", "main"]
