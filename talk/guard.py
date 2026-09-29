"""Refuse to read challenge data from inside the talk.

The talk uses only synthetic patients (talk/synthetic.py) and our own aggregate scores
(talk/data/*.csv). `install()` wraps `open` so that any attempt to read a challenge file, a
submission, out-of-fold predictions or the quarantined reference archive raises instead.
"""
import builtins
from pathlib import Path

# Names of challenge files and derived-data locations the talk must never touch.
FORBIDDEN = ("DB-1773398340961", "/artifacts/", "submission_", "oof_predictions",
             "reference_archive")


def is_forbidden(path) -> bool:
    text = "/" + Path(path).as_posix()
    return any(name in text for name in FORBIDDEN)


def install() -> None:
    if getattr(builtins.open, "_talk_guard", False):
        return
    original = builtins.open

    def guarded(file, *args, **kwargs):
        if isinstance(file, (str, Path)) and is_forbidden(file):
            raise PermissionError(f"talk guard: refusing to open challenge data {file}")
        return original(file, *args, **kwargs)

    guarded._talk_guard = True
    builtins.open = guarded
