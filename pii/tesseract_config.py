"""Resolve the Tesseract executable path without depending on PATH."""

import json
import os
from pathlib import Path


DEFAULT_TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "ocr.json"


def resolve_tesseract_cmd(config_path=DEFAULT_CONFIG_PATH, environ=None):
    """Return the configured command: environment, JSON config, then Windows default.

    Paths are returned as supplied; existence checks are deliberately left to
    pytesseract when OCR is invoked, keeping resolution deterministic.
    """
    environ = os.environ if environ is None else environ
    env_command = environ.get("TESSERACT_CMD")
    if env_command:
        return env_command

    with open(config_path, encoding="utf-8") as config_file:
        config = json.load(config_file)
    return config.get("tesseract_cmd") or DEFAULT_TESSERACT_CMD
