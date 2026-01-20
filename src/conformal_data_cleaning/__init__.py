from __future__ import annotations

import os
from logging import Formatter, StreamHandler, getLogger

import numpy as np
from . import cleaner
from . import conformal_inference

__all__ = ["cleaner", "conformal_inference"]




def setup_logger(name: str) -> None:
    """Sets up a common logging format.

    Args:
        name (str): `name` of the logger to setup
    """
    level = os.getenv("LOG_LEVEL", "INFO")
    logger = getLogger(name)
    logger.setLevel(level)
    handler = StreamHandler()
    formatter = Formatter("[%(levelname)s] %(name)s:%(lineno)d - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


setup_logger(__name__)
