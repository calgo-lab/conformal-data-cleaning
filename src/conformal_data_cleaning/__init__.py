from __future__ import annotations

import os
from logging import Formatter, StreamHandler, getLogger

import numpy as np


def seed_and_get_generator(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed=seed) if seed is not None else np.random.default_rng()


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
