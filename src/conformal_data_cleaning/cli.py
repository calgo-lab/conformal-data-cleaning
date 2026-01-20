from enum import Enum
from logging import getLogger
from typing import Annotated

import typer
from tab_err.error_mechanism import EAR, ECAR, ENAR
from typer.params import Option

from cleaner import ConformalForestCleaner, ForestCleaner, PrecisionForestCleaner
from experiment import Experiment

cli = typer.Typer()

logger = getLogger(__name__)


class Cleaner(str, Enum):
    ConformalForestCleaner = ConformalForestCleaner.__name__
    ForestCleaner = ForestCleaner.__name__
    PrecisionForestCleaner = PrecisionForestCleaner.__name__


@cli.command()
def main(
    experiment_name: Annotated[str, Option(...)],
    dataset_id: Annotated[int, Option(...)],
    error_rate: Annotated[float, Option(min=0, max=0.99999)],
    num_repetitions: Annotated[int, Option(...)],
    cleaner: Annotated[str, Option(...)],
    train_size: Annotated[float, Option(min=0.0001, max=0.99999)] = 0.8,
):
    for error_mechanism in [ECAR, ENAR, EAR]:
        msg = f"Starting experiment with {error_mechanism.__name__}"
        logger.info(msg)

        for confidence_level in [0.9, 0.99, 0.999, 0.9999, 0.99999]:
            msg = f"Starting experiment with confidence_level {confidence_level}"
            logger.info(msg)

            for seed in range(num_repetitions):
                msg = f"Starting repetition {seed}"
                logger.info(msg)

                experiment = Experiment(name=experiment_name, dataset_id=dataset_id, seed=seed, train_size=train_size)

                _ = experiment.fit_and_get_baseline_model()
                _ = experiment.perturb_and_get_perturbed_test_data_and_error_mask(error_rate=error_rate, error_mechanism=error_mechanism)
                _ = experiment.fit_and_get_cleaner(cleaner=cleaner, confidence_level=confidence_level)
                _ = experiment.multiple_clean_and_get_multiple_cleaned_test_data_and_multiple_cleaned_mask()
                _ = experiment.clean_and_get_cleaned_test_data_and_cleaned_mask()

                experiment.compute_and_save_results()
