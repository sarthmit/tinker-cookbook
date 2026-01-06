import math
import re
from functools import partial
from typing import Literal, Sequence, cast, List

from tinker_cookbook.recipes.mixed_rl.math_env import (
    PolarisDatasetBuilder,
    MathDatasetBuilder,
    DeepMathDatasetBuilder,
    Gsm8kDatasetBuilder,
)
from tinker_cookbook.recipes.mixed_rl.code_env import DeepcoderDatasetBuilder
from tinker_cookbook.rl.types import RLDatasetBuilder

# Populate the dataset builder map after all classes are defined
DATASET_BUILDER_MAP = {
    "math": MathDatasetBuilder,
    "polaris": PolarisDatasetBuilder,
    "deepmath": DeepMathDatasetBuilder,
    "gsm8k": Gsm8kDatasetBuilder,
    "deepcoder": DeepcoderDatasetBuilder,
}

def get_dataset_builders(
    dataset_names: List[str],
    splits: List[float],
    batch_size: int,
    model_name_for_tokenizer: str,
    renderer_name: str,
    group_size: int,
    seed: int = 0,
) -> List[RLDatasetBuilder]:
    """
    Unified function to get any combination of dataset builders.
    Args:
        dataset_names: List of dataset names to combine
        batch_size: List of batch sizes corresponding to each dataset
        model_name_for_tokenizer: Model name for tokenizer
        renderer_name: Name of the renderer to use
        group_size: Number of environments per group
        seed: Random seed for data shuffling (default: 0)
    Returns:
        A combined dataset builder instance
    """
    if len(dataset_names) != len(batch_size):
        raise ValueError("dataset_names and batch_size must have the same length")

    builders = []
    for name, split in zip(dataset_names, splits):
        if name not in DATASET_BUILDER_MAP:
            raise ValueError(
                f"Unknown math dataset: {name}. Available: {list(DATASET_BUILDER_MAP.keys())}"
            )
        builder_class = DATASET_BUILDER_MAP[name]
        builders.append(
            builder_class(
                batch_size=int(split * batch_size),
                model_name_for_tokenizer=model_name_for_tokenizer,
                renderer_name=renderer_name,
                group_size=group_size,
                seed=seed,
            )
        )

    return builders