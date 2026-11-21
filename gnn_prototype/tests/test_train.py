from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from train import _stratified_split


def make_dataset(class_counts):
    """Create a synthetic dataset with joint classification labels."""
    dataset = []

    for (bottleneck, security), count in class_counts.items():
        for _ in range(count):
            dataset.append(
                SimpleNamespace(
                    y=torch.tensor(
                        [[0.5, bottleneck, security]],
                        dtype=torch.float32,
                    )
                )
            )

    return dataset


def get_joint_distribution(subset):
    return Counter(
        (
            int(subset[i].y[0, 1]),
            int(subset[i].y[0, 2]),
        )
        for i in range(len(subset))
    )


def test_split_preserves_joint_class_distribution():
    dataset = make_dataset({
        (0, 0): 60,
        (0, 1): 45,
        (0, 2): 30,
        (1, 0): 30,
        (1, 1): 25,
        (1, 2): 10,
    })

    train, validation, test = _stratified_split(
        dataset,
        0.70,
        0.15,
        0.15,
        seed=42,
    )

    total = get_joint_distribution(dataset)

    for subset in (train, validation, test):
        distribution = get_joint_distribution(subset)

        assert set(distribution) == set(total)

        for label, total_count in total.items():
            expected = total_count / len(dataset)
            actual = distribution[label] / len(subset)

            assert abs(actual - expected) < 0.05


def test_split_is_deterministic_and_disjoint():
    dataset = make_dataset({
        (0, 0): 60,
        (0, 1): 45,
        (0, 2): 30,
        (1, 0): 30,
        (1, 1): 25,
        (1, 2): 10,
    })

    first = _stratified_split(
        dataset, 0.70, 0.15, 0.15, seed=42
    )

    second = _stratified_split(
        dataset, 0.70, 0.15, 0.15, seed=42
    )

    first_indices = [set(split.indices) for split in first]
    second_indices = [set(split.indices) for split in second]

    assert first_indices == second_indices

    assert first_indices[0].isdisjoint(first_indices[1])
    assert first_indices[0].isdisjoint(first_indices[2])
    assert first_indices[1].isdisjoint(first_indices[2])

    assert set.union(*first_indices) == set(range(len(dataset)))


def test_split_rejects_joint_class_with_insufficient_samples():
    dataset = make_dataset({
        (0, 0): 30,
        (0, 1): 20,
        (1, 2): 2,
    })

    with pytest.raises(
        ValueError,
        match="fewer than 3 samples",
    ):
        _stratified_split(
            dataset,
            0.70,
            0.15,
            0.15,
            seed=42,
        )