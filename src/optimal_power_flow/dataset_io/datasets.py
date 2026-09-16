"""Create deterministic PyTorch datasets and loaders for OPF tabular data."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class OPFDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """
    Store canonical OPF feature and target matrices as float32 tensors.

    Parameters
    ----------
    data : pandas.DataFrame
        Canonical numeric rows containing all selected features and targets.
    input_columns : tuple[str, ...]
        Ordered feature-column names.
    output_columns : tuple[str, ...]
        Ordered target-column names.

    Raises
    ------
    ValueError
        If data is empty, a requested column is absent, or values are not
        finite numeric values.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        input_columns: tuple[str, ...],
        output_columns: tuple[str, ...],
    ) -> None:
        """Initialize tensor-backed OPF features and targets."""
        if data.empty:
            raise ValueError("OPF dataset must contain at least one row.")
        requested_columns = input_columns + output_columns
        missing_columns = set(requested_columns).difference(data.columns)
        if missing_columns:
            raise ValueError(
                f"OPF dataset is missing columns: {sorted(missing_columns)}."
            )
        values = data.loc[:, list(requested_columns)].to_numpy(
            dtype=np.float32, copy=True
        )
        if not np.isfinite(values).all():
            raise ValueError("OPF dataset values must be finite.")
        input_size = len(input_columns)
        self._inputs = torch.from_numpy(values[:, :input_size])
        self._outputs = torch.from_numpy(values[:, input_size:])

    def __len__(self) -> int:
        """
        Return the number of OPF scenarios.

        Returns
        -------
        int
            Number of feature-target pairs.
        """
        return len(self._inputs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return one feature vector and target vector.

        Parameters
        ----------
        index : int
            Scenario row index.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Float32 feature and target vectors.
        """
        return self._inputs[index], self._outputs[index]


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    """
    Store deterministic training and test row partitions.

    Parameters
    ----------
    training_data : pandas.DataFrame
        Rows selected for optimization.
    test_data : pandas.DataFrame
        Remaining rows reserved for evaluation.
    """

    training_data: pd.DataFrame
    test_data: pd.DataFrame


@dataclass(frozen=True, slots=True)
class OPFDataLoaders:
    """
    Store PyTorch loaders and their corresponding deterministic data split.

    Parameters
    ----------
    split : DatasetSplit
        Underlying train/test row partition.
    training : torch.utils.data.DataLoader
        Shuffled training loader.
    test : torch.utils.data.DataLoader
        Ordered test loader.
    """

    split: DatasetSplit
    training: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    test: DataLoader[tuple[torch.Tensor, torch.Tensor]]


def split_train_test(
    data: pd.DataFrame, train_fraction: float, seed: int
) -> DatasetSplit:
    """
    Split canonical OPF rows deterministically without changing global RNG state.

    Parameters
    ----------
    data : pandas.DataFrame
        Canonical OPF feature and target rows.
    train_fraction : float
        Fraction of rows selected for training.
    seed : int
        Seed passed to a local legacy NumPy generator, preserving the notebook's
        pandas ``sample(random_state=seed)`` selection behavior.

    Returns
    -------
    DatasetSplit
        Training rows in sampled order and remaining test rows in source order.

    Raises
    ------
    ValueError
        If data cannot form non-empty train and test partitions, or settings
        are invalid.
    """
    if data.empty:
        raise ValueError("Cannot split an empty OPF dataframe.")
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between 0 and 1.")
    training_data = data.sample(
        frac=train_fraction,
        random_state=np.random.RandomState(seed),
    )
    test_data = data.drop(training_data.index)
    if training_data.empty or test_data.empty:
        raise ValueError("Train/test split must create two non-empty partitions.")
    return DatasetSplit(training_data=training_data, test_data=test_data)


def build_data_loaders(
    data: pd.DataFrame,
    input_columns: tuple[str, ...],
    output_columns: tuple[str, ...],
    train_fraction: float,
    seed: int,
    batch_size: int,
) -> OPFDataLoaders:
    """
    Build reproducible training and test loaders from canonical OPF data.

    Parameters
    ----------
    data : pandas.DataFrame
        Canonical OPF feature and target rows.
    input_columns : tuple[str, ...]
        Ordered feature-column names.
    output_columns : tuple[str, ...]
        Ordered target-column names.
    train_fraction : float
        Fraction of rows selected for training.
    seed : int
        Local seed for split selection and training-loader shuffling.
    batch_size : int
        Maximum number of rows per loader batch.

    Returns
    -------
    OPFDataLoaders
        Reproducible train/test split and its paired PyTorch loaders.

    Raises
    ------
    ValueError
        If batch size is invalid or the data cannot form valid datasets.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    split = split_train_test(data, train_fraction, seed)
    training_dataset = OPFDataset(split.training_data, input_columns, output_columns)
    test_dataset = OPFDataset(split.test_data, input_columns, output_columns)
    generator = torch.Generator().manual_seed(seed)
    return OPFDataLoaders(
        split=split,
        training=DataLoader(
            training_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        ),
        test=DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    )
