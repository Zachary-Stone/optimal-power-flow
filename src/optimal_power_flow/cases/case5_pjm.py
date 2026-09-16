"""Provide the PYPOWER-compatible five-bus PJM tutorial case."""

import numpy as np
from numpy.typing import NDArray

PowerFlowCase = dict[str, float | NDArray[np.float64]]


def case5_pjm() -> PowerFlowCase:
    """
    Construct a fresh PYPOWER-compatible PGLib/PJM five-bus network case.

    The values are converted from PGLib-OPF's ``case5_pjm.m`` and match the
    inline case used in the tutorial notebook. A new dictionary and new arrays
    are returned on every call because PYPOWER transforms and augments cases
    while solving them.

    Returns
    -------
    PowerFlowCase
        Base-MVA value and the MATPOWER/PYPOWER ``areas``, ``bus``, ``gen``,
        ``gencost``, and ``branch`` matrices.
    """
    return {
        "baseMVA": 100.0,
        "areas": np.array([[1.0, 4.0]]),
        "bus": np.array(
            [
                [1, 2, 0.0, 0.0, 0.0, 0.0, 1, 1.0, 0.0, 230.0, 1, 1.1, 0.9],
                [2, 1, 300.0, 98.61, 0.0, 0.0, 1, 1.0, 0.0, 230.0, 1, 1.1, 0.9],
                [3, 2, 300.0, 98.61, 0.0, 0.0, 1, 1.0, 0.0, 230.0, 1, 1.1, 0.9],
                [4, 3, 400.0, 131.47, 0.0, 0.0, 1, 1.0, 0.0, 230.0, 1, 1.1, 0.9],
                [5, 2, 0.0, 0.0, 0.0, 0.0, 1, 1.0, 0.0, 230.0, 1, 1.1, 0.9],
            ],
            dtype=np.float64,
        ),
        "gen": np.array(
            [
                [1, 20.0, 0.0, 30.0, -30.0, 1.0, 100.0, 1, 40.0, 0.0],
                [1, 85.0, 0.0, 127.5, -127.5, 1.0, 100.0, 1, 170.0, 0.0],
                [3, 260.0, 0.0, 390.0, -390.0, 1.0, 100.0, 1, 520.0, 0.0],
                [4, 100.0, 0.0, 150.0, -150.0, 1.0, 100.0, 1, 200.0, 0.0],
                [5, 300.0, 0.0, 450.0, -450.0, 1.0, 100.0, 1, 600.0, 0.0],
            ],
            dtype=np.float64,
        ),
        "gencost": np.array(
            [
                [2, 0.0, 0.0, 3, 0.0, 14.0, 0.0],
                [2, 0.0, 0.0, 3, 0.0, 15.0, 0.0],
                [2, 0.0, 0.0, 3, 0.0, 30.0, 0.0],
                [2, 0.0, 0.0, 3, 0.0, 40.0, 0.0],
                [2, 0.0, 0.0, 3, 0.0, 10.0, 0.0],
            ],
            dtype=np.float64,
        ),
        "branch": np.array(
            [
                [1, 2, 0.00281, 0.0281, 0.00712, 400, 400, 400, 0, 0, 1, -30, 30],
                [1, 4, 0.00304, 0.0304, 0.00658, 426, 426, 426, 0, 0, 1, -30, 30],
                [1, 5, 0.00064, 0.0064, 0.03126, 426, 426, 426, 0, 0, 1, -30, 30],
                [2, 3, 0.00108, 0.0108, 0.01852, 426, 426, 426, 0, 0, 1, -30, 30],
                [3, 4, 0.00297, 0.0297, 0.00674, 426, 426, 426, 0, 0, 1, -30, 30],
                [4, 5, 0.00297, 0.0297, 0.00674, 240, 240, 240, 0, 0, 1, -30, 30],
            ],
            dtype=np.float64,
        ),
    }
