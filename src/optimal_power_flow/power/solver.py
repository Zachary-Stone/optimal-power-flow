"""Run AC optimal-power-flow solves without notebook side effects."""

import copy
from typing import Any

from pypower.api import ppoption, runopf

from optimal_power_flow.cases.case5_pjm import PowerFlowCase


def solve_ac_opf(case: PowerFlowCase, verbose: bool = False) -> dict[str, Any]:
    """
    Solve an AC optimal-power-flow case and return its PYPOWER result.

    A defensive deep copy prevents PYPOWER's internal preprocessing from
    changing the supplied case. Solver reports are disabled by default so the
    function is safe to use in tests, training jobs, and non-notebook runs.

    Parameters
    ----------
    case : PowerFlowCase
        PYPOWER-compatible network case to solve.
    verbose : bool, optional
        Whether to enable PYPOWER solver progress output. Default is False.

    Returns
    -------
    dict[str, typing.Any]
        Successful PYPOWER AC-OPF result dictionary.

    Raises
    ------
    RuntimeError
        If PYPOWER reports an unsuccessful solve.
    """
    options = ppoption(VERBOSE=int(verbose), OUT_ALL=0)
    result = runopf(copy.deepcopy(case), options)
    if not result["success"]:
        raise RuntimeError("AC-OPF solve was unsuccessful.")
    return result
