from __future__ import annotations

from typing import Any, Union, Callable
import numpy as np

from grapharray.classes import NodeArray, EdgeArray


def sum(var: Union[NodeArray, EdgeArray]) -> float:
    """Sum up all variables"""
    return np.sum(var._array)


def max(var: Union[NodeArray, EdgeArray]) -> float:
    """The maximum of all variables"""
    return np.max(var._array)


def min(var: Union[NodeArray, EdgeArray]) -> float:
    """The minimum of all variables"""
    return np.min(var._array)
