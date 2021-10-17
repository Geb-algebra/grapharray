"""Functions for treating graph variables."""

from __future__ import annotations

import json
from typing import Union, Callable
import _io
import numpy as np

from grapharray.classes import BaseGraph, NodeArray, EdgeArray


def apply_array_function(var: Union[NodeArray, EdgeArray], function: Callable):
    """Execute a function for np.ndarray to the array of NodeVar or EdgeVar.

    Args:
        var: A variable to apply function
        function: A function for np.ndarray to apply.

    Returns:
        An instance of the same class as var's, whose array is the result of
        the function passed.
    """
    res_array = function(var.array)
    if isinstance(var, NodeArray):
        return NodeArray(
            var.base_graph, init_val=res_array, is_array_2d=var.is_2d
        )
    elif isinstance(var, EdgeArray):
        return EdgeArray(
            var.base_graph, init_val=res_array, is_array_2d=var.is_2d
        )
    else:
        raise TypeError(
            f"Invalid type of argument {type(var)}. "
            f"It must be NodeVar or EdgeVar"
        )


def exp(var: Union[NodeArray, EdgeArray]):
    """Element-wise exponential"""
    return apply_array_function(var, np.exp)


def log(var: Union[NodeArray, EdgeArray]):
    """Element-wise natural logarithm"""
    return apply_array_function(var, np.log)


def sum(var: Union[NodeArray, EdgeArray]):
    """Sum up all variables"""
    return np.sum(var.array)
