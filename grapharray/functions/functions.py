"""Functions for treating graph variables."""

from __future__ import annotations

from typing import Any, Union, Callable
import numpy as np

from grapharray.classes import NodeArray, EdgeArray


def apply_element_wise_function(
    var: Union[NodeArray, EdgeArray], function: Callable
) -> Union[NodeArray, EdgeArray]:
    """Execute a element-wise function for np.ndarray to NodeVar or EdgeVar.

    Args:
        var: A variable to apply function
        function: A function for np.ndarray to apply.

    Returns:
        An instance of the same class as var's, whose array is the result of
        the function passed i.e., function(var.array).
    """
    res_array = function(var._array)
    if isinstance(var, NodeArray):
        return NodeArray(var.base_graph, init_val=res_array)
    elif isinstance(var, EdgeArray):
        return EdgeArray(var.base_graph, init_val=res_array)
    else:
        raise TypeError(
            f"Invalid type of argument {type(var)}. "
            f"It must be NodeVar or EdgeVar"
        )


def exp(var: Union[NodeArray, EdgeArray]) -> Union[NodeArray, EdgeArray]:
    """Element-wise exponential"""
    return apply_element_wise_function(var, np.exp)


def log(var: Union[NodeArray, EdgeArray]) -> Union[NodeArray, EdgeArray]:
    """Element-wise natural logarithm"""
    return apply_element_wise_function(var, np.log)


def get_representative_value(
    var: Union[NodeArray, EdgeArray], function: Callable
) -> float:
    """Apply a function for np.ndarray that returns a scalar to Node/EdgeArray

    Args:
        var: A variable to apply function
        function: A function for np.ndarray to apply.

    Returns:
        (float) The result of "function(var)"
    """
    return function(var)


def argmin(var: Union[NodeArray, EdgeArray]) -> Any:
    """Return the Node/Edge index that has the maximum value"""
    return min(var.index.keys(), key=lambda x: var[x])


def argmax(var: Union[NodeArray, EdgeArray]) -> Any:
    """Return the Node/Edge index that has the minimum value"""
    return max(var.index.keys(), key=lambda x: var[x])
