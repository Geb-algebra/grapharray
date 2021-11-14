import pytest

import numpy as np
from grapharray.classes import NodeArray
from grapharray.functions import exp, argmin, argmax


@pytest.fixture
def node_edge_array(graph, NodeEdgeArray, dict_init_val):
    return NodeEdgeArray(graph, init_val=dict_init_val)


def test_exp(node_edge_array, graph, NodeEdgeArray):
    assert exp(node_edge_array) == NodeEdgeArray(
        graph, np.exp(node_edge_array.array)
    )


def test_argmax(node_edge_array, graph, NodeEdgeArray):
    assert (
        argmax(node_edge_array) == 6
        if type(node_edge_array) == NodeArray
        else (4, 6)
    )


def test_argmin(node_edge_array, graph, NodeEdgeArray):
    assert (
        argmin(node_edge_array) == 0
        if type(node_edge_array) == NodeArray
        else (0, 2)
    )

