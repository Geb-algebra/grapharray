import pytest
import numpy as np

from grapharray.classes import (
    NodeArray,
    EdgeArray,
    AdjacencyMatrix,
    IncidenceMatrix,
)


@pytest.fixture
def inc_matrix(graph):
    return IncidenceMatrix(graph)


@pytest.fixture
def adj_matrix(graph):
    edge_f = {(0, 2): 6, (0, 4): 4, (2, 4): 3, (2, 6): 1, (4, 6): 2}
    weight = EdgeArray(graph, edge_f)
    return AdjacencyMatrix(weight)


def test_is_matrix_correct(adj_matrix) -> None:
    true_matrix = np.array(
        [[0, 6, 4, 0], [0, 0, 3, 1], [0, 0, 0, 2], [0, 0, 0, 0]]
    )
    assert np.all(adj_matrix.array == true_matrix)


def test_is_adj_matmul_correct(adj_matrix, graph):
    nv = NodeArray(graph, init_val={0: 1, 2: 2, 4: 3, 6: 4})
    result = adj_matrix @ nv
    answer = NodeArray(graph, init_val={0: 24, 2: 13, 4: 8, 6: 0})
    assert result == answer


def test_is_inc_matmul_correct(graph, inc_matrix):
    edge_f = {(0, 2): 6, (0, 4): 4, (2, 4): 3, (2, 6): 1, (4, 6): 2}
    od_f = {0: -10, 2: 2, 4: 5, 6: 3}
    edge_flow = EdgeArray(graph, init_val=edge_f)
    od_flow = NodeArray(graph, init_val=od_f)
    matmul_res = inc_matrix @ edge_flow
    assert matmul_res == od_flow


def test_is_transposed_matmul_correct(graph, inc_matrix):
    label = {0: 0, 2: 2, 4: 3, 6: 5}
    diff = {(0, 2): 2, (0, 4): 3, (2, 4): 1, (2, 6): 3, (4, 6): 2}
    node_label = NodeArray(graph, init_val=label)
    difference = EdgeArray(graph, init_val=diff)
    res = inc_matrix.T @ node_label
    assert res == difference
