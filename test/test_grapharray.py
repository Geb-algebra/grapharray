import pytest
import unittest

import networkx as nx
import numpy as np
from grapharray.classes import (
    BaseGraph,
    BaseGraphArray,
    NodeArray,
    EdgeArray,
    AdjacencyMatrix,
    IncidenceMatrix,
)
from grapharray.functions import exp


def assert_is_array_equal(a, b):
    if a.shape != b.shape:
        pytest.fail(f"Shapes does not match: {a.shape} vs {b.shape}")
    elif np.any(abs(a - b) >= 1e-10):
        failed_index = list(zip(*np.where(abs(a - b) >= 1e-10)))
        message = f"Elements {failed_index} does not equal.\n"
        for i in failed_index:
            message += f" {i}: {a[i]} vs {b[i]}\n"
        pytest.fail(message)


@pytest.fixture
def graph():
    g = [(0, 2), (0, 4), (2, 4), (2, 6), (4, 6)]
    return BaseGraph(nx.DiGraph(g))


@pytest.fixture(params=[NodeArray, EdgeArray])
def NodeEdgeArray(request):
    return request.param


@pytest.fixture
def node_edge_index(graph, NodeEdgeArray):
    if NodeEdgeArray == NodeArray:
        return graph.node_to_index
    else:
        return graph.edge_to_index


@pytest.fixture
def dict_init_val(node_edge_index):
    return {n: 3.1415 * i for i, n in enumerate(node_edge_index)}


def test_is_nodes_edges_sorted(graph):
    gvar = BaseGraphArray(graph)
    assert gvar.nodes == (0, 2, 4, 6)
    assert gvar.edges == ((0, 2), (0, 4), (2, 4), (2, 6), (4, 6))


def test_can_init_val_set_with_scalar(graph, NodeEdgeArray):
    gvar = NodeEdgeArray(graph, init_val=3.1415)
    assert_is_array_equal(gvar.array, np.ones(len(gvar.index)) * 3.1415)


def test_can_init_val_set_with_dict(graph, NodeEdgeArray, dict_init_val):
    gvar = NodeEdgeArray(graph, init_val=dict_init_val)
    assert_is_array_equal(gvar.array, np.array(list(dict_init_val.values())))


def test_can_init_val_set_with_self(graph, NodeEdgeArray):
    original = NodeEdgeArray(graph, init_val=3.1415)
    new = NodeEdgeArray(graph, init_val=original)
    assert_is_array_equal(new.array, original.array)


def test_can_array_set_vertical(graph, NodeEdgeArray):
    gvar = NodeEdgeArray(graph, is_array_2d=True)
    assert gvar.array.shape == (len(gvar.index), 1)


def test_can_get_values_as_dict(graph, NodeEdgeArray, dict_init_val):
    gvar = NodeEdgeArray(graph, init_val=dict_init_val)
    var_dict = gvar.as_dict()
    assert var_dict == dict_init_val


def test_can_get_values_as_graph(graph, NodeEdgeArray, dict_init_val):
    gvar = NodeEdgeArray(graph, init_val=dict_init_val)
    res_graph = gvar.as_nx_graph()
    if NodeEdgeArray == NodeArray:
        for key in dict_init_val:
            assert res_graph.nodes[key]["value"] == dict_init_val[key]
    else:
        for key in dict_init_val:
            assert res_graph.edges[key]["value"] == dict_init_val[key]


def test_can_get_item(graph, NodeEdgeArray, dict_init_val):
    gvar = NodeEdgeArray(graph, init_val=dict_init_val)
    for k in dict_init_val:
        assert gvar[k] == dict_init_val[k]
    gvar = NodeEdgeArray(graph, init_val=dict_init_val, is_array_2d=True)
    for k in dict_init_val:
        assert gvar[k] == dict_init_val[k]


def test_can_set_item_correctly(graph, NodeEdgeArray, dict_init_val):
    index = 4 if NodeEdgeArray == NodeArray else (2, 4)
    array = np.zeros(len(dict_init_val))
    array[2] = 5

    gvar = NodeEdgeArray(graph)
    gvar[index] = 5
    assert_is_array_equal(gvar.array, array)
    gvar = NodeEdgeArray(graph, is_array_2d=True)
    gvar[index] = 5
    assert_is_array_equal(gvar.array, array.reshape((-1, 1)))


@pytest.fixture
def operated_vals(graph, NodeEdgeArray):
    gvar_1 = NodeEdgeArray(graph)
    for ne, i in gvar_1.index.items():
        gvar_1[ne] = 10 * i
    gvar_2 = NodeEdgeArray(graph)
    for ne, i in gvar_1.index.items():
        gvar_2[ne] = (8 - i) * 10
    return gvar_1, gvar_2


def test_is_add_correct(operated_vals):
    a, b = operated_vals
    add = a + b
    assert_is_array_equal(add.array, a.array + b.array)
    add = a + 5
    assert_is_array_equal(add.array, a.array + 5)


def test_is_subtract_correct(operated_vals):
    a, b = operated_vals
    sub = a - b
    assert_is_array_equal(sub.array, a.array - b.array)
    sub = a - 5
    assert_is_array_equal(sub.array, a.array - 5)


def test_is_multiply_correct(operated_vals):
    a, b = operated_vals
    mul = a * b
    assert_is_array_equal(mul.array, a.array * b.array)
    mul = a * 5
    assert_is_array_equal(mul.array, a.array * 5)


def test_is_true_divide_correct(operated_vals):
    a, b = operated_vals
    truediv = a / b
    assert_is_array_equal(truediv.array, a.array / b.array)
    truediv = a / 4
    assert_is_array_equal(truediv.array, a.array / 4)


def test_is_matmul_correct(graph, NodeEdgeArray):
    gvar_1 = NodeEdgeArray(graph, init_val=5)
    gvar_2 = NodeEdgeArray(graph, init_val=10)
    assert gvar_1 @ gvar_2 == 200
    gvar_1 = NodeEdgeArray(graph, init_val=5, is_array_2d=True)
    gvar_2 = NodeEdgeArray(graph, init_val=10, is_array_2d=True)
    assert gvar_1.T @ gvar_2 == 200


@pytest.fixture
def adj_matrix(graph):
    edge_f = {(0, 2): 6, (0, 4): 4, (2, 4): 3, (2, 6): 1, (4, 6): 2}
    weight = EdgeArray(graph, edge_f)
    return AdjacencyMatrix(weight)


def test_is_matrix_correct(adj_matrix) -> None:
    true_matrix = np.array(
        [[0, 6, 4, 0], [0, 0, 3, 1], [0, 0, 0, 2], [0, 0, 0, 0]]
    )
    assert_is_array_equal(adj_matrix.matrix, true_matrix)


def test_is_matmul_correct(adj_matrix):
    nv_val = {0: 1, 2: 2, 4: 3, 6: 4}
    nv = NodeArray(graph, nv_val)
    result = (adj_matrix @ nv).as_dict()
    answer_val = {0: 24, 2: 13, 4: 8, 6: 0}
    assert result == answer_val


@pytest.fixture
def inc_matrix(graph):
    return IncidenceMatrix(graph)


def test_is_matmul_correct(graph, inc_matrix):
    edge_f = {(0, 2): 6, (0, 4): 4, (2, 4): 3, (2, 6): 1, (4, 6): 2}
    od_f = {0: -10, 2: 2, 4: 5, 6: 3}
    edge_flow = EdgeArray(graph, init_val=edge_f)
    od_flow = NodeArray(graph, init_val=od_f)
    matmul_res = inc_matrix @ edge_flow
    assert_is_array_equal(matmul_res.array, od_flow.array)


def test_is_transposed_matmul_correct(graph, inc_matrix):
    label = {0: 0, 2: 2, 4: 3, 6: 5}
    diff = {(0, 2): 2, (0, 4): 3, (2, 4): 1, (2, 6): 3, (4, 6): 2}
    node_label = NodeArray(graph, init_val=label)
    difference = EdgeArray(graph, init_val=diff)
    res = inc_matrix.T @ node_label
    assert_is_array_equal(res.array, difference.array)


@pytest.fixture
def node_edge_array(graph, NodeEdgeArray, dict_init_val):
    return NodeEdgeArray(graph, init_val=dict_init_val)


def test_exp(node_edge_array):
    assert_is_array_equal(
        exp(node_edge_array).array, np.exp(node_edge_array.array)
    )

