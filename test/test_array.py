import pytest

import networkx as nx
import numpy as np
import copy
from grapharray.classes import (
    BaseGraph,
    BaseGraphArray,
    NodeArray,
)


def test_is_invalid_type_base_graph_denied():
    with pytest.raises(TypeError):
        BaseGraphArray(1)
    with pytest.raises(TypeError):
        BaseGraphArray(nx.DiGraph())


def test_can_set_item(graph, NodeEdgeArray, dict_init_val):
    index = 4 if (NodeEdgeArray == NodeArray) else (2, 4)
    array = np.zeros(len(dict_init_val))
    array[2] = 5

    tested = NodeEdgeArray(graph)
    tested[index] = 5
    assert np.all(tested.array == array)
    tested = NodeEdgeArray(graph, is_array_2d=True)
    tested[index] = 5
    assert np.all(tested.array == array.reshape((-1, 1)))


def test_can_get_item(graph, NodeEdgeArray, dict_init_val):
    tested = NodeEdgeArray(graph, is_array_2d=False)
    for i, v in dict_init_val.items():
        tested[i] = v
    for k in dict_init_val:
        assert tested[k] == dict_init_val[k]
    tested = NodeEdgeArray(graph, is_array_2d=True)
    for i, v in dict_init_val.items():
        tested[i] = v
    for k in dict_init_val:
        assert tested[k] == dict_init_val[k]


def test_can_init_val_set_with_scalar(graph, NodeEdgeArray):
    init_val = 3.1415
    tested = NodeEdgeArray(graph, init_val=init_val)
    correct = NodeEdgeArray(graph)
    for i in correct.index:
        correct[i] = init_val
    assert tested == correct


def test_can_init_val_set_with_dict(graph, NodeEdgeArray, dict_init_val):
    tested = NodeEdgeArray(graph, init_val=dict_init_val)
    correct = NodeEdgeArray(graph)
    for i in correct.index:
        correct[i] = dict_init_val[i]
    assert tested == correct


def test_can_init_val_set_with_self(graph, NodeEdgeArray):
    original = NodeEdgeArray(graph, init_val=3.1415)
    tested = NodeEdgeArray(graph, init_val=original)
    assert tested == original


def test_can_array_set_vertical(graph, NodeEdgeArray):
    tested = NodeEdgeArray(graph, is_array_2d=True)
    assert tested.array.shape == (len(tested.index), 1)


def test_can_get_values_as_dict(graph, NodeEdgeArray, dict_init_val):
    tested = NodeEdgeArray(graph, init_val=dict_init_val)
    var_dict = tested.as_dict()
    assert var_dict == dict_init_val


def test_can_get_values_as_graph(graph, NodeEdgeArray, dict_init_val):
    tested = NodeEdgeArray(graph, init_val=dict_init_val)
    res_graph = tested.as_nx_graph()
    if NodeEdgeArray == NodeArray:
        for key in dict_init_val:
            assert res_graph.nodes[key]["value"] == dict_init_val[key]
    else:
        for key in dict_init_val:
            assert res_graph.edges[key]["value"] == dict_init_val[key]


def test_disabling(graph, NodeEdgeArray):
    tested = NodeEdgeArray(graph)
    items = copy.copy(tested._enabled_items)
    for i in items:
        tested.disable(i)
        items.remove(i)
        assert tested._enabled_items == items
    with pytest.raises(ValueError) as exc:
        tested.disable(i)
    assert exc.value.args[0] == f"{i} is already disabled."


def test_is_operation_between_different_graphs_denied(graph, NodeEdgeArray):
    another_graph = BaseGraph(graph)
    another_graph.freeze()
    gvar_1 = NodeEdgeArray(graph)
    gvar_2 = NodeEdgeArray(another_graph)
    with pytest.raises(ValueError):
        gvar_1 + gvar_2


@pytest.fixture
def operated_vals(graph, NodeEdgeArray):
    gvar_1 = NodeEdgeArray(graph)
    for ne, i in gvar_1.index.items():
        gvar_1[ne] = 10 * i
    gvar_2 = NodeEdgeArray(graph)
    for ne, i in gvar_1.index.items():
        gvar_2[ne] = (8 - i) * 10
    return gvar_1, gvar_2


def test_is_add_correct(operated_vals, graph, NodeEdgeArray):
    a, b = operated_vals
    tested = a + b
    correct = NodeEdgeArray(graph, init_val=a.array + b.array)
    assert tested == correct
    tested = a + 5
    correct = NodeEdgeArray(graph, init_val=a.array + 5)
    assert tested == correct


def test_is_subtract_correct(operated_vals, graph, NodeEdgeArray):
    a, b = operated_vals
    tested = a - b
    correct = NodeEdgeArray(graph, init_val=a.array - b.array)
    assert tested == correct
    tested = a - 5
    correct = NodeEdgeArray(graph, init_val=a.array - 5)
    assert tested == correct


def test_is_multiply_correct(operated_vals, graph, NodeEdgeArray):
    a, b = operated_vals
    tested = a * b
    correct = NodeEdgeArray(graph, init_val=a.array * b.array)
    assert tested == correct
    tested = a * 5
    correct = NodeEdgeArray(graph, init_val=a.array * 5)
    assert tested == correct


def test_is_true_divide_correct(operated_vals, graph, NodeEdgeArray):
    a, b = operated_vals
    tested = a / b
    correct = NodeEdgeArray(graph, init_val=a.array / b.array)
    assert tested == correct
    tested = a / 4
    correct = NodeEdgeArray(graph, init_val=a.array / 4)
    assert tested == correct


def test_is_matmul_correct(graph, NodeEdgeArray):
    gvar_1 = NodeEdgeArray(graph, init_val=5)
    gvar_2 = NodeEdgeArray(graph, init_val=10)
    assert gvar_1 @ gvar_2 == 50 * len(gvar_1.index)
    gvar_1 = NodeEdgeArray(graph, init_val=5, is_array_2d=True)
    gvar_2 = NodeEdgeArray(graph, init_val=10, is_array_2d=True)
    assert gvar_1.T @ gvar_2 == 50 * len(gvar_1.index)
