import pytest

from grapharray.classes import (
    BaseGraph,
    NodeArray,
    EdgeArray,
)


@pytest.fixture
def graph():
    g = [(0, 2), (0, 4), (2, 4), (2, 6), (4, 6)]
    bg = BaseGraph(g)
    bg.freeze()
    return bg


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
