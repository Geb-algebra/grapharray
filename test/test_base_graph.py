import pytest

import networkx as nx
from grapharray.classes import BaseGraph


@pytest.fixture
def base_graph():
    return BaseGraph()


def test_are_nodes_edges_to_index_created_when_freeze():
    g = BaseGraph([(1, 2), (2, 3)])
    with pytest.raises(AttributeError):
        g.edge_to_index
    with pytest.raises(AttributeError):
        g.node_to_index
    g.freeze()
    assert g.edge_to_index == {(1, 2): 0, (2, 3): 1}
    assert g.node_to_index == {1: 0, 2: 1, 3: 2}
