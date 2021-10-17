import pytest

import networkx as nx
from grapharray.classes import BaseGraph


@pytest.fixture
def base_graph():
    return BaseGraph()


def test_are_nodes_edges_sorted():
    g = BaseGraph([(1, 2), (2, 3)])
    assert g.edge_to_index == {(1, 2): 0, (2, 3): 1}
    assert g.node_to_index == {1: 0, 2: 1, 3: 2}


def test_does_adding_one_node_updates_node_to_index(base_graph):
    base_graph.add_node(1)
    assert base_graph.node_to_index == {1: 0}
    base_graph.add_node(2)
    assert base_graph.node_to_index == {1: 0, 2: 1}


def test_does_adding_one_edge_updates_node_and_edge_to_index(base_graph):
    base_graph.add_edge(1, 2)
    assert base_graph.edge_to_index == {(1, 2): 0}
    assert base_graph.node_to_index == {1: 0, 2: 1}
    base_graph.add_edge(2, 3)
    assert base_graph.edge_to_index == {(1, 2): 0, (2, 3): 1}
    assert base_graph.node_to_index == {1: 0, 2: 1, 3: 2}


def test_does_adding_nodes_updates_node_to_index(base_graph):
    base_graph.add_nodes_from((1, 2))
    assert base_graph.node_to_index == {1: 0, 2: 1}


def test_does_adding_edges_updates_node_to_index(base_graph):
    base_graph.add_edges_from(((1, 2), (2, 3)))
    assert base_graph.edge_to_index == {(1, 2): 0, (2, 3): 1}
    assert base_graph.node_to_index == {1: 0, 2: 1, 3: 2}
