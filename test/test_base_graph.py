import pytest

from grapharray.classes import BaseGraph


class TestBaseGraph:
    bg = BaseGraph()
    bg.add_edges_from([[0, 1], [1, 2]])

    def test_is_node_to_index_not_accessible_before_freeze(self):
        with pytest.raises(ValueError):
            self.bg.node_to_index

    def test_is_edge_to_index_not_accessible_before_freeze(self):
        with pytest.raises(ValueError):
            self.bg.edge_to_index

    def test_are_nodes_edges_to_index_created_when_freeze(self):
        g = BaseGraph([(1, 2), (2, 3)])
        g.freeze()
        assert g.edge_to_index == {(1, 2): 0, (2, 3): 1}
        assert g.node_to_index == {1: 0, 2: 1, 3: 2}
