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


def compare_array(a, b):
    return np.all(abs(a - b) < 1e-10)


class GraphVarBaseTestCase(unittest.TestCase):
    def setUp(self) -> None:
        #  Tested by the Braess' network
        g = [(0, 2), (0, 4), (2, 4), (2, 6), (4, 6)]
        self.graph = BaseGraph(nx.DiGraph(g))
        self.var_types = ["node", "edge"]
        self.num_nodes_and_edges = {"node": 4, "edge": 5}

    def test_is_order_correct(self):
        gvar = BaseGraphArray(self.graph)
        self.assertEqual(gvar.edges, ((0, 2), (0, 4), (2, 4), (2, 6), (4, 6)))
        self.assertEqual(gvar.nodes, (0, 2, 4, 6))


class NodeArrayTestCase(unittest.TestCase):
    def setUp(self) -> None:
        #  Tested by the Braess' network
        self.edges = [(0, 2), (0, 4), (2, 4), (2, 6), (4, 6)]
        self.nodes = [0, 2, 4, 6]
        self.graph = BaseGraph(nx.DiGraph(self.edges))
        self.num_nodes = 4
        self.num_edges = 5

    def test_is_scalar_init_val_set_correctly(self):
        gvar = NodeArray(self.graph, init_val=3.1415)
        self.assertTrue(
            compare_array(gvar.array, np.ones(self.num_nodes) * 3.1415)
        )

    def test_is_dict_init_val_set_correctly(self):
        weight = np.array([3.1415 * i for i in range(self.num_nodes)])
        init_val = {n: weight[i] for i, n in enumerate(self.nodes)}
        gvar = NodeArray(self.graph, init_val=init_val)

        self.assertTrue(compare_array(gvar.array, weight))

    def test_is_vertical_array_available(self):
        gvar = NodeArray(self.graph, is_array_2d=True)
        self.assertEqual(gvar.array.shape, (4, 1))

    def test_is_var_dict_correct(self):
        weight = np.array([3.1415 * i for i in range(self.num_nodes)])
        init_val = {n: weight[i] for i, n in enumerate(self.nodes)}
        gvar = NodeArray(self.graph, init_val=init_val)
        var_dict = gvar.as_dict()
        for key in init_val:
            self.assertEqual(var_dict[key], init_val[key])

    def test_is_values_correctly_gotten_as_graph(self):
        weight = np.array([3.1415 * i for i in range(self.num_nodes)])
        init_val = {n: weight[i] for i, n in enumerate(self.nodes)}
        gvar = NodeArray(self.graph, init_val=init_val)
        res_graph = gvar.as_nx_graph()
        for key in init_val:
            self.assertEqual(res_graph.nodes[key]["value"], init_val[key])

    def test_can_get_item_correctly(self):
        weight = np.array([3.1415 * i for i in range(self.num_nodes)])
        init_val = {n: weight[i] for i, n in enumerate(self.nodes)}
        gvar = NodeArray(self.graph, init_val=init_val)
        for n in init_val:
            self.assertEqual(gvar[n], init_val[n])
        gvar = NodeArray(self.graph, init_val=init_val, is_array_2d=True)
        for n in init_val:
            self.assertEqual(gvar[n], init_val[n])

    def test_can_set_item_correctly(self):
        gvar = NodeArray(self.graph)
        gvar[2] = 5
        self.assertTrue(compare_array(gvar.array, np.array([0, 5, 0, 0])))
        gvar = NodeArray(self.graph, is_array_2d=True)
        gvar[2] = 5
        self.assertTrue(
            compare_array(gvar.array, np.array([0, 5, 0, 0]).reshape((-1, 1)))
        )

    def make_gvars_for_operation(self):
        gvar_1 = NodeArray(self.graph)
        for i in self.graph.nodes:
            gvar_1[i] = 10 * i
        gvar_2 = NodeArray(self.graph)
        for i in self.graph.nodes:
            gvar_2[i] = (8 - i) * 10
        return gvar_1, gvar_2

    def test_is_add_correct(self):
        a, b = self.make_gvars_for_operation()
        add = a + b
        self.assertTrue(compare_array(add.array, np.ones(4) * 80.0))
        add = a + 5
        self.assertTrue(compare_array(add.array, np.array([5, 25, 45, 65])))

    def test_is_subtract_correct(self):
        a, b = self.make_gvars_for_operation()
        sub = a - b
        self.assertTrue(compare_array(sub.array, np.array([-80, -40, 0, 40])))
        sub = a - 5
        self.assertTrue(compare_array(sub.array, np.array([-5, 15, 35, 55])))

    def test_is_multiply_correct(self):
        a, b = self.make_gvars_for_operation()
        mul = a * b
        self.assertTrue(
            compare_array(mul.array, np.array([0, 1200, 1600, 1200]))
        )
        mul = a * 5
        self.assertTrue(compare_array(mul.array, np.array([0, 100, 200, 300])))

    def test_is_true_divide_correct(self):
        a, b = self.make_gvars_for_operation()
        truediv = a / b
        self.assertTrue(
            compare_array(truediv.array, np.array([0, 1 / 3, 1, 3]))
        )
        truediv = a / 4
        self.assertTrue(compare_array(truediv.array, np.array([0, 5, 10, 15])))

    def test_is_matmul_correct(self):
        gvar_1 = NodeArray(self.graph, init_val=5)
        gvar_2 = NodeArray(self.graph, init_val=10)
        self.assertEqual(gvar_1 @ gvar_2, 200)
        gvar_1 = NodeArray(self.graph, init_val=5, is_array_2d=True)
        gvar_2 = NodeArray(self.graph, init_val=10, is_array_2d=True)
        self.assertEqual(gvar_1.T @ gvar_2, 200)


class EdgeArrayTestCase(unittest.TestCase):
    def setUp(self) -> None:
        #  Tested by the Braess' network
        self.edges = [(0, 2), (0, 4), (2, 4), (2, 6), (4, 6)]
        self.nodes = [0, 2, 4, 6, 8]
        self.graph = BaseGraph(nx.DiGraph(self.edges))
        self.num_nodes = 4
        self.num_edges = 5

    def test_is_scalar_init_val_set_correctly(self):
        gvar = EdgeArray(self.graph, init_val=3.1415)
        self.assertTrue(
            compare_array(gvar.array, np.ones(self.num_edges) * 3.1415)
        )

    def test_is_dict_init_val_set_correctly(self):
        weight = np.array([3.1415 * i for i in range(self.num_edges)])
        init_val = {e: weight[i] for i, e in enumerate(self.edges)}
        gvar = EdgeArray(self.graph, init_val=init_val)

        self.assertTrue(compare_array(gvar.array, weight))

    def test_is_as_dict_correct(self):
        weight = np.array([3.1415 * i for i in range(self.num_edges)])
        init_val = {e: weight[i] for i, e in enumerate(self.edges)}
        gvar = EdgeArray(self.graph, init_val=init_val)
        var_dict = gvar.as_dict()
        for key in init_val:
            self.assertEqual(var_dict[key], init_val[key])

    def test_is_values_correctly_gotten_as_graph(self):
        weight = np.array([3.1415 * i for i in range(self.num_edges)])
        init_val = {e: weight[i] for i, e in enumerate(self.edges)}
        gvar = EdgeArray(self.graph, init_val=init_val)
        res_graph = gvar.as_nx_graph()
        for key in init_val:
            self.assertEqual(res_graph.edges[key]["value"], init_val[key])

    def test_can_get_item_correctly(self):
        weight = np.array([3.1415 * i for i in range(self.num_edges)])
        init_val = {e: weight[i] for i, e in enumerate(self.edges)}
        gvar = EdgeArray(self.graph, init_val=init_val)
        for n in init_val:
            self.assertEqual(gvar[n], init_val[n])
        gvar = EdgeArray(self.graph, init_val=init_val, is_array_2d=True)
        for n in init_val:
            self.assertEqual(gvar[n], init_val[n])

    def test_can_set_item_correctly(self):
        gvar = EdgeArray(self.graph)
        gvar[0, 4] = 5
        self.assertTrue(compare_array(gvar.array, np.array([0, 5, 0, 0, 0])))
        gvar = EdgeArray(self.graph, is_array_2d=True)
        gvar[0, 4] = 5
        self.assertTrue(
            compare_array(
                gvar.array, np.array([0, 5, 0, 0, 0]).reshape((-1, 1))
            )
        )

    def make_gvars_for_operation(self):
        gvar_1 = EdgeArray(self.graph)
        for i, edge in enumerate(self.graph.edges):
            gvar_1[edge] = 10 * i
        gvar_2 = EdgeArray(self.graph)
        for i, edge in enumerate(self.graph.edges):
            gvar_2[edge] = (5 - i) * 10
        return gvar_1, gvar_2

    # todo: add assertion to check type correctness.
    def test_is_add_correct(self):
        a, b = self.make_gvars_for_operation()
        add = a + b
        self.assertTrue(compare_array(add.array, np.ones(5) * 50.0))
        add = a + 5
        self.assertTrue(
            compare_array(add.array, np.array([5, 15, 25, 35, 45]))
        )

    def test_is_subtract_correct(self):
        a, b = self.make_gvars_for_operation()
        sub = a - b
        self.assertTrue(
            compare_array(sub.array, np.array([-50, -30, -10, 10, 30]))
        )
        sub = a - 5
        self.assertTrue(
            compare_array(sub.array, np.array([-5, 5, 15, 25, 35]))
        )

    def test_is_multiply_correct(self):
        a, b = self.make_gvars_for_operation()
        mul = a * b
        self.assertTrue(
            compare_array(mul.array, np.array([0, 400, 600, 600, 400]))
        )
        mul = a * 5
        self.assertTrue(
            compare_array(mul.array, np.array([0, 50, 100, 150, 200]))
        )

    def test_is_true_divide_correct(self):
        a, b = self.make_gvars_for_operation()
        truediv = a / b
        self.assertTrue(
            compare_array(truediv.array, np.array([0, 0.25, 2 / 3, 1.5, 4]))
        )
        truediv = a / 4
        self.assertTrue(
            compare_array(truediv.array, np.array([0, 2.5, 5, 7.5, 10]))
        )


class AdjacencyMatrixTestCase(unittest.TestCase):
    def setUp(self) -> None:
        g = [(0, 2), (0, 4), (2, 4), (2, 6), (4, 6)]
        self.graph = BaseGraph(nx.DiGraph(g))
        self.num_nodes = 4
        self.num_edges = 5
        edge_f = {(0, 2): 6, (0, 4): 4, (2, 4): 3, (2, 6): 1, (4, 6): 2}
        weight = EdgeArray(self.graph, edge_f)
        self.mat = AdjacencyMatrix(weight)

    def test_is_matrix_correct(self) -> None:
        matrix = self.mat.matrix.toarray()
        true_matrix = np.array(
            [[0, 6, 4, 0], [0, 0, 3, 1], [0, 0, 0, 2], [0, 0, 0, 0]]
        )
        self.assertTrue(compare_array(matrix, true_matrix))

    def test_is_matmul_correct(self):
        nv_val = {0: 1, 2: 2, 4: 3, 6: 4}
        nv = NodeArray(self.graph, nv_val)
        result = (self.mat @ nv).as_dict()
        answer_val = {0: 24, 2: 13, 4: 8, 6: 0}
        for node in answer_val:
            self.assertEqual(result[node], answer_val[node])


class IncidenceMatrixTestCase(unittest.TestCase):
    def setUp(self) -> None:
        g = [(0, 2), (0, 4), (2, 4), (2, 6), (4, 6)]
        self.graph = BaseGraph(nx.DiGraph(g))
        self.num_nodes = 4
        self.num_edges = 5
        self.mat = IncidenceMatrix(self.graph)

    def test_is_matmul_correct(self):
        edge_f = {(0, 2): 6, (0, 4): 4, (2, 4): 3, (2, 6): 1, (4, 6): 2}
        od_f = {0: -10, 2: 2, 4: 5, 6: 3}
        edge_flow = EdgeArray(self.graph, init_val=edge_f)
        od_flow = NodeArray(self.graph, init_val=od_f)
        matmul_res = self.mat @ edge_flow
        self.assertTrue(compare_array(matmul_res.array, od_flow.array))

    def test_is_transposed_matmul_correct(self):
        label = {0: 0, 2: 2, 4: 3, 6: 5}
        diff = {(0, 2): 2, (0, 4): 3, (2, 4): 1, (2, 6): 3, (4, 6): 2}
        node_label = NodeArray(self.graph, init_val=label)
        difference = EdgeArray(self.graph, init_val=diff)
        res = self.mat.T @ node_label
        self.assertTrue(compare_array(res.array, difference.array))


class FunctionTestCase(unittest.TestCase):
    def setUp(self) -> None:
        g = [(0, 2), (0, 4), (2, 4), (2, 6), (4, 6)]
        self.graph = BaseGraph(nx.DiGraph(g))
        self.num_nodes = 4
        self.num_edges = 5
        self.node_weight = np.array([3 * i for i in range(self.num_nodes)])
        init_val = {
            n: self.node_weight[i] for i, n in enumerate(self.graph.nodes)
        }
        self.node_var = NodeArray(self.graph, init_val=init_val)
        self.edge_weight = np.array([3 * i for i in range(self.num_edges)])
        init_val = {
            e: self.edge_weight[i] for i, e in enumerate(self.graph.edges)
        }
        self.edge_var = EdgeArray(self.graph, init_val=init_val)

    def test_exp(self):
        self.assertTrue(
            compare_array(exp(self.node_var).array, np.exp(self.node_weight))
        )
        self.assertTrue(
            compare_array(exp(self.edge_var).array, np.exp(self.edge_weight))
        )


if __name__ == "__main__":
    unittest.main()
