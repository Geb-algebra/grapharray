"""Graph Variables
"""

from __future__ import annotations
from typing import Union
import numpy as np
import networkx as nx
from types import MappingProxyType


class BaseGraph(nx.DiGraph):
    """ Directed graph object on which variables are defined.

    This object works
    - as a key to check if variables are defined on the same network and
    - as a holder of the orders of nodes and edges.
    Note that this object itself does not contain any variables.

    Notes: These two attributes are referred from all variables defined on this
        object as read-only properties.
    """

    def __init__(
            self,
            graph: nx.DiGraph,
    ):
        """Run DiGraph.__init__ and setup attributes for defining variables."""
        if not isinstance(graph, nx.DiGraph):
            raise TypeError('graph must be an instance of nx.DiGraph')
        super(BaseGraph, self).__init__(graph)
        nx.freeze(self)
        self._node_to_index = MappingProxyType(
            {node: i for i, node in enumerate(sorted(self.nodes))}
        )  # behave like a read-only dictionary
        self._edge_to_index = MappingProxyType(
            {edge: i for i, edge in enumerate(sorted(self.edges))}
        )

    @property
    def ordered_nodes(self):
        return tuple(self.node_to_index.keys())

    @property
    def ordered_edges(self):
        return tuple(self.edge_to_index.keys())

    @property
    def node_to_index(self):
        return self._node_to_index

    @property
    def edge_to_index(self):
        return self._edge_to_index


class BaseGraphArray:
    """ Base graph variable object.

    This set up attributes used in both node variables and edge variables.
    All attributes are aliases of that of BaseGraph and thus are read-only.
    """

    def __init__(
            self,
            base_graph: BaseGraph,
    ):
        """Store BaseGraph instance on that the variable is defined."""
        self._base_graph = base_graph
        self._is_transposed = False

    @property
    def base_graph(self):
        return self._base_graph

    @property
    def ordered_nodes(self):
        return self.base_graph.ordered_nodes

    @property
    def ordered_edges(self):
        return self.base_graph.ordered_edges

    @property
    def number_of_nodes(self):
        return self.base_graph.number_of_nodes()

    @property
    def number_of_edges(self):
        return self.base_graph.number_of_edges()

    @property
    def nodes(self):
        return self.base_graph.nodes

    @property
    def edges(self):
        return self.base_graph.edges

    def _operation_error_check(self, other, allowed_classes):
        """Error check prior to doing mathematical operations.

        Args:
            other: opponents of the operation.
            allowed_classes: tuple of classes that other allowed to be.

        Returns:

        """
        if not isinstance(other, allowed_classes):
            raise TypeError(
                f'{type(self)} can be operated only with '
                f'{allowed_classes}, not {type(other)}.'
            )
        elif isinstance(other, BaseGraphArray) and \
                id(other.base_graph) != id(self.base_graph):
            raise ValueError(
                f'Cannot compute between variables '
                f'associated with different graphs.'
            )

    def __str__(self):
        return (f"{self.__class__.__name__} object with "
                f"{self.number_of_nodes} nodes and "
                f"{self.number_of_edges} edges.")


class GraphArray(BaseGraphArray):
    """Extracted codes shared between NodeVar and EdgeVar.
    """
    def __init__(
            self,
            base_graph: BaseGraph,
            init_val: any = 0,
            is_array_2dim: bool = False
    ):
        """Set initial value of variables.

        Args:
            base_graph: A BaseGraph object on that the variable is defined.
            init_val: The initial value of the array of variables.
            is_array_2dim:

        Notes:
            init_val must be either scalar-value, NodeVar object,
            {node: value} dictionary, nx.DiGraph with 'value' attributes
            on all nodes.
            if a scalar-value is given, all values are set to init_val.
            if a NodeVar object is given, a copy its array is used
            as initial values.
            if a dictionary or nx.DiGraph is given,
            the value on each node is used
            as initial value of corresponding node.
            if np.ndarray is given, it is directly used as the initial values.
            This is used to make arithmetic operations faster
            by avoiding unnecessary array creation.
        """
        super(GraphArray, self).__init__(base_graph)
        if isinstance(init_val, np.ndarray):
            self.array = init_val
        elif isinstance(init_val, self.__class__):
            self.array = init_val.array
        else:
            self.array = np.ones(len(self.index))
            if isinstance(init_val, (int, float)):
                self.array *= init_val
            elif isinstance(init_val, dict):
                for item, index in self.index.items():
                    self.array[index] = init_val[item]
            else:
                raise TypeError(
                    f'Invalid type of init_val ({type(init_val)}). '
                    f'Init_val must be either '
                    f'{type(self)}, scalar, dict or np.ndarray.'
                )
        self._is_2dim = is_array_2dim
        if is_array_2dim:  # reshape the array to 2-dimension.
            self.array = self.array.reshape((-1, 1))
        self.array_shape = self.array.shape

    def _initialize_array_other_than_array_and_self(self, init_val):
        """Initialize self.array with other types than supported this class.

        This is only a dummy implementation here. You can implement any
        initialization method by creating subclasses and overriding
        this method.
        """
        pass

    @property
    def is_2dim(self):
        return self._is_2dim

    @property
    def is_transposed(self):
        return self._is_transposed

    @property
    def T(self):
        """Transpose the array"""
        self.array = self.array.T
        self._is_transposed = not self._is_transposed
        return self

    @property
    def index(self):
        return {}

    def as_dict(self) -> dict:
        """Return values of variables as a dictionary keyed by node/edge.
        """
        value = {item: self.array[index] for item, index in self.index.items()}
        return value

    def as_nx_graph(self, assign_to=None):
        """
        return a nx.DiGraph with values of variables as node/edge attributes.
        """
        var_dict = self.as_dict()
        res_graph = nx.DiGraph(self.base_graph)
        # todo: avoid using the assign_to argument.
        if assign_to == "node":
            for node, value in var_dict.items():
                res_graph.nodes[node]['value'] = value
        elif assign_to == "edge":
            for edge, value in var_dict.items():
                res_graph.edges[edge]['value'] = value
        else:
            raise ValueError("assign_to must be 'node' or 'edge'")
        return res_graph

    def _create_operation_result_object(self, array):
        """Create an object of the same class as self with given array.

        This is used to create the result object of mathematical operations.

        This is a dummy implementation here. You must implement this to enable
        mathematical operations.

        If there exists any way to create a new object of self.__class__,
        this function is not needed, but I don't know how.
        """
        pass

    def _operation(self, other, operation_func):
        """Arithmetic operation.

        Args:
            other: An instance operated with self.
            operation_func: Either self.__add__, self.__sub__,
                self.__mul__ or self.__truediv__.

        Returns:
            An instance containing the operation result.

        """
        self._operation_error_check(other, (int, float, self.__class__))
        if isinstance(other, (int, float)):
            #  same as res.array = self.array {+, -, * etc.} other
            res_array = operation_func(other)
        else:
            #  same as res.array  = self.array {+, -, * etc.} other.array
            res_array = operation_func(other.array)
        return self._create_operation_result_object(res_array)

    def __getitem__(self, item):
        index = self.index[item]
        if self.is_2dim:
            if self.is_transposed:
                index = (0, index)
            else:
                index = (index, 0)
        return self.array[index]

    def __setitem__(self, item, value):
        index = self.index[item]
        if self.is_2dim:
            if self.is_transposed:
                index = (0, index)
            else:
                index = (index, 0)
        self.array[index] = value

    def __add__(self, other):
        return self._operation(other, self.array.__add__)

    def __sub__(self, other):
        return self._operation(other, self.array.__sub__)

    def __mul__(self, other):
        return self._operation(other, self.array.__mul__)

    def __truediv__(self, other):
        return self._operation(other, self.array.__truediv__)

    def __pow__(self, other):
        return self._operation(other, self.array.__pow__)

    def __matmul__(self, other):
        self._operation_error_check(other, (self.__class__, ))
        return self.array @ other.array

    def __repr__(self):
        var_dict = self.as_dict()
        res = 'index\tvalue\n'
        for index, value in var_dict.items():
            res += f'{index}\t{value}\n'
        return res


class NodeArray(GraphArray):
    """Object of variables defined on the nodes.
    """
    @property
    def index(self):
        return self.base_graph.node_to_index

    def _create_operation_result_object(self, array):
        return NodeArray(self.base_graph,
                         init_val=array,
                         is_array_2dim=self.is_2dim)

    def as_nx_graph(self):
        return super(NodeArray, self).as_nx_graph(assign_to='node')


class EdgeArray(GraphArray):
    """Object of variables defined on the edges.
    """
    @property
    def index(self):
        return self.base_graph.edge_to_index

    def _create_operation_result_object(self, array):
        return EdgeArray(self.base_graph,
                         init_val=array,
                         is_array_2dim=self.is_2dim)

    def as_nx_graph(self):
        return super(EdgeArray, self).as_nx_graph(assign_to='edge')


class BaseMatrix(BaseGraphArray):
    def __init__(
            self,
            base_graph: BaseGraph,
    ):
        """Create incidence matrix."""
        super(BaseMatrix, self).__init__(base_graph)

    @property
    def is_transposed(self):
        return self._is_transposed

    @property
    def T(self):
        """Transpose the array"""
        self.matrix = self.matrix.transpose()
        self._is_transposed = not self._is_transposed
        return self


class AdjacencyMatrix(BaseMatrix):
    """N x N matrix"""

    def __init__(
            self,
            weight: EdgeArray,
            sparse_format: str = 'csr'
    ):
        """Create a matrix

        Args:

        """
        super(AdjacencyMatrix, self).__init__(weight.base_graph)
        self.matrix = nx.to_scipy_sparse_matrix(
            weight.as_nx_graph(),
            nodelist=self.ordered_nodes,
            weight='value',
            format=sparse_format
        )

    def __matmul__(self, other):
        if not isinstance(other, NodeArray):
            raise TypeError(
                f'Adjacency matrix can be multiplied only '
                f'with NodeVar, not {type(other)}.'
            )

        res_array = self.matrix @ other.array
        return NodeArray(self.base_graph, init_val=res_array,
                         is_array_2dim=other.is_2dim)

    def getrow(self, index):
        return self.matrix.getrow(self.base_graph.nodes[index]['index'])

    def getcol(self, index):
        return self.matrix.getcol(self.base_graph.nodes[index]['index'])


class IncidenceMatrix(BaseMatrix):
    """Node-edge incidence matrix that can multiplied with NodeVar and EdgeVar.
    """

    def __init__(
            self,
            base_graph: BaseGraph,
    ):
        """Create incidence matrix."""
        super(IncidenceMatrix, self).__init__(base_graph)
        self.matrix = nx.incidence_matrix(
            base_graph,
            nodelist=self.ordered_nodes,
            edgelist=self.ordered_edges,
            oriented=True
        )

    def __matmul__(self, other):
        """Return the result of self multiplied by NodeVar or EdgeVar.

        Args:
            other:

        Returns:
            NodeVar if not transposed, otherwise EdgeVar.
        """
        if not self._is_transposed:
            type_other = EdgeArray
            type_result = NodeArray
        else:
            type_other = NodeArray
            type_result = EdgeArray

        if not isinstance(other, type_other):
            raise TypeError(
                f'{"transposed "*self.is_transposed}incidence matrix can '
                f'be multiplied only with {str(type_other)}, '
                f'not {type(other)}.'
            )

        res_array = self.matrix @ other.array
        return type_result(self.base_graph, init_val=res_array,
                           is_array_2dim=other.is_2dim)
