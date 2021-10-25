"""Graph Variables
"""

from __future__ import annotations
import numpy as np
import networkx as nx
from types import MappingProxyType


class BaseGraph(nx.DiGraph):
    """ Directed graph object on which arrays are defined.

    This object works (1) as a key to check if variables are defined on
    the same network and (2) as a holder of the orders of nodes and edges.
    Note that this object itself does not contain any variables.

    """

    @property
    def node_to_index(self):
        """Correspondence between nodes and array indices"""
        return self._node_to_index

    @property
    def edge_to_index(self):
        """Correspondence between edges and array indices"""
        return self._edge_to_index

    def freeze(self):
        """Freeze the graph and map between nodes / edges and array indices

            This method must be called before the instance is passed to 
            array initialization methods.
        """
        nx.freeze(self)
        self._node_to_index = MappingProxyType(
            {node: i for i, node in enumerate(self.nodes)}
        )
        self._edge_to_index = MappingProxyType(
            {edge: i for i, edge in enumerate(self.edges)}
        )


class BaseGraphArray:
    """ Base object for creating vectors and matrices on networks.

    This set up attributes used in both node variables and edge variables.
    All attributes are aliases of that of BaseGraph and thus are read-only.
    """

    def __init__(
        self, base_graph: BaseGraph,
    ):
        """Store BaseGraph instance on that the variable is defined."""
        if not isinstance(base_graph, BaseGraph):
            raise TypeError(
                f"BaseGraph must be an instance of BaseGraph, "
                f"not {type(base_graph)}."
            )
        elif not nx.is_frozen(base_graph):
            raise ValueError("base_graph is not freezed.")
        self._base_graph: BaseGraph = base_graph
        self._is_transposed: bool = False
        self._array: np.ndarray = None  # Dummy implementation

    @property
    def array(self):
        """Core array"""
        return self._array.copy()

    @property
    def base_graph(self):
        """BaseGraph object on that this array itself is defined"""
        return self._base_graph

    @property
    def node_to_index(self):
        """Correspondence between nodes and array indices"""
        return self.base_graph.node_to_index

    @property
    def edge_to_index(self):
        """Correspondence between edges and array indices"""
        return self.base_graph.edge_to_index

    @property
    def nodes(self):
        """Tuple of all nodes"""
        return tuple(self.node_to_index.keys())

    @property
    def edges(self):
        """Tuple of all edges"""
        return tuple(self.edge_to_index.keys())

    @property
    def number_of_nodes(self):
        """The number of nodes in the base graph"""
        return self.base_graph.number_of_nodes()

    @property
    def number_of_edges(self):
        """The number of edges in the base graph"""
        return self.base_graph.number_of_edges()

    @property
    def is_transposed(self):
        """Whether the array is transposed or not."""
        return self._is_transposed

    @property
    def T(self):
        """Transpose the array"""
        self._array = self._array.transpose()
        self._is_transposed = not self._is_transposed
        return self

    def _operation_error_check(self, other, allowed_classes):
        """Error check prior to doing mathematical operations.

        Args:
            other: opponents of the operation.
            allowed_classes: tuple of classes that other allowed to be.

        Raises:
            TypeError: when the class of the opponent is not the same as 
                the class of self
            ValueError: when both of self and the opponent are BaseGraphArray
                but they are defined on different base graphs.
        """
        if not isinstance(other, allowed_classes):
            raise TypeError(
                f"{type(self)} can be operated only with "
                f"{allowed_classes}, not {type(other)}."
            )
        elif (
            isinstance(other, BaseGraphArray)
            and other.base_graph is not self.base_graph
        ):
            raise ValueError(
                "Cannot compute between variables "
                "associated with different graphs."
            )

    def __str__(self):
        """Return a string for print function"""
        return (
            f"{self.__class__.__name__} object with "
            f"{self.number_of_nodes} nodes and "
            f"{self.number_of_edges} edges."
        )


class GraphArray(BaseGraphArray):
    """Extracted codes shared between NodeArray and EdgeArray.

    Args:
        base_graph (BaseGraph): The graph on that the variable is defined.
        init_val: The initial value of the array.
        is_array_2d (bool): Whether the array is 2-dimensional column vectors.
            Default is False, which means that the array is 1-dimensional.

    Notes:
        init_val must be either scalar, NodeVar object or
        {node: value} dictionary.
        if a scalar is given, all elements of array are set to the init_val.
        if a NodeVar object is given, a copy of its array is used
        as initial values.
        if a dictionary is given, the value on each node/edge is used
        as initial value of corresponding node/node.
        if np.ndarray is given, it is directly used as the initial values.
        This is used to make arithmetic operations faster
        by avoiding unnecessary array creation.

    Attributes:
        array (np.ndarray): An array that has values linked to nodes/edges.

    """

    def __init__(
        self, base_graph: BaseGraph, init_val=0, is_array_2d: bool = False,
    ):
        """Set the initial value of array."""
        super(GraphArray, self).__init__(base_graph)
        if isinstance(init_val, np.ndarray):
            self._array = init_val
        elif isinstance(init_val, self.__class__):
            self._array = init_val.array
        else:
            self._array = np.zeros(len(self.index))
            if isinstance(init_val, (int, float)):
                self._array += init_val
            elif isinstance(init_val, dict):
                for item, index in self.index.items():
                    self._array[index] = init_val[item]
            else:
                raise TypeError(
                    f"Invalid type of init_val ({type(init_val)}). "
                    f"Init_val must be either "
                    f"{type(self)}, scalar, dict or np.ndarray."
                )
        self._is_2d = is_array_2d
        if is_array_2d:  # reshape the array to 2-dimension.
            self._array = self._array.reshape((-1, 1))

    @property
    def index(self):
        """Correspondence between the array indices and the nodes/edges.

        This is only a dummy implementation here and overridden in subclasses.
        """
        return {}

    @property
    def is_2d(self):
        """Whether the array is 2-dimensional or not"""
        return self._is_2d

    def as_dict(self) -> dict:
        """Return values of variables as a dictionary keyed by node/edge.
        """
        value = {
            item: self._array[index] for item, index in self.index.items()
        }
        return value

    def as_nx_graph(self, assign_to=None):
        """Return a nx.DiGraph with the array elements as node/edge attributes.
        """
        var_dict = self.as_dict()
        res_graph = nx.DiGraph(self.base_graph)
        # todo: avoid using the assign_to argument.
        if assign_to == "node":
            for node, value in var_dict.items():
                res_graph.nodes[node]["value"] = value
        elif assign_to == "edge":
            for edge, value in var_dict.items():
                res_graph.edges[edge]["value"] = value
        else:
            raise ValueError("assign_to must be 'node' or 'edge'")
        return res_graph

    def get_copy(self):
        """Make a copy of self. 
        
        The array of the copy created by this method is a copy of the original,
        while the base_graph of the copy is the same instance of the original.
        This is different from the copy created by copy.deepcopy() in that both
        the array and the base_graph is a copy of the original.
        """
        return type(self)(
            self.base_graph,
            init_val=self._array.copy(),
            is_array_2d=self.is_2d,
        )

    def _operation(self, other, operation_func):
        """Do an arithmetic operation.

        Args:
            other: An instance operated with self.
            operation_func: Magic method for operation, like self.__add__.

        Returns:
            An instance of the same class as self containing the result.

        """
        self._operation_error_check(other, (int, float, self.__class__))
        if isinstance(other, (int, float)):
            #  same as res.array = self.array {+, -, * etc.} other
            res_array = operation_func(other)
        else:
            #  same as res.array  = self.array {+, -, * etc.} other.array
            res_array = operation_func(other._array)
        return type(self)(
            self.base_graph, init_val=res_array, is_array_2d=self.is_2d
        )

    def __add__(self, other):
        """Element-wise addition"""
        return self._operation(other, self._array.__add__)

    def __sub__(self, other):
        """Element-wise subtraction"""
        return self._operation(other, self._array.__sub__)

    def __mul__(self, other):
        """Element-wise multiplication"""
        return self._operation(other, self._array.__mul__)

    def __truediv__(self, other):
        """Element-wise division"""
        return self._operation(other, self._array.__truediv__)

    def __pow__(self, other):
        """Element-wise exponentiation"""
        return self._operation(other, self._array.__pow__)

    def __matmul__(self, other):
        """Inner product of two arrays"""
        self._operation_error_check(other, (self.__class__,))
        return self._array @ other._array

    def __eq__(self, other):
        """Whether all the elements of two arrays are equal"""
        self._operation_error_check(other, (self.__class__,))
        return np.all(self._array == other._array)

    def _get_array_index(self, key):
        """Get the array index corresponding to the specified node or edge"""
        index = self.index[key]
        if self.is_2d:
            if self.is_transposed:
                index = (0, index)
            else:
                index = (index, 0)
        return index

    def __getitem__(self, key):
        """Get the array element corresponding to the specified node or edge"""
        return self._array[self._get_array_index(key)]

    def __setitem__(self, key, value):
        """Set value to the array corresponding to the specified node or edge"""
        self._array[self._get_array_index(key)] = value

    def __repr__(self):
        """Return a string representation of the array"""
        var_dict = self.as_dict()
        res = "index\tvalue\n"
        for index, value in var_dict.items():
            res += f"{index}\t{value}\n"
        return res

    def __len__(self):
        """Return the length of array"""
        return len(self._array)


class NodeArray(GraphArray):
    """Object of variables defined on the nodes."""

    @property
    def index(self):
        """Correspondence between the array indices and the nodes."""
        return self.base_graph.node_to_index

    def as_nx_graph(self):
        """Return a nx.DiGraph with the array elements as its node attributes.
        """
        return super(NodeArray, self).as_nx_graph(assign_to="node")


class EdgeArray(GraphArray):
    """Object of variables defined on the edges."""

    @property
    def index(self):
        """Correspondence between the array indices and the edges."""
        return self.base_graph.edge_to_index

    def as_nx_graph(self):
        """Return a nx.DiGraph with the array elements as its edge attributes.
        """
        return super(EdgeArray, self).as_nx_graph(assign_to="edge")


class AdjacencyMatrix(BaseGraphArray):
    """N x N matrix"""

    def __init__(self, weight: EdgeArray, sparse_format: str = "csr"):
        """Create a matrix

        Args:
            weight: the element values of the matrix. The value of 
                weight[init, term] is set to the (init, term) element of the 
                matrix.
            sparse_format: str in {‘bsr’, ‘csr’, ‘csc’, ‘coo’, ‘lil’, ‘dia’, 
            ‘dok’}
            the format of the sparse matrix.
        """
        super(AdjacencyMatrix, self).__init__(weight.base_graph)
        self._array = nx.to_scipy_sparse_matrix(
            weight.as_nx_graph(),
            nodelist=self.nodes,
            weight="value",
            format=sparse_format,
        )

    def __matmul__(self, other):
        """Return the vector-matrix product as an NodeArray object
        
        The opponent of the operation must be Nodearray object.
        """
        if not isinstance(other, NodeArray):
            raise TypeError(
                f"Adjacency matrix can be multiplied only "
                f"with NodeArray, not {type(other)}."
            )

        res_array = self._array @ other._array
        return NodeArray(
            self.base_graph, init_val=res_array, is_array_2d=other.is_2d
        )


class IncidenceMatrix(BaseGraphArray):
    """Node-edge incidence matrix"""

    def __init__(
        self, base_graph: BaseGraph,
    ):
        """Create incidence matrix."""
        super(IncidenceMatrix, self).__init__(base_graph)
        self._array = nx.incidence_matrix(
            base_graph,
            nodelist=self.nodes,
            edgelist=self.edges,
            oriented=True,
        )

    def __matmul__(self, other):
        """Return the vector-matrix product.
        
        If the matrix is not transposed, the opponent of the operation 
        must be an EdgeArray object and the result is a Nodearray object.
        Otherwise the opponent must be an NodeArray object and the result 
        is EdgeArray.

        Args:
            other: EdgeVar if not transposed, otherwise NodeVar.

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
                f"be multiplied only with {str(type_other)}, "
                f"not {type(other)}."
            )

        res_array = self._array @ other._array
        return type_result(
            self.base_graph, init_val=res_array, is_array_2d=other.is_2d
        )

