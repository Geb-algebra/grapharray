# GraphArray

[![PyPI version](https://badge.fury.io/py/grapharray.svg)](https://badge.fury.io/py/grapharray)
[![Conda version](https://anaconda.org/conda-forge/grapharray/badges/version.svg)](https://anaconda.org/conda-forge/grapharray/)
![Total downloads](https://anaconda.org/conda-forge/grapharray/badges/downloads.svg)

GraphArray is a class of arrays defined on a network, which allows for 
fast computation and easy visualization.

When you plan to code some algorithms on networks, using GraphArray as the class
of control variables provide you easier debugging and summarization of the result
while keeping your calculation fast.

* [Documentation](https://geb-algebra.github.io/grapharray/)

# Simple Example

## Define a graph

```
>>> import grapharray as ga
>>> BG = ga.BaseGraph([("A", "B"), ("B", "C"), ("C", "A")])
>>> BG.freeze()
```

## Arrays for nodes
```
>>> a = ga.NodeArray(BG, init_val=1)   # Define
>>> a
index   value
A       1.0
B       1.0
C       1.0

>>> a["B"]  # Get a specific element using a node index
10.0
>>> a["B"] = 10  # Modify
>>> a
index   value
A       1.0
B       10.0
C       1.0

>>> b = ga.NodeArray(BG, init_val=2)
>>> a+b  # Mathematical operations
index   value
A       3.0
B       12.0
C       3.0
```

## Array for edges
```
>>> a = ga.EdgeArray(BG, init_val=5)   # Define
>>> a
index   value
('A', 'B')      5.0
('B', 'C')      5.0
('C', 'A')      5.0

>>> a["A", "B"]  # You can get and...
5.0
>>> a["A", "B"] = 100  # modify elements with edge indexes
>>> a
index   value
('A', 'B')      100.0
('B', 'C')      5.0
('C', 'A')      5.0

```
Mathematical operations can be done as arrays for nodes.

# Installation

From pypi:
```
pip install grapharray
```
or from conda:
```
conda install -c conda-forge grapharray
```

