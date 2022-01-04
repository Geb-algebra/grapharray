from typing import Union
import networkx as nx


def draw(
    grapharray,  # TODO: support nodearray
    pos: dict = None,
    arrows: bool = True,
    with_labels: bool = True,
    connectionstyle: Union[str, None] = None,
    edgelist=None,
    nodelist=None,
    **kwargs,
):
    if edgelist is None:
        edgelist = grapharray.base_graph.edges()
    if nodelist is None:
        nodelist = grapharray.base_graph.nodes()
    draw_graph = nx.DiGraph(grapharray.base_graph)
    edge_color = [grapharray[i] for i in edgelist]
    if connectionstyle is None:
        for e in edgelist:
            if (e[1], e[0]) in edgelist:
                connectionstyle = "arc3, rad=0.2"
                break
        else:
            connectionstyle = "arc3"

    nx.draw_networkx(
        draw_graph,
        pos=pos,
        arrows=arrows,
        with_labels=with_labels,
        connectionstyle=connectionstyle,
        edge_color=edge_color,
        edgelist=edgelist,
        nodelist=nodelist,
        **kwargs,
    )
