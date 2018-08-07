from typing import Optional

import networkx as nx
from pathlib import Path

from . import TIKZ_COLORS as COLORS


def graph2tikz(graph: nx.Graph, background_image: Optional[Path]) -> str:
    lines = []

    from .templates import tikz_main, tikz_background, tikz_edge, tikz_node

    if background_image is not None:
        background = tikz_background.substitute(path=background_image.absolute().with_suffix('.png'))
    else:
        background = ''
    used_nodes = set()
    for node_id, node_data in graph.nodes(data=True):
        used_nodes.add(node_id)
        degree = min(graph.degree[node_id], len(COLORS) - 1)
        node_data['r'] *= 2
        lines.append(tikz_node.substitute(node_data, node_id=node_id, color=COLORS[degree]))

    for edge in graph.edges:
        if edge[0] in used_nodes and edge[1] in used_nodes:
            lines.append(tikz_edge.substitute(node_a=edge[0], node_b=edge[1]))

    return tikz_main.substitute(graph='\n'.join(lines), background=background)