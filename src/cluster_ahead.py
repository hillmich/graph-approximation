#!/usr/bin/env python3

"""
Clustering using a custom algorithm focusing on coordinates and doing some look behind to track nodes that are
missing for a given number of  pictures.
"""

import argparse
import pathlib
import itertools
from datetime import datetime
from typing import List
import logging
import collections

import networkx as nx
import numpy as np
import scipy.spatial.distance as ssd

from common import read_gpickle, memorize, LOG_FORMAT

# factor to adjust the allowed distance between two nodes in adjacent graphs
_DISTANCE_FACTOR = 0.5

# metric to use for distance calculation. Mostly the concrete distance is not that important, but just the information
# which distance is shorter. Squared euclidean speeds this up.
_METRIC = 'sqeuclidean'


@memorize
def extract_coordinates(graph: nx.Graph) -> np.ndarray:
    """
    Extracts the coordinates of nodes in a graph to a numpy array, where the first dimension is the node_id - 1.
    :param graph: networkx graph
    :return numpy array containing the coordinates of the nodes in the given graph.
    """

    coordinates = [[d['x'], d['y']] for _, d in sorted(graph.nodes(data=True), key=lambda e: e[0])]
    return np.array(coordinates, dtype=np.float)


def seeded_during_conversion(graphs: List[nx.Graph], index: int, ahead: int, index_node: int, ahead_node: int) -> bool:
    """
    Checks if two nodes were connected during conversion, i.e. if ahead node was seeded by index node.
    :param graphs: list of graphs
    :param index: current index
    :param ahead: current number of steps ahead
    :param index_node: node id of the node at index
    :param ahead_node: node id of the node ahead of the index node
    :return: True if the two nodes were linked during conversion, False otherwise
    """
    current_node = ahead_node
    for between in range(index+ahead, index, -1):
        current_node = graphs[between].nodes[current_node].get('convert_seed', 0)
        if current_node == 0:
            return False
    return current_node == index_node


def annotate_with_temporal_clusters(graphs: List[nx.Graph], look_ahead_steps: int) -> None:
    """
    Annotate the nodes of the graphs in the given list with clusters.

    :param graphs: list of graphs which nodes will be annotated with clusters
    :param look_ahead_steps:
    :return: None
    """
    cluster_ids = itertools.count(1)
    graphs = list(reversed(graphs))

    used_clusters = collections.defaultdict(set)
    max_radii = collections.defaultdict(int)

    link = 0
    zero = 0
    close = 0

    for i, graph in enumerate(graphs):
        graph.graph['clustered_by'] = 'cluster_ahead.py'
        graph.graph['look_ahead_steps'] = look_ahead_steps
        graph.graph['clustered_date'] = datetime.now()

        for node_id, node_data in graph.nodes(data=True):
            max_radii[i] = max(max_radii[i], node_data['r'])
            if node_data.get('cluster', 0) == 0:
                node_data['cluster'] = next(cluster_ids)
                node_data['seed'] = True

            used_clusters[i].add(node_data['cluster'])

        for ahead in range(1, look_ahead_steps + 1):
            if i + ahead >= len(graphs):
                break

            graph_ahead = graphs[i + ahead]
            distances = ssd.cdist(extract_coordinates(graph), extract_coordinates(graph_ahead), _METRIC)
            distances_sorted = np.stack(np.unravel_index(distances.argsort(axis=None), distances.shape), axis=-1)

            for indices in distances_sorted:
                indices = tuple(indices)  # zero indexed
                dist = distances[indices]

                c_node = indices[0] + graph.graph.get('nodes_indexed_by', 1)
                n_node = indices[1] + graph_ahead.graph.get('nodes_indexed_by', 1)

                if dist > max_radii[i] * max_radii[i] * _DISTANCE_FACTOR ** ahead:
                    break

                c_cluster = graph.nodes[c_node]['cluster']
                c_radius = graph.nodes[c_node]['r']

                if c_cluster not in used_clusters[i + ahead] and seeded_during_conversion(graphs, i, ahead, c_node, n_node):
                    graph_ahead.nodes[n_node]['cluster'] = c_cluster
                    graph_ahead.nodes[n_node]['cluster_reason'] = 'conversion link'
                    link += 1
                    used_clusters[i + ahead].add(c_cluster)
                    continue

                if dist == 0 and c_cluster not in used_clusters[i + ahead]:
                    graph_ahead.nodes[n_node]['cluster'] = c_cluster
                    graph_ahead.nodes[n_node]['cluster_reason'] = 'zero distance'
                    zero += 1
                    used_clusters[i + ahead].add(c_cluster)
                    continue

                allowed_distance = c_radius * c_radius * _DISTANCE_FACTOR ** ahead
                # distance wise it gets worse with each iteration, so don't change prior decisions
                if (dist <= allowed_distance
                        and c_cluster not in used_clusters[i + ahead]
                        and graph_ahead.nodes[n_node].get('cluster', 0) == 0):
                    graph_ahead.nodes[n_node]['cluster'] = c_cluster
                    graph_ahead.nodes[n_node]['cluster_reason'] = 'closest'
                    close += 1
                    used_clusters[i + ahead].add(c_cluster)

    logging.debug('cluster count {}'.format(next(cluster_ids)))
    logging.debug('linked: {},  zero dist: {},  closest: {}'.format(link, zero, close))


def fill_gaps_with_virtual_nodes(graphs: List[nx.Graph], max_gap_size: int) -> None:
    """
    Fills nodes that are missing for at most max_gap_size in graphs.

    This is most useful if max_gap_size equals look_ahead_steps - 1.

    :param graphs: list of graphs to be filled
    :param max_gap_size: maximal number of time steps a node may be missing
    :return: None
    """
    if max_gap_size < 1:
        return
    created_vnodes = 0
    graphs = list(reversed(graphs))

    for i, graph in enumerate(graphs):
        for node_id, node_data in graph.nodes(data=True):
            cluster_id = node_data['cluster']
            gap_len = 0
            gap_closed = False
            for ahead in range(1, max_gap_size + 1):
                if i + ahead + 1 >= len(graphs):
                    break

                ahead_graph = graphs[i+ahead]

                if cluster_id in set(nx.get_node_attributes(ahead_graph, 'cluster').values()):
                    gap_closed = True
                    break

                gap_len += 1
            else:
                if cluster_id in set(nx.get_node_attributes(graphs[i + gap_len + 1], 'cluster').values()):
                    gap_closed = True

            if gap_len == 0 or not gap_closed:
                continue
            after_gap_graph = graphs[i + gap_len + 1]
            ag_nid, ag_nd = [(nid, nd) for nid, nd in after_gap_graph.nodes(data=True) if nd['cluster'] == cluster_id][0]

            for gap_i in range(1, gap_len + 1):
                gap_graph = graphs[i + gap_i]
                next_id = max(nid for nid in gap_graph.nodes()) + 1

                w = gap_len + 1

                x = round(((w - gap_i) * node_data['x'] + gap_i * ag_nd['x']) / w)
                y = round(((w - gap_i) * node_data['y'] + gap_i * ag_nd['y']) / w)
                r = ((w - gap_i) * node_data['r'] + gap_i * ag_nd['r']) / w

                gap_graph.add_node(next_id,
                                   x=x,
                                   y=y,
                                   r=r,
                                   cluster=cluster_id,
                                   cluster_reason='gap',
                                   seed=False,
                                   virtual=True)
                created_vnodes += 1
                for neighbor in graph.neighbors(node_id):
                    neighbor_cluster = graph.nodes[neighbor]['cluster']
                    gap_neighbor = [nid for nid, nd in gap_graph.nodes(data=True) if nd['cluster'] == neighbor_cluster]
                    if gap_neighbor:
                        assert len(gap_neighbor) == 1, '{} nodes with cluster {}'.format(len(gap_neighbor), neighbor_cluster)
                        gap_graph.add_edge(next_id, gap_neighbor[0])
    logging.debug('created vnodes: {}'.format(created_vnodes))


def annotate_with_connected_components(graphs: List[nx.Graph]) -> None:
    """
    Iterates through graphs and determines all connected components inside each graph.
    Additionally the connected components are assigned ids and they are tracked across time. If merging occurs the new
    component is assigned the id of the larger parent component (or the lower id if both are the same size).
    :param graphs:
    :return: None
    """
    next_component_id = itertools.count(1)

    for i, graph in enumerate(graphs):
        components = nx.connected_components(graph)

        for nodes in components:
            component_id = next(next_component_id)
            for node in nodes:
                if graph.nodes[node].get('cc', 0) == 0:
                    graph.nodes[node]['cc'] = component_id
                    graph.nodes[node]['cc_seed'] = True
                else:
                    break  # components are always assigned at once

        components = sorted(nx.connected_components(graph), key=len, reverse=True)

        if i == len(graphs) - 1:
            break

        next_graph = graphs[i + 1]
        for nodes in components:
            for node in nodes:
                component_id = graph.nodes[node]['cc']
                cluster_id = graph.nodes[node]['cc']
                next_node_candidate = [nid for nid, ndata in next_graph.nodes(data=True) if ndata['cluster'] == cluster_id]

                assert len(next_node_candidate) < 2
                if len(next_node_candidate) == 0:
                    continue

                next_component_set = nx.node_connected_component(next_graph, next_node_candidate[0])
                for node in next_component_set:
                    if next_graph.nodes[node].get('cc', 0) == 0:
                        next_graph.nodes[node]['cc'] = component_id
                        next_graph.nodes[node]['cc_seed'] = False
                break


def sanity_check(graphs: List[nx.Graph]) -> bool:
    """
    Perform some checks to increase confidence in the annotation.

    - every cluster appears at most once per graph
    - every node in a connected component has the same component id

    :param graphs: list of annotated graphs
    :return: True if the checks passed, False otherwise
    """
    return_value = True
    for graph in graphs:
        cluster_set = set()
        for node_id, node_data in graph.nodes(data=True):
            cluster = node_data['cluster']
            if cluster not in cluster_set:
                cluster_set.add(cluster)
            else:
                logging.error('cluster {} found multiple times in graph {}'.format(cluster, graph.name))
                return_value = False

        for node_set in nx.connected_components(graph):
            component_ids = set(graph.nodes[nid]['cc'] for nid in node_set)
            if len(component_ids) != 1:
                logging.error('connected component with multiple ids {} in graph {}'.format(component_ids, graph.name))

    return return_value


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('crop_dir', metavar='crop-dir',
                        help='path to folder containing crop')
    parser.add_argument('--output-dir', dest='output_dir', default='clustered_ahead',
                        help='path to the folder where the annotated graphs will be saved (in a subfolder)')
    parser.add_argument('--skip', type=int, default=0, metavar='N',
                        help='skip the first N graphs')
    parser.add_argument('--debug', action='store_true',
                        help='more detailed information during processing')
    parser.add_argument('--no-sanity-check', dest='no_sanity_check', action='store_true',
                        help='skip the sanity check after annotating')
    parser.add_argument('--look-ahead-steps', dest='look_ahead_steps', default=0, type=int,
                        help='how many steps to look past the next graph for clusters (defaults to 0)')
    parser.add_argument('--max-gap-fill', dest='max_gap_fill', default=0, type=int,
                        help='fill gaps between clusters up to this size (defaults to 0)')
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)
    else:
        logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    crop_dir = pathlib.Path(args.crop_dir)
    output_dir = pathlib.Path(args.output_dir)
    if not crop_dir.exists() or not crop_dir.is_dir():
        raise ValueError('argument \'{}\' is not a directory'.format(args.crop_dir))

    start_time = datetime.now()

    logging.info('Starting {} at {}'.format(crop_dir.stem, start_time))
    files = sorted(crop_dir.glob('*.pickle'))[args.skip:]
    logging.info('Found {:>4} pickle files in {} after skipping the first {} files.'
                 .format(len(files), crop_dir.name, args.skip))
    results = [read_gpickle(gpickle) for gpickle in files]

    logging.debug('Clustering nodes looking ahead {}'.format(args.look_ahead_steps))
    annotate_with_temporal_clusters(results, args.look_ahead_steps)

    logging.debug('Bridging gaps up to size {}'.format(args.max_gap_fill))
    fill_gaps_with_virtual_nodes(results, args.max_gap_fill)

    logging.debug('Finding and annotating connected components')
    annotate_with_connected_components(results)

    if not args.no_sanity_check:
        if sanity_check(results):
            logging.info('sanity check passed')
        else:
            logging.warning('sanity check FAILED')

    output_dir.joinpath(crop_dir.name).mkdir(parents=True, exist_ok=True)
    for graph in results:
        filename = str(output_dir.joinpath(crop_dir.name).joinpath(graph.name).with_suffix('.pickle'))
        nx.write_gpickle(graph, filename)

    end_time = datetime.now()
    logging.info('finished {} at {} after {}'.format(crop_dir.stem, end_time, end_time-start_time))


if __name__ == '__main__':
    main()
