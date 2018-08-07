#!/usr/bin/env python3

"""
Read clustered data and do statistics for
- plots for the probability of node degrees-
- plots and additional text files for clusters and gaps
- CVS file for events and degrees
- plots for events up to a given degree

The produced pgf files expect a length elen defined in LaTeX!
"""

import argparse
from collections import OrderedDict, Counter
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, NamedTuple, Optional
import logging
import collections
import itertools
import math
import pickle

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import scipy.signal as ssig
import numpy as np

from common import read_gpickle, LOG_FORMAT

EventType = Tuple[int, int, int, int]
TimeStep = int
ClusterId = int
FIGURE_SIZE = (5.6, 3.1)  # in inches
INTERVAL = 1  # seconds per time step
TIME_AXIS = 'time steps'


class EventInfo(NamedTuple):
    """Saving information about an event to keep track and allow easier sorting"""
    is_new: bool
    u: ClusterId
    v: ClusterId
    stability: int
    stability_degree: int


def pretty_event(event: EventType, nonexisting: str = r'\varnothing') -> str:
    """
    Takes an event type an prettifies it.

    :param event: event type to prettify
    :param nonexisting: string to represent non-existing nodes
    :return: prettified event type as string
    """
    stringified_event = ((str(d) if d >= 0 else nonexisting) for d in event)
    return r'({},{})->({},{})'.format(*stringified_event)


def pretty_event_latex(event: EventType, nonexisting: str = r'\varnothing') -> str:
    """
    Takes an event type an prettifies it.

    :param event: event type to prettify
    :param nonexisting: string to represent non-existing nodes
    :return: prettified event type as string \makebox[\elen][c]{{{}}}
    """
    stringified_event = ((str(d) if d >= 0 else nonexisting) for d in event)
    return r'\settowidth{{\elen}}{{${}$}} $(\mathmakebox[\elen][c]{{{}}},\mathmakebox[\elen][c]{{{}}})\to (\mathmakebox[\elen][c]{{{}}},\mathmakebox[\elen][c]{{{}}})$'.format(
        nonexisting, *stringified_event)


def get_degree(graph: nx.Graph, cmap: Dict[int, int], cluster: ClusterId) -> int:
    """
    Gets the degree of a node in graph.

    :param graph: Graph which contains the node of which you want the degree
    :param cmap: map from cluster id to node id
    :param cluster: cluster id of the node whose degree you want
    :return: degree of the node or -1 if the node does not exist
    """
    try:
        return graph.degree[cmap[cluster]]
    except KeyError:
        return -1


def count_real_events(graphs: List[nx.Graph]) -> Tuple[
    Dict[EventType, Dict[TimeStep, int]], Dict[TimeStep, Dict[EventType, int]]]:
    """
    Count the events occurring in the given list of graphs.

    :param graphs: list of graphs
    :return: Tuple of dictionaries
    """
    events_by_time = {}
    events_by_type = collections.defaultdict(lambda: collections.defaultdict(int))

    for i, current_graph, next_graph in zip(itertools.count(0), graphs, graphs[1:]):
        events_by_time[i] = collections.defaultdict(int)
        current_edges = frozenset(
            tuple(sorted([current_graph.nodes[u]['cluster'], current_graph.nodes[v]['cluster']])) for u, v in
            current_graph.edges)
        next_edges = frozenset(
            tuple(sorted([next_graph.nodes[u]['cluster'], next_graph.nodes[v]['cluster']])) for u, v in
            next_graph.edges)

        current_c2n = {node_data['cluster']: node_id for node_id, node_data in current_graph.nodes(data=True)}
        next_c2n = {node_data['cluster']: node_id for node_id, node_data in next_graph.nodes(data=True)}

        raw_events = current_edges.symmetric_difference(next_edges)

        for u, v in raw_events:
            degree_u = get_degree(current_graph, current_c2n, u)
            degree_v = get_degree(current_graph, current_c2n, v)
            next_degree_u = get_degree(next_graph, next_c2n, u)
            next_degree_v = get_degree(next_graph, next_c2n, v)

            if degree_u < degree_v:
                event = (degree_u, degree_v, next_degree_u, next_degree_v)
            else:
                event = (degree_v, degree_u, next_degree_v, next_degree_u)

            events_by_time[i][event] += 1
            events_by_type[event][i] += 1

    logging.info('found {} event types'.format(len(events_by_type.keys())))

    return dict(events_by_type), dict(events_by_time)


def count_virtual_events(graphs: List[nx.Graph]) -> Tuple[
    Dict[EventType, Dict[TimeStep, int]], Dict[TimeStep, Dict[EventType, int]]]:
    """
    Count the events occurring in the given graph list and process them to simulate that no events occur at the same
    time. This reduces the number of different event types.

    :param graphs: list of graphs
    :return: Tuple of dictionaries
    """
    events_by_time = {}
    events_by_type = collections.defaultdict(lambda: collections.defaultdict(int))

    for i, current_graph_original, next_graph in zip(itertools.count(0), graphs, graphs[1:]):
        events_by_time[i] = collections.defaultdict(int)
        current_graph = current_graph_original.copy()
        current_edges = frozenset(
            tuple(sorted([current_graph.nodes[u]['cluster'], current_graph.nodes[v]['cluster']])) for u, v in
            current_graph.edges)
        next_edges = frozenset(
            tuple(sorted([next_graph.nodes[u]['cluster'], next_graph.nodes[v]['cluster']])) for u, v in
            next_graph.edges)

        current_c2n = {cluster_id: node_id for node_id, cluster_id in current_graph.nodes(data='cluster')}
        next_c2n = {cluster_id: node_id for node_id, cluster_id in next_graph.nodes(data='cluster')}

        edge_difference = current_edges.symmetric_difference(next_edges)
        event_list = []

        for u, v in edge_difference:
            is_new = (u, v) in next_edges
            uc = u in current_c2n
            un = u in next_c2n
            vc = v in current_c2n
            vn = v in next_c2n

            current_degree_u = get_degree(current_graph, current_c2n, u) + 1
            current_degree_v = get_degree(current_graph, current_c2n, v) + 1
            next_degree_u = get_degree(next_graph, next_c2n, u) + 1
            next_degree_v = get_degree(next_graph, next_c2n, v) + 1

            stability_degree = -(abs(current_degree_u - next_degree_u) + abs(current_degree_v - next_degree_v))
            if is_new:
                stability = uc + un + vc + vn
            else:
                stability = 6 - (uc + un + vc + vn)

            event_list.append(EventInfo(is_new, u, v, stability, stability_degree))

        while event_list:
            event_list.sort(key=lambda e: (e.stability, e.stability_degree), reverse=True)
            event = event_list.pop()

            current_degree_u = get_degree(current_graph, current_c2n, event.u)
            current_degree_v = get_degree(current_graph, current_c2n, event.v)
            next_degree_u = get_degree(next_graph, next_c2n, event.u)
            next_degree_v = get_degree(next_graph, next_c2n, event.v)

            if event.is_new:
                min_c = min(current_degree_u, current_degree_v)
                max_c = max(current_degree_u, current_degree_v)
                event_type = (min_c, max_c, min_c + 1 if min_c > 0 else 1, max_c + 1 if max_c > 0 else 1)
            else:
                min_c = min(current_degree_u, current_degree_v)
                max_c = max(current_degree_u, current_degree_v)

                if current_degree_u < current_degree_v:
                    event_type = (
                    min_c, max_c, min_c - 1 if next_degree_u != -1 else -1, max_c - 1 if next_degree_v != -1 else -1)
                else:
                    event_type = (
                    min_c, max_c, min_c - 1 if next_degree_v != -1 else -1, max_c - 1 if next_degree_u != -1 else -1)

            events_by_time[i][event_type] += 1
            events_by_type[event_type][i] += 1

            # update the copy of current_graph

            current_node_u = current_c2n.get(event.u, 0)
            current_node_v = current_c2n.get(event.v, 0)
            next_node_u = next_c2n.get(event.u, 0)
            next_node_v = next_c2n.get(event.v, 0)

            if event.is_new:
                if not current_node_u:
                    current_node_u = max(current_graph.nodes()) + 1
                    current_graph.add_node(current_node_u, cluster=event.u, substep=True)
                if not current_node_v:
                    current_node_v = max(current_graph.nodes()) + 1
                    current_graph.add_node(current_node_v, cluster=event.v, substep=True)
                current_graph.add_edge(current_node_u, current_node_v, substep=True)
            else:
                current_graph.remove_edge(current_node_u, current_node_v)
                if not next_node_u and len(current_graph[current_node_u]) == 0:
                    current_graph.remove_node(current_node_u)
                if not next_node_v and len(current_graph[current_node_v]) == 0:
                    current_graph.remove_node(current_node_v)

            current_c2n = {cluster_id: node_id for node_id, cluster_id in current_graph.nodes(data='cluster')}
            next_c2n = {cluster_id: node_id for node_id, cluster_id in next_graph.nodes(data='cluster')}

            new_event_list = []
            for e in event_list:
                uc = e.u in current_c2n
                un = e.u in next_c2n
                vc = e.v in current_c2n
                vn = e.v in next_c2n

                current_degree_u = get_degree(current_graph, current_c2n, e.u) + 1
                current_degree_v = get_degree(current_graph, current_c2n, e.v) + 1
                next_degree_u = get_degree(next_graph, next_c2n, e.u) + 1
                next_degree_v = get_degree(next_graph, next_c2n, e.v) + 1

                stability_degree = -(abs(current_degree_u - next_degree_u) + abs(current_degree_v - next_degree_v))
                if e.is_new:
                    stability = uc + un + vc + vn
                else:
                    stability = 6 - (uc + un + vc + vn)

                new_event_list.append(EventInfo(e.is_new, e.u, e.v, stability, stability_degree))
            event_list = new_event_list

    logging.info('event types: {}'.format(len(events_by_type.keys())))
    logging.info('total events: {}'.format(sum(sum(v.values()) for v in events_by_type.values())))

    return dict(events_by_type), dict(events_by_time)


def generate_event_csv(graphs: List[nx.Graph], events_by_time: Dict[TimeStep, Dict[EventType, int]],
                       output_dir: Path) -> None:
    """
    Generate a large CSV file where each row is a timestep and the columns contain frequency of events and node degrees

    :param graphs: list of graphs
    :param events_by_time: maps a timestep to event type and frequency
    :param output_dir: path to the top level folder where the output will be written, more subfolders will be created
    :return: None
    """
    all_events = sorted({event for dicts in events_by_time.values() for event in dicts.keys()})
    max_degree = max(len(nx.degree_histogram(g)) for g in graphs)

    with output_dir.joinpath('events.csv').open('w') as csv:

        print('ts; {events_total}; {events_by_type}; {nodes_total}; {nodes_by_degree}'.format(
            events_total='events_total',
            events_by_type='; '.join(str(event) for event in all_events),
            nodes_total='nodes_total',
            nodes_by_degree='; '.join('degree_{}'.format(i) for i in range(max_degree))
        ), file=csv)

        for ts in range(len(events_by_time)):
            events = events_by_time.get(ts, {})
            print('{}; '.format(ts + 1), end='', file=csv)
            print('{}; '.format(sum(e for e in events.values())), end='', file=csv)
            for event in all_events:
                print('{}; '.format(events.get(event, 0)), end='', file=csv)
            degree_hist = nx.degree_histogram(graphs[ts])
            print('{}; '.format(graphs[ts].number_of_nodes()), end='', file=csv)
            for degree in range(max_degree):
                print('{}; '.format(degree_hist[degree] if len(degree_hist) > degree else 0), end='', file=csv)

            print(file=csv)


def generate_event_plots(graphs: List[nx.Graph], events_by_type: Dict[EventType, Dict[TimeStep, int]],
                         max_degree: Optional[int], output_dir: Path) -> None:
    """
    Plots the occurrence of events for each time step.

    :param graphs: list of graphs
    :param events_by_type: dict mapping the type of an event to timesteps and frequency
    :param max_degree: maximum degree up to which plots will be generated
    :param output_dir: path to the top level folder where the output will be written, more subfolders will be created
    :return: None
    """
    if max_degree is None:
        return

    for event in (event for event in events_by_type.keys() if event[1] <= max_degree):
        event_array = np.zeros((len(graphs),), dtype=np.int)
        for i in events_by_type[event]:
            event_array[i] = events_by_type[event][i]
        fig = plt.figure(figsize=FIGURE_SIZE)
        ax = fig.gca()
        ax.scatter(np.arange(len(graphs), dtype=np.int), event_array, s=1, color='green')
        ax.set_xlabel('time')
        ax.set_ylabel('count')
        fig.savefig(str(output_dir.joinpath('event-{}.pdf'.format(event))), bbox_inches='tight')
        fig.clf()
        plt.close(fig)


def generate_events_by_time_plot(graphs: List[nx.Graph], events_by_time: Dict[TimeStep, Dict[EventType, int]],
                                 output_dir: Path) -> None:
    """
    Plots the number of events per timestep.

    :param graphs: list of graphs
    :param events_by_time: dict grouping events first by timestep then by event type
    :param output_dir: path to the folder where the images will be saved to
    :return: None
    """

    event_counts = np.array([sum(d.values()) for d in OrderedDict(sorted(events_by_time.items())).values()])
    node_counts = np.array([graph.number_of_nodes() for graph in graphs])

    event_node_count = event_counts / node_counts[:-1]
    fig, ax = plt.subplots()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: r'\num{{{:g}}}'.format(x * INTERVAL)))
    ax.plot(event_node_count, 'green', alpha=0.3)
    ax.plot(ssig.savgol_filter(event_node_count, min(101, len(event_node_count) - 3) | 1, 3, mode='nearest'), 'green',
            alpha=1,
            label='events per node')
    ax.grid()
    ax.set_xlabel(TIME_AXIS)
    ax.set_ylabel('events per node')
    fig.savefig(str(output_dir.joinpath('event-bytime.pdf')), bbox_inches='tight')
    fig.savefig(str(output_dir.joinpath('event-bytime.pgf')), bbox_inches='tight')
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.02, 1, 0.102))
    ax_nodes = ax.twinx()
    ax_nodes.plot(ssig.savgol_filter(event_counts, min(101, len(event_counts) - 3) | 1, 3, mode='nearest'), 'blue',
                  alpha=1,
                  label='number of events')
    ax_nodes.plot(ssig.savgol_filter(node_counts, min(101, len(node_counts) - 3) | 1, 3, mode='nearest'), 'black',
                  alpha=1,
                  label='number of nodes')
    ax_nodes.legend(loc='lower right', bbox_to_anchor=(0, 1.02, 1, 0.102))
    fig.savefig(str(output_dir.joinpath('event-bytime-extended.pdf')), bbox_inches='tight')
    fig.savefig(str(output_dir.joinpath('event-bytime-extended.pgf')), bbox_inches='tight')
    fig.clf()
    plt.close(fig)


def generate_event_histograms(events_by_type: Dict[EventType, Dict[TimeStep, int]], cut_off: float,
                              output_dir: Path) -> None:
    """
    Plots how often the events occur in relation to the total number of events.

    :param events_by_type: dict mapping the type of an event to timesteps and frequency
    :param min_count: minimum occurrence of event to not be excluded
    :param output_dir: path to the folder where the images will be saved to
    :return: None
    """

    num_events = sum(sum(d.values()) for d in events_by_type.values())

    events = OrderedDict(sorted({event_type: sum(d.values()) for event_type, d in events_by_type.items()}.items(),
                                key=lambda t: t[1]))
    min_count = int(num_events * cut_off)

    events_squashed = OrderedDict(sorted({event_type: sum(d.values()) for event_type, d in events_by_type.items() if
                                          sum(d.values()) >= min_count}.items(),
                                         key=lambda t: t[1]))
    num_squashed_events = sum(events_squashed.values())

    logging.debug('kept {} events of a total of {} events with threshold {} (that\'s {:.4f} of all events)'
                  .format(num_squashed_events, num_events, min_count, num_squashed_events / num_events))

    event_array = np.zeros((len(events_squashed),), dtype=np.int)
    event_names = {}

    for i, (event, count) in enumerate(events_squashed.items()):
        event_names[i] = pretty_event_latex(event)
        event_array[i] = count
    index = np.arange(len(events_squashed), dtype=np.int)

    fig, ax_c = plt.subplots(figsize=(5.5, 9))
    ax_bar = ax_c.twiny()
    ax_c.set_zorder(10)
    ax_c.patch.set_visible(False)
    ax_bar.set_xlim(0, 0.11)
    ax_c.set_xlim(0, 1.1)

    ax_bar.barh(index, event_array / num_events, color='green', zorder=5)
    ax_bar.set_xlabel('percentage')
    ax_c.set_ylabel('{} of {} event types'.format(len(events_squashed), len(events)))
    ax_bar.set_yticks(index)
    if len(events_squashed) < 70:
        ax_bar.set_yticklabels(event_names.values(), fontsize=8)
    else:
        ax_bar.set_yticklabels([])
    with output_dir.joinpath('event-histogram.txt').open('w') as out_file:
        for event, event_prop in reversed(list(zip(events_squashed.keys(), event_array / num_events))):
            out_file.write('{}: {:.3f}\n'.format(pretty_event(event, 'X'), event_prop))

    event_count = list(reversed(events.values()))
    event_percent = np.zeros((len(events),), dtype=np.float)
    event_percent[0] = list(event_count)[0]

    for i, count in enumerate(event_count[1:], start=1):
        event_percent[i] = count + event_percent[i - 1]

    event_percent /= num_events
    index = np.flipud(np.arange(len(events), dtype=np.int)[:len(events_squashed)])

    color = 'tab:red'
    ax_c.plot(event_percent[:len(events_squashed)], index, color=color, zorder=10)
    ax_c.tick_params(axis='x', labelcolor=color)
    ax_c.set_xlabel('cumulative percentage', color=color)
    plt.grid(axis='x', zorder=7)

    fig.savefig(str(output_dir.joinpath('event-histogram.pdf')), bbox_inches='tight')
    fig.savefig(str(output_dir.joinpath('event-histogram.pgf')), bbox_inches='tight')
    fig.clf()
    plt.close(fig)


def generate_stats_plots(graphs: List[nx.Graph], output_dir: Path) -> None:
    """
    Plots probability of node degrees against time and against other probabilities.

    :param graphs: list of graphs
    :param output_dir: path to the folder where the images will be saved to
    :return: None
    """
    nodes = np.empty((len(graphs),), dtype=int)
    stats = np.empty((len(graphs), 6), dtype=float)
    additional = np.empty((len(graphs), 1), dtype=float)

    for i, graph in enumerate(graphs):
        N = graph.number_of_nodes()
        hist = nx.degree_histogram(graph)

        stats[i] = [a / N for a, _ in itertools.zip_longest(hist[:5], [0] * 5, fillvalue=0.0)] + [sum(hist[5:]) / N]
        additional[i] = [sum(dict(graph.degree).values()) / graph.number_of_nodes()]
        nodes[i] = N

    colors = ['red', 'green', 'gray', 'blue', 'orange', 'purple']

    fig, ax_p = plt.subplots(figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1] + 0.1))
    ax_n = ax_p.twinx()

    for i, row in enumerate(stats.T):
        ax_p.plot(stats[:, i], colors[i], alpha=0.3)
        ax_p.plot(ssig.savgol_filter(stats[:, i], min(101, len(graphs) - 3) | 1, 3, mode='nearest'), colors[i], alpha=1,
                  label=r'$p_{}$'.format(i if i < 5 else r'{\geqslant 5}'))

    ax_p.legend(loc='lower left', ncol=3, bbox_to_anchor=(0, 1.02, 1, 0.102))
    ax_p.set_xlabel(TIME_AXIS)
    ax_p.set_ylabel('probability')
    ax_p.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: r'\num{{{:g}}}'.format(x * INTERVAL)))

    ax_n.plot(nodes, 'black', alpha=0.3)
    ax_n.plot(ssig.savgol_filter(nodes, min(101, len(graphs) - 3) | 1, 3, mode='nearest'), 'black',
              alpha=1, label='number of nodes')

    ax_n.legend(loc='lower right', bbox_to_anchor=(0, 1.02, 1, 0.102))
    ax_n.set_ylabel('number of nodes')
    ax_n.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: r'\num{{{:g}}}'.format(x)))

    fig.savefig(str(output_dir.joinpath('nodes-probability.pdf')), bbox_inches='tight')
    fig.savefig(str(output_dir.joinpath('nodes-probability.pgf')), bbox_inches='tight')
    fig.clf()
    plt.close(fig)

    norm = mcolors.Normalize(vmin=np.min(additional[:, 0]), vmax=np.max(additional[:, 0]))

    fig = plt.figure(figsize=FIGURE_SIZE)
    ax = fig.gca()
    ax.set_ylim([0, 15])
    with np.errstate(divide='ignore', invalid='ignore'):
        p_34 = stats[:, 3] / stats[:, 4]
        p_04 = stats[:, 0] / stats[:, 4]

    ax.scatter(p_04, p_34, s=1, c=norm(additional[:, 0]), cmap=cm.jet)
    ax.set_xscale('log')
    ax.set_ylabel(r'$p=\frac{p_3}{p_4}$')
    ax.set_xlabel(r'$q=\frac{p_0}{p_4}$')
    scalar_map = cm.ScalarMappable(norm=norm, cmap=cm.jet)
    scalar_map.set_array(additional[:, 0])
    fig.colorbar(scalar_map).set_label(r'$\langle k\rangle$')
    fig.savefig(str(output_dir.joinpath('nodes-scatter.pdf')), bbox_inches='tight')
    fig.clf()
    plt.close(fig)


def generate_cluster_stats(graphs: List[nx.Graph], output_dir: Path) -> None:
    """
    Generate statistics regarding the clusters and their gaps of a crop

    :param graphs: list of graphs
    :param output_dir: path to the folder where the images will be saved to
    :return: None
    """
    picture_path = output_dir.joinpath('nodes-cluster.pdf')

    cluster_graphs = collections.defaultdict(set)
    clusters_by_size = collections.defaultdict(int)
    clusters_by_gaps = collections.defaultdict(int)
    num_all_nodes = 0

    for i, graph in enumerate(graphs):
        for node_id, node_data in graph.nodes(data=True):
            graph_id = graph.graph.get('graph_id', i)
            cluster_graphs[node_data['cluster']].add(graph_id)
        num_all_nodes += graph.number_of_nodes()

    with picture_path.with_suffix('.txt').open('w') as f:
        for cluster, graph_set in sorted(cluster_graphs.items()):
            clusters_by_size[len(graph_set)] += 1
            gaps = max(graph_set) - min(graph_set) - len(graph_set) + 1
            clusters_by_gaps[gaps] += 1

            groups = (list(x) for _, x in
                      itertools.groupby(sorted(graph_set), lambda x, c=itertools.count(): next(c) - x))

            print('{:>5}: '.format(cluster), ','.join('-'.join(map(str, (g[0], g[-1])[:len(g)])) for g in groups),
                  file=f)

    clusters_by_size = collections.OrderedDict(sorted(clusters_by_size.items()))
    x = np.array(list(clusters_by_size.keys()), dtype=int)
    y = np.array(list(clusters_by_size.values()), dtype=int)

    fig = plt.figure(figsize=FIGURE_SIZE)
    ax = fig.gca()
    ax.set_xlabel('no of nodes in cluster')
    ax.set_ylabel('number of clusters')
    ax.set_yscale('log')
    ax.scatter(x, y, s=2, c='green')

    fig.savefig(str(picture_path), bbox_inches='tight')
    fig.savefig(str(picture_path.with_suffix('.pgf')), bbox_inches='tight')
    fig.clf()
    plt.close(fig)

    cluster_sizes_cumulated = np.cumsum(
        np.array(list(sorted([len(cluster_set) for cluster_set in cluster_graphs.values()], reverse=True)), dtype=float)
        / num_all_nodes
    )
    cluster_percentage = np.arange(len(cluster_sizes_cumulated), dtype=float) / len(cluster_sizes_cumulated)

    fig = plt.figure(figsize=FIGURE_SIZE)
    ax = fig.gca()
    ax.grid()
    ax.set_xlabel('percentage of temporal clusters')
    ax.set_ylabel('percentage of nodes')
    ax.plot(cluster_percentage, cluster_sizes_cumulated, color='green')

    fig.savefig(str(output_dir.joinpath('nodes-cluster-relative.pdf')), bbox_inches='tight')
    fig.savefig(str(output_dir.joinpath('nodes-cluster-relative.pgf')), bbox_inches='tight')
    fig.clf()
    plt.close(fig)


def generate_event_heatmaps(events_by_type: Dict[EventType, Dict[TimeStep, int]],
                            events_by_time: Dict[TimeStep, Dict[EventType, int]], cut_off: float,
                            output_dir: Path) -> None:
    """
    Generates two heatmaps for the given event dict. One shows the absolute number per event and timestep, the other
    shows the percentage an event has of all events per timestep.

    :param events_by_type: dict grouping events first by event type then by timestep
    :param events_by_time: dict grouping events first by timestep then by event type
    :param cut_off: relative number of time an event has to occur over all time steps to not be stripped before
                    generating the heatmaps
    :param output_dir: path to the folder where the images will be saved to
    :return: None
    """
    num_events = sum(sum(d.values()) for d in events_by_type.values())
    minimum_count = int(num_events * cut_off)
    events_by_type_squashed = {}

    for event_type, event_dict in events_by_type.items():
        count = sum(event_dict.values())

        if count >= minimum_count:
            events_by_type_squashed[event_type] = event_dict

    events_by_type_squashed = OrderedDict(sorted(events_by_type_squashed.items(), key=lambda t: sum(t[1].values())))

    combined: np.ndarray = np.zeros((len(events_by_type_squashed), len(events_by_time)), dtype=np.uint16)

    event_types = list(events_by_type_squashed.keys())

    for i, event_dict in enumerate(events_by_time.values()):
        for event, count in event_dict.items():
            try:
                combined[event_types.index(event)][i] = count
            except ValueError:
                pass
    axis_sum = combined.sum(axis=0)
    axis_sum[axis_sum == 0] = 1
    combined_normalized = combined / axis_sum

    with output_dir.joinpath('event-heatmap.md').open('w') as txt:
        txt.write('|   id |           type | count |\n')
        txt.write('|-----:|---------------:|------:|\n')
        for i, etype in enumerate(event_types):
            txt.write('| {:>4} | {} | {:>5} |\n'.format(i, etype, np.sum(combined[event_types.index(etype)])))

    logging.debug('shape: {},  #unique: {},  max: {}'.format(
        combined.shape, np.unique(combined).shape, np.max(combined))
    )

    norm = mcolors.PowerNorm(0.5, vmin=0, vmax=np.max(combined))
    fig = plt.figure(figsize=(5.5, 8))
    ax = fig.gca()
    ax.pcolormesh(combined, norm=norm, cmap=cm.Greens)
    ax.set_yticks(np.arange(len(event_types), dtype=np.float) + 0.5)
    if len(event_types) < 70:
        ax.set_yticklabels([pretty_event_latex(e) for e in event_types], fontsize=8)
    else:
        ax.set_yticklabels([])

    scalar_map = cm.ScalarMappable(norm=norm, cmap=cm.Greens)
    scalar_map.set_array(combined)
    fig.colorbar(scalar_map).set_label('count')
    ax.set_xlabel('time steps')

    fig.savefig(str(output_dir.joinpath('event-heatmap.pdf')), bbox_inches='tight')
    fig.clf()
    plt.close(fig)

    np.savetxt(str(output_dir.joinpath('event-heatmap.txt')), combined_normalized, fmt='%0.4f')

    norm = mcolors.PowerNorm(0.5, vmin=0, vmax=np.max(combined_normalized))
    fig = plt.figure(figsize=(5.5, 8))
    ax = fig.gca()
    ax.pcolormesh(combined_normalized, norm=norm, cmap=cm.Greens)
    ax.set_yticks(np.arange(len(event_types), dtype=np.float) + 0.5)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: r'\num{{{:g}}}'.format(x * INTERVAL)))
    if len(event_types) < 70:
        ax.set_yticklabels([pretty_event_latex(e) for e in event_types], fontsize=8)
    else:
        ax.set_yticklabels([])

    scalar_map = cm.ScalarMappable(norm=norm, cmap=cm.Greens)
    scalar_map.set_array(combined_normalized)
    fig.colorbar(scalar_map, aspect=40).set_label('percentage of events per time step')
    ax.set_xlabel(TIME_AXIS)

    fig.savefig(str(output_dir.joinpath('event-heatmap-normed.pdf')), bbox_inches='tight')
    fig.clf()
    plt.close(fig)


def generate_connected_components_plots(graphs: List[nx.Graph], output_dir: Path) -> None:
    """
    Generate plot summarizing information about the connected components.

    - overall number of connected components per time step
    - average size of connected components per time step
    - average size of connected components without the larged component per time step

    :param graphs: list of graphs
    :param output_dir: path to the folder where the images will be saved to
    :return: None
    """
    cc_count = [nx.number_connected_components(graph) for graph in graphs]

    fig = plt.figure(figsize=FIGURE_SIZE)
    ax = fig.gca()
    ax.set_xlabel('time steps')
    ax.set_ylabel('number of CC')
    ax.set_yscale('log')
    ax.plot(cc_count, 'blue', alpha=0.3)
    ax.plot(ssig.savgol_filter(cc_count, min(101, len(cc_count) - 3) | 1, 3, mode='nearest'), 'blue', alpha=1)
    fig.savefig(str(output_dir.joinpath('nodes-cc.pdf')), bbox_inches='tight')
    fig.clf()
    plt.close(fig)

    cc_mean = []
    cc_mean_largest = []
    cc_mean_wo_largest = []

    for graph in graphs:
        number_of_nodes = graph.number_of_nodes()
        nodes_in_largest_cc = len(max(nx.connected_components(graph), key=len))

        assert number_of_nodes == sum(len(node_set) for node_set in nx.connected_components(graph))

        cc_mean_largest.append(nodes_in_largest_cc / number_of_nodes)
        cc_mean.append(1 / nx.number_connected_components(graph))
        cc_mean_wo_largest.append((number_of_nodes - nodes_in_largest_cc) /
                                  max(1, number_of_nodes * nx.number_connected_components(graph) - number_of_nodes))

    fig = plt.figure(figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1] * 2))

    ax = fig.add_subplot(311)
    ax.set_ylabel('mean size')
    ax.plot(cc_mean, 'green', alpha=0.3)
    ax.plot(ssig.savgol_filter(cc_mean, min(101, len(cc_mean) - 3) | 1, 3, mode='nearest'), 'green', alpha=1)
    ax.grid()

    ax = fig.add_subplot(312)
    ax.set_ylabel('mean size without largest')
    ax.plot(cc_mean_wo_largest, 'green', alpha=0.3)
    ax.plot(ssig.savgol_filter(cc_mean_wo_largest, min(101, len(cc_mean_wo_largest) - 3) | 1, 3, mode='nearest'),
            'green', alpha=1)
    ax.grid()

    ax = fig.add_subplot(313)
    ax.set_xlabel('time steps')
    ax.set_ylabel('size of largest')
    ax.plot(cc_mean_largest, 'green', alpha=0.3)
    ax.plot(ssig.savgol_filter(cc_mean_largest, min(101, len(cc_mean) - 3) | 1, 3, mode='nearest'), 'green', alpha=1)
    ax.grid()

    fig.savefig(str(output_dir.joinpath('nodes-cc-mean.pdf')), bbox_inches='tight')
    fig.savefig(str(output_dir.joinpath('nodes-cc-mean.pgf')), bbox_inches='tight')
    fig.clf()
    plt.close(fig)


def generate_reducibility_plots(graphs: List[nx.Graph], output_dir: Path) -> None:
    """
    Calculates how many events are reducible and irreducible.

    :param graphs: list of graphs
    :param output_dir: path to the folder where the images will be saved to
    :return: None
    """
    tracker = np.zeros((len(graphs) - 1, 2), dtype=np.int)

    # this is really similar to the event counting but as it does not depend on real/virtual events
    # it seems better to write a separate function
    for i, current_graph, next_graph in zip(itertools.count(0), graphs, graphs[1:]):
        current_edges = frozenset(
            tuple(sorted([current_graph.nodes[u]['cluster'], current_graph.nodes[v]['cluster']])) for u, v in
            current_graph.edges)
        next_edges = frozenset(
            tuple(sorted([next_graph.nodes[u]['cluster'], next_graph.nodes[v]['cluster']])) for u, v in
            next_graph.edges)

        raw_events = current_edges.symmetric_difference(next_edges)
        node_count = Counter(itertools.chain.from_iterable(raw_events))

        reducible = 0
        irreducible = 0

        for u, v in raw_events:
            if node_count[u] > 1 or node_count[v] > 1:
                reducible += 1
            else:
                irreducible += 1

        tracker[i][0] = reducible
        tracker[i][1] = irreducible

    all_reducible = tracker[:, 0].sum()
    all_irreducible = tracker[:, 1].sum()
    all_events = all_reducible + all_irreducible

    logging.debug('reducible events: {:,d} ({:.2f})'.format(all_reducible, all_reducible / all_events))
    logging.debug('irreducible events: {:,d} ({:.2f})'.format(all_irreducible, all_irreducible / all_events))

    with output_dir.joinpath('event-reducibility.txt').open('w') as txt:
        print('reducible events: {:,d} ({:.2f})\nirreducible events: {:,d} ({:.2f})\nall events: {:,d} ({:.2f})'.format(
            all_reducible, all_reducible / all_events,
            all_irreducible, all_irreducible / all_events,
                           all_reducible + all_irreducible, 1.0
        ), file=txt)

    fig = plt.figure(figsize=FIGURE_SIZE)
    ax = fig.gca()
    ax.set_xlabel('time steps')
    ax.set_ylabel('percentage reducible')
    ax.set_ylim(0, 1)

    tracker_sum = tracker.sum(axis=1)
    tracker_sum[tracker_sum == 0] = 1

    reducible_proportion = tracker[:, 0] / tracker_sum
    ax.grid()
    ax.plot(reducible_proportion, 'green', alpha=0.3)
    ax.plot(ssig.savgol_filter(reducible_proportion, min(101, len(reducible_proportion) - 3) | 1, 3, mode='nearest'),
            'green', alpha=1)
    fig.savefig(str(output_dir.joinpath('event-reducibility.pdf')), bbox_inches='tight')
    fig.savefig(str(output_dir.joinpath('event-reducibility.pgf')), bbox_inches='tight')
    fig.clf()
    plt.close(fig)


def generate_edge_plots(graphs: List[nx.Graph], output_dir: Path) -> None:
    """
    Generate plots showing:

    - number of edges per time step
    - sum of length of edges per time step
    - mean length of edges per time step

    :param graphs: list of graphs
    :param output_dir: path to the folder where the images will be saved to
    :return: None
    """

    edge_counts = [graph.number_of_edges() for graph in graphs]
    fig = plt.figure(figsize=FIGURE_SIZE)
    ax = fig.gca()
    ax.plot(edge_counts, 'blue', alpha=0.3)
    ax.plot(ssig.savgol_filter(edge_counts, min(101, len(edge_counts) - 3) | 1, 3, mode='nearest'), 'blue', alpha=1)
    ax.set_xlabel('time')
    ax.set_ylabel('count')
    fig.savefig(str(output_dir.joinpath('edges-count.pdf')), bbox_inches='tight')
    fig.clf()
    plt.close(fig)

    edges_length = []
    edges_length_mean = []

    def edge_distance(graph: nx.Graph, u: int, v: int) -> float:
        xu, yu = graph.nodes[u]['x'], graph.nodes[u]['y']
        xv, yv = graph.nodes[v]['x'], graph.nodes[v]['y']

        dx = xu - xv
        dy = yu - yv

        return math.sqrt(dx * dx + dy * dy)

    for graph in graphs:
        edges_sum = sum(edge_distance(graph, u, v) for u, v in graph.edges)
        edges_length.append(edges_sum)
        edges_length_mean.append(edges_sum / max(1, graph.number_of_edges()))

    fig, (ax_l, ax_m) = plt.subplots(nrows=2, figsize=FIGURE_SIZE)

    ax_l.set_xlabel('time steps')
    ax_l.set_ylabel('sum of edge length in px')
    ax_l.plot(edges_length, 'blue', alpha=0.3)
    ax_l.plot(ssig.savgol_filter(edges_length, min(101, len(edges_length) - 3) | 1, 3, mode='nearest'), 'blue', alpha=1)

    ax_m.set_xlabel('time steps')
    ax_m.set_ylabel('mean edge length in px')
    ax_m.plot(edges_length_mean, 'orange', alpha=0.3)
    ax_m.plot(ssig.savgol_filter(edges_length_mean, min(101, len(edges_length_mean) - 3) | 1, 3, mode='nearest'),
              'orange', alpha=1)

    fig.savefig(str(output_dir.joinpath('edges-length.pdf')), bbox_inches='tight')
    fig.clf()
    plt.close(fig)


def generate_area_plots(graphs: List[nx.Graph], output_dir: Path) -> None:
    """
    Generate plots showing

    - pixels covered by nodes per time step
    - mean area of nodes per time step

    :param graphs: list of graphs
    :param output_dir: path to the folder where the images will be saved to
    :return: None
    """
    area_sum = []
    area_mean = []

    for graph in graphs:
        covered_area = sum(math.pi * ndata['r'] * ndata['r'] for nid, ndata in graph.nodes(data=True))
        area_sum.append(covered_area)
        area_mean.append(covered_area / graph.number_of_nodes())

    fig, (ax_t, ax_m) = plt.subplots(nrows=2, figsize=FIGURE_SIZE)

    ax_t.set_xlabel('time steps')
    ax_t.set_ylabel('covered area in px')
    ax_t.plot(area_sum, 'blue', alpha=0.3)
    ax_t.plot(ssig.savgol_filter(area_sum, min(101, len(area_sum) - 3) | 1, 3, mode='nearest'), 'blue', alpha=1)

    ax_m.set_xlabel('time steps')
    ax_m.set_ylabel('mean area of nodes')
    ax_m.plot(area_mean, 'orange', alpha=0.3)
    ax_m.plot(ssig.savgol_filter(area_mean, min(101, len(area_mean) - 3) | 1, 3, mode='nearest'), 'orange', alpha=1)

    fig.savefig(str(output_dir.joinpath('nodes-area.pdf')), bbox_inches='tight')
    fig.clf()
    plt.close(fig)


def generate_transition_graph(graphs: List[nx.Graph], histogram_pairs: List[Tuple[int, int]], max_degree: Optional[int],
                              output_dir: Path) -> None:
    """
    Generate some plots to show statistics regarding degree transitions.

    :param graphs: list of graphs
    :param histogram_pairs: List of tuples for "interesting" transitions that will have a histogram plotted.
        -1 represents vanishing nodes in the second number in each tuple
    :param max_degree: if not None all nodes with a degree greater than max_degree will be treated as if they hat
        a degree of max_max_degree
    :param output_dir: path to the folder where the images will be saved to
    :return: None
    """
    if max_degree is None:
        max_degree = max(max(graph.degree, key=lambda e: e[1])[1] for graph in graphs)
        forced_max_degree = False
        logging.debug('max degree found in graphs: {}'.format(max_degree))
    else:
        logging.debug('overriding max degree, forcibly setting it to {}'.format(max_degree))
        forced_max_degree = True

    transitions = np.zeros((max_degree + 1, max_degree + 3), dtype=np.uint32)
    run_lengths = collections.defaultdict(list)

    for graph, next_graph in zip(graphs[:-1], graphs[1:]):
        next_c2n = {node_data['cluster']: node_id for node_id, node_data in next_graph.nodes(data=True)}
        for node_id, cluster_id in graph.nodes(data='cluster'):
            node_degree = min(graph.degree[node_id], max_degree)
            run_lengths[cluster_id].append(node_degree)
            next_node_degree = min(get_degree(next_graph, next_c2n, cluster_id), max_degree)
            transitions[node_degree, next_node_degree] += 1

    lifetimes = collections.defaultdict(lambda: collections.defaultdict(list))
    for cluster_id, runnings in run_lengths.items():
        runnings.append(max_degree + 2)
        last_degree = runnings[0]
        lifetime = 1
        for degree in runnings[1:]:
            if degree == last_degree:
                lifetime += 1
                continue
            lifetimes[last_degree][degree].append(lifetime)
            last_degree = degree
            lifetime = 1

    np.fill_diagonal(transitions, 0)
    sums = transitions.sum(axis=1)
    sums[sums == 0] = 1
    transitions_normed = (transitions.T / sums).T
    logging.debug('transition count: min {}    max {}   #unique {}'.format(transitions.min(), transitions.max(),
                                                                           len(np.unique(transitions))))

    from draw import TIKZ_COLORS
    from draw.templates import transition_main as main, transition_edge as edge, transition_node as node

    nodes = []
    edges = []

    for i in range(transitions_normed.shape[0] - int(forced_max_degree)):
        nodes.append(node.substitute(degree=i, color=TIKZ_COLORS[i], label=i))
    if forced_max_degree:
        nodes.append(node.substitute(degree=i + 1, color=TIKZ_COLORS[i + 1], label=r'\geqslant {}'.format(i + 1)))

    TRANSITION_THRESHOLD = 0.01
    for i, j in itertools.product(range(transitions_normed.shape[0]), range(transitions_normed.shape[1])):
        if transitions_normed[i, j] >= TRANSITION_THRESHOLD:
            j_mapped = j if j <= max_degree else 'X'
            if i + 1 == j:
                bend = 20
            elif j > max_degree + 1:
                bend = 0
            elif i < j and transitions_normed[j, i] >= TRANSITION_THRESHOLD:
                bend = 20
            elif abs(i - j) == (max_degree + 1) / 2:
                bend = 20
            else:
                bend = 0
            edge_out = edge.substitute(
                source=i,
                target=j_mapped,
                bend=bend,
                color=TIKZ_COLORS[i],
                prob='{:.2f}'.format(transitions_normed[i, j]),
                lifetime='{:.1f}'.format(sum(lifetimes[i][j]) / len(lifetimes[i][j]))
            )
            edges.append(edge_out)

    with output_dir.joinpath('event-decisions.tex').open('w') as out_combined:
        out_combined.write(main.substitute(nodes='\n'.join(nodes), edges='\n'.join(edges), num_nodes=max_degree + 1))

    for (d1, d2) in histogram_pairs:
        d2 = d2 if d2 >= 0 else max_degree + 2
        bincount = np.bincount(np.array(lifetimes[d1][d2]))
        binrange = np.flatnonzero(bincount)
        np.savetxt(str(output_dir.joinpath(f'event-decision-{d1}-{d2}.txt')), bincount, fmt='% d')
        summed_lifetime = sum(lifetimes[d1][d2])
        fig, ax = plt.subplots(figsize=(2.8, 1.5))
        ax.set_xlabel(TIME_AXIS)
        ax.set_ylabel('count')
        ax.text(0.95, 0.95,
                r'\begin{{tabular}}{{r}}total transitions \num{{{}}}\\total lifetime \SI{{{:d}}}{{\second}}\end{{tabular}}'
                .format(len(lifetimes[d1][d2]), summed_lifetime * INTERVAL),
                verticalalignment='top', horizontalalignment='right',
                transform=ax.transAxes)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: r'\num{{{:g}}}'.format(x * INTERVAL)))
        ax.set_title('transition from ${}$ to ${}$'.format(d1, d2 if d2 < max_degree + 2 else r'\varnothing'))
        ax.bar(binrange, bincount[binrange], color='green', bottom=1, log=True, width=1.0)
        fig.savefig(str(output_dir.joinpath(f'event-decision-{d1}-{d2}.pgf')), bbox_inches='tight')
        fig.savefig(str(output_dir.joinpath(f'event-decision-{d1}-{d2}.pdf')), bbox_inches='tight')
        fig.clear()
        plt.close(fig)


def generate_degree_boxplots(graphs: List[nx.Graph], max_degree: Optional[int], output_dir: Path) -> None:
    """
    Boxplots for the time a node remains with its degree.

    :param graphs: list of graphs
    :param max_degree: if not None all nodes with a degree greater than max_degree will be treated as if they hat
        a degree of max_max_degree
    :param output_dir: path to the folder where the images will be saved to
    :return: None
    """
    if max_degree is None:
        max_degree = max(max(graph.degree, key=lambda e: e[1])[1] for graph in graphs)
    logging.debug('max degree in graphs: {}'.format(max_degree))

    time_steps = len(graphs)
    max_cluster = max(max(cluster_id for nid, cluster_id in graph.nodes(data='cluster')) for graph in graphs)

    node_degrees = np.full((max_cluster + 1, time_steps), -1, dtype=np.int8)

    for ts, graph in enumerate(graphs):
        for node_id, cluster_id in graph.nodes(data='cluster'):
            node_degrees[cluster_id, ts] = min(graph.degree[node_id], max_degree)

    degree_lifetimes = collections.defaultdict(list)
    cluster_lifetimes = []

    for np_list in node_degrees:
        cluster_lifetimes.append(np.count_nonzero(np_list + 1))
        for key, group in itertools.groupby([e for e in np_list if e != -1]):
            degree_lifetimes[key].append(len(list(group)))

    cluster_lifetimes = np.array(cluster_lifetimes[1:])
    logging.debug(
        'cluster lifetimes: min {}    max {}    #unique {}'.format(cluster_lifetimes.min(), cluster_lifetimes.max(),
                                                                   np.unique(cluster_lifetimes).size))

    for degree, lifetimes in sorted(degree_lifetimes.items()):
        lifetimes = np.array(lifetimes)
        logging.debug(
            'degree <{:>2}>: # {:>5}   min {}    max {:>4}    #unique {:>3}    mean {:5.2f}    median {:5.2f}'.format(
                degree, lifetimes.size, lifetimes.min(), lifetimes.max(), np.unique(lifetimes).size, np.mean(lifetimes),
                np.median(lifetimes)))

    degree_lifetimes_keys = list(sorted(degree_lifetimes.keys()))

    fig = plt.figure(figsize=FIGURE_SIZE)
    ax = fig.gca()
    ax.set_title('lifetime in time steps by node degree')
    ax.set_yscale('log')
    ax.boxplot([degree_lifetimes[key] for key in degree_lifetimes_keys], labels=degree_lifetimes_keys, whis=(1, 99),
               showmeans=True)
    ax.set_ylabel('time steps')
    ax.set_xlabel('node degree')

    fig.savefig(str(output_dir.joinpath('nodes-lifetimes.pdf')), bbox_inches='tight')
    fig.clf()
    plt.close(fig)


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('crop_dir', metavar='crop-dir',
                        help='path to folder containing pickle files of crop')
    parser.add_argument('--output-dir', dest='output_dir', default='figures',
                        help='path to the folder where the annotated graphs will be saved (in a subfolder)')
    parser.add_argument('--debug', action='store_true',
                        help='more detailed information during processing')
    parser.add_argument('--max-degree', dest='max_degree', metavar='N', type=int, default=None,
                        help='maximum degree to consider for event plots, defaults to no plots.')
    parser.add_argument('--event-cutoff', dest='cut_off', metavar='N', type=float, default=0.0025,
                        help='Cut off events with less occurrences in relative numbers, defaults to 0.0025')
    args = parser.parse_args()

    crop_dir = Path(args.crop_dir)
    if not crop_dir.exists() or not crop_dir.is_dir():
        raise ValueError('argument \'{}\' is not a directory'.format(args.crop_dir))

    output_dir = Path(args.output_dir).joinpath(crop_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_formatter = logging.Formatter(LOG_FORMAT)

    file_handler = logging.FileHandler(str(output_dir.joinpath('crop-stats.log')), mode='w')
    file_handler.setFormatter(log_formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)

    root_logger = logging.getLogger()
    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)

    if args.debug:
        root_logger.setLevel(logging.DEBUG)
    else:
        root_logger.setLevel(logging.INFO)
    logging.getLogger('matplotlib').setLevel(logging.WARN)

    logging.debug('output will be written to {}'.format(output_dir))

    start_time = datetime.now()

    logging.debug('Starting {} at {}'.format(crop_dir.stem, start_time))
    files = sorted(crop_dir.glob('*.pickle'))
    logging.info('Found {:>4} pickle files in {}'
                 .format(len(files), crop_dir.name))
    results = [read_gpickle(gpickle) for gpickle in files]

    pickle.dump(results, output_dir.joinpath('graphs.pickle').open('wb'), protocol=pickle.HIGHEST_PROTOCOL)

    logging.debug('matplotlib backend: {}'.format(plt.get_backend()))

    event_stats = [
        ('real', count_real_events),
        ('virtual', count_virtual_events)
    ]

    for name, event_counter in event_stats:
        current_output_dir = output_dir.joinpath(name)
        current_output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(' === Event stats for {} events ==='.format(name, current_output_dir))
        events_pickle: Path = current_output_dir.joinpath('events.pickle')
        if events_pickle.exists():
            logging.debug('Loading events from cache.')
            events_by_type, events_by_time = pickle.load(events_pickle.open('rb'))
        else:
            logging.debug('Counting events (and caching the result)')
            events_by_type, events_by_time = event_counter(results)
            pickle.dump((events_by_type, events_by_time), events_pickle.open('wb'), protocol=pickle.HIGHEST_PROTOCOL)

        logging.info('Generating CSV file')
        generate_event_csv(results, events_by_time, current_output_dir)

        logging.info('Generating plot for events by time')
        generate_events_by_time_plot(results, events_by_time, current_output_dir)

        logging.info('Generating histogram for events with at least {} occurrences'.format(args.cut_off))
        generate_event_histograms(events_by_type, args.cut_off, current_output_dir)

        logging.info('Generating heatmap for events')
        generate_event_heatmaps(events_by_type, events_by_time, args.cut_off, current_output_dir)

    logging.info(' === Event agnostic stats ===')

    logging.info('Generating transition plots')
    histogram_pairs: List[Tuple[int, int]] = [(0, 1), (1, 0), (2, 3), (2, -1), (3, 2), (4, 3), (5, 2)]
    generate_transition_graph(results, histogram_pairs, args.max_degree, output_dir)

    logging.info('Generating plots for node probabilities')
    generate_stats_plots(results, output_dir)

    logging.info('Generating plots for connected components')
    generate_connected_components_plots(results, output_dir)

    logging.info('Generating plots for temporal clusters')
    generate_cluster_stats(results, output_dir)

    logging.info('Generating plots for reducibility')
    generate_reducibility_plots(results, output_dir)

    logging.info('Generating edge plots')
    generate_edge_plots(results, output_dir)

    logging.info('Generating area plots')
    generate_area_plots(results, output_dir)

    logging.info('Generating degree life boxplots')
    generate_degree_boxplots(results, args.max_degree, output_dir)

    end_time = datetime.now()
    logging.info('finished {} at {} after {}'.format(crop_dir.stem, end_time, end_time - start_time))


if __name__ == '__main__':
    main()
