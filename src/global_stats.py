#!/usr/bin/env python3

"""
This file will process all available graphs and display some statistics about it.

Currently the following statistic is included:

* Average degree plotted against q=p_0/p_4 and p=p_3/p_4
"""

import argparse
import pathlib
import itertools

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def read_gpickle(gpickle: pathlib.Path) -> list:
    """Takes a single pickle, reads it and calculates

    * p_0/p_4,
    * p_3/p_4 and
    * average node degree.

    Graphs where no node has degree four are ignored.

    :param gpickle: Path to pickle file
    :return: list containing the elements listed above in that order
    """
    with gpickle.open('rb') as pickle_file:
        graph = nx.read_gpickle(pickle_file)

    hist = nx.degree_histogram(graph) + [0.0]*5

    try:
        avg_degree = sum(graph.degree().values()) / graph.number_of_nodes()
        p_04 = hist[0] / hist[4]
        p_34 = hist[3] / hist[4]
    except ZeroDivisionError:
        return []

    return [p_04, p_34, avg_degree]


def read_crop_dir(crop_dir: pathlib.Path) -> list:
    """Given a crop folder, this function calls read_gpickle for all pickle files found

    :param crop_dir: Path to the folder containing the pickle files
    :return: flat list containing the calculated data for all pickles
    """
    files = sorted(crop_dir.glob('*.pickle'))

    result = []
    print('Found {:>5} gpickles in [{}]'.format(len(files), crop_dir))

    for gpickle in files:
        result.extend(read_gpickle(gpickle))

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('data_dir', help='Path to folder containing all crop folders.')
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError

    crop_dirs = [dir for dir in data_dir.iterdir() if dir.is_dir()]
    print('starting parallel execution')
    results = (read_crop_dir(path) for path in crop_dirs)
    print('fetching and reshaping results')
    results = np.array(list(itertools.chain.from_iterable(results)), dtype=float).reshape((-1, 3))
    print('shape {}'.format(results.shape))

    norm = mcolors.Normalize(vmin=np.min(results[:,2]), vmax=np.max(results[:,2]))

    print('plotting')
    plt.figure(1)
    plt.ylim([0, 15])
    plt.xlim([0.001, 1000])
    plt.xscale('log')
    plt.scatter(results[:,0], results[:,1], s=1, c=norm(results[:,2]), cmap=cm.jet)
    plt.ylabel(r'$p=\frac{p_3}{p_4}$')
    plt.xlabel(r'$q=\frac{p_0}{p_4}$')
    scalar_map = cm.ScalarMappable(norm=norm, cmap=cm.jet)
    scalar_map.set_array(results[:,2])
    plt.colorbar(scalar_map).set_label(r'$\langle k\rangle$')
    plt.show()
