#!/usr/bin/env python3

"""
This file converts MATLAB files to networkx graphs and stores them as pickles.

The following directory layout is expected:

    crop_dir/{X,Y,R,C,D}.mat
"""

import argparse
import pathlib
from datetime import datetime
import logging

import numpy as np
import scipy.io as sio
import networkx as nx

X_MAT = 'X.mat'  # x coordinates
Y_MAT = 'Y.mat'  # y coordinates
R_MAT = 'R.mat'  # radii
C_MAT = 'C.mat'  # contains connected components
D_MAT = 'D.mat'  # contains distances between nodes, 0 means no connection


def convert_crop(in_path: pathlib.Path, out_path: pathlib.Path) -> None:
    """Converts MATLAB files to networkx objects and writes them to the disk.

    :param in_path: folder containing the MATLAB files
    :param out_path: base folder where the data should be written. The folder with the current crop name is
                     created and the pickles will be stored there.
    :return: nothing
    """
    X = sio.loadmat(str(in_path.joinpath(X_MAT)))['XCell'].squeeze(1)
    Y = sio.loadmat(str(in_path.joinpath(Y_MAT)))['YCell'].squeeze(1)
    R = sio.loadmat(str(in_path.joinpath(R_MAT)))['RCell'].squeeze(1)
    D = sio.loadmat(str(in_path.joinpath(D_MAT)))['DCell'].squeeze(1)

    assert len(X) == len(Y) == len(R)
    assert in_path.is_dir()

    crops = len(X)
    logging.debug('Found {:04} entries in [{}]'.format(crops, in_path))

    out_path = out_path.joinpath(in_path.name)
    out_path.mkdir(parents=True, exist_ok=True)

    for i, (xx, yy, rr, dd) in enumerate(zip(X, Y, R, D)):
        xx = xx.squeeze(1)
        yy = yy.squeeze(1)
        rr = rr.squeeze(1)

        assert len(xx) == len(yy) == len(rr)

        graph = nx.Graph()
        for j, (x, y, r) in enumerate(zip(xx, yy, rr)):
            graph.add_node(j+1, x=np.asscalar(x), y=np.asscalar(y), r=float(r))

        list_u, list_v = np.where(dd.toarray())

        for u, v in zip(list_u, list_v):
            graph.add_edge(int(u+1), int(v+1))

        graph.name = '{}_{:05}'.format(in_path.stem, i)
        graph.graph['graph_id'] = i
        graph.graph['converted_by'] = 'convert_matlab.py'
        graph.graph['converted_date'] = datetime.now()
        graph.graph['nodes_indexed_by'] = 1
        nx.write_gpickle(graph, out_path.joinpath(graph.name).with_suffix('.pickle').open(mode="wb"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('crop_dir', metavar='crop-dir',
                        help='Path to the crop folder.')
    parser.add_argument('--output-dir', dest='output_dir', default='converted_matlab',
                        help='Path to the output. Existing files will be overwritten')
    parser.add_argument('--debug', action='store_true',
                        help='enable more verbose console output')
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    crop_dir = pathlib.Path(args.crop_dir)
    out_dir = pathlib.Path(args.output_dir)

    convert_crop(crop_dir, out_dir)


if __name__ == '__main__':
    main()
