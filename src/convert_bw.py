#!/usr/bin/env python3

"""
Convert bw images to graphs.
Can currently only handle up to around 32000 nodes per image.
"""

import argparse
import pathlib
from datetime import datetime
import logging
import itertools
from typing import List, Tuple, Set, NamedTuple

import networkx as nx
import numpy as np
import cv2.cv2 as cv2
import skimage.morphology as sim
import skimage.filters as sif

from common import LOG_FORMAT


class CircularMask(NamedTuple):
    """Slicing and mask to operate on a smaller part of an numpy array.

    This class allows to only operate on a smaller part of a numpy array in a way that is hopefully more
    readable than using a plain tuple.
    """
    mask: np.ndarray
    slice: Tuple[slice, slice]
    x: int
    y: int
    r: int

    def mind_offset(self, coordinates: Tuple[int, int]) -> Tuple[int, int]:
        """
        Transforms coordinates in the slice to coordinates in the original numpy array.
        :param coordinates: coordinates inside the slice
        :return: coordinates inside the original / unsliced numpy array
        """
        return (coordinates[0] + self.x, coordinates[1] + self.y)


def circular_mask(c_x: int, c_y: int, r: float, shape: Tuple[int, int]) -> CircularMask:
    """
    Formerly commented as numpy magic. Creates a circular mask to select in numpy arrays.

    :param c_x: center x
    :param c_y: center y
    :param r:  radius
    :param shape: shape of array to process on
    :return: circular mask
    """
    r_up = int(r)
    x1_corner = max(c_x - r_up, 0)
    x2_corner = min(c_x + r_up, shape[0]-1)
    y1_corner = max(c_y - r_up, 0)
    y2_corner = min(c_y + r_up, shape[1]-1)

    x_width = x2_corner - x1_corner
    y_height = y2_corner - y1_corner

    thatslice = (slice(x1_corner, x2_corner), slice(y1_corner, y2_corner))

    x, y = np.ogrid[-r_up:x_width - r_up, -r_up:y_height - r_up]
    mask = x * x + y * y < r * r

    return CircularMask(mask, thatslice, x1_corner, y1_corner, r_up)


def update_image(gray: np.ndarray, dist: np.ndarray, coords: Tuple[int, int], radius: int) -> None:
    """
    Delete a circle from gray and updates the dist array accordingly. gray and dist will be modified inplace.

    :param gray: ndarray containing the binary image
    :param dist: ndarray containing the result of the distance transformation on gray
    :param coords: coordinates of the center of the circle that will be removed from gray
    :param radius: radius of that circle
    :return: None
    """
    x, y = (int(z) for z in coords)
    r = int(radius)

    x_low = max(0, x-4*r)
    x_high = min(dist.shape[0], x+4*r)
    y_low = max(0, y-4*r)
    y_high = min(dist.shape[1], y+4*r)

    slice_segment = (slice(x_low, x_high), slice(y_low, y_high))

    cv2.circle(gray, (y, x), r, 0, -1)
    sub_gray = gray[slice_segment].copy()

    sub_dist_old = dist[slice_segment]
    sub_dist_new = cv2.distanceTransform(sub_gray, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    dist[slice_segment] = np.minimum(sub_dist_new, sub_dist_old)


def convert(files: List[pathlib.Path], min_radius: float, wiggle_radius: float, seeded_radius_factor: float) -> List[nx.Graph]:
    """
    Converts black&white pictures to graphs which nodes cover the foreground (white). The nodes are created in a way
    that there radii are decreasing until no node with a radius greater or equal to the min_radius is found.
    :param files: Paths to the images, will be converted in that order
    :param min_radius: minimum radius for nodes, lower values increase accuracy and runtime
    :param wiggle_radius: radius in which tracked nodes may move to maximize area
    :param seeded_radius_factor: allow the min radius of tracked nodes to be smaller by this factor
    :return: a list with graphs in the same order as the input
    """
    graphs = []
    seeds = set()

    for i, image_path in enumerate(files):
        logging.debug('processing graph {} ({:>4}/{:>4})'.format(image_path, i + 1, len(files)))
        graph = nx.Graph()
        graph.name = '{}_{:05}'.format(image_path.stem, i)
        graph.graph['graph_id'] = i
        graph.graph['converted_by'] = 'convert_bw.py'
        graph.graph['converted_date'] = datetime.now()
        graph.graph['nodes_indexed_by'] = 2 # id of the first found node. 0 and 1 are reserved for background and (yet unlabeled foreground)
        graph.graph['min_radius'] = min_radius
        graph.graph['wiggle_radius'] = wiggle_radius
        graphs.append(graph)

        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        gray = np.pad(gray, 1, mode='constant')  # pad one pixel on each side with default constant being zero
        skeleton = gray.copy()
        thresh = sif.threshold_mean(skeleton)
        skeleton = sim.skeletonize(skeleton > thresh).astype(np.int16)  # to allow for more nodes, change to larger type
        # meaning of values in skeleton
        #  0 -> background
        #  1 -> skeleton without attached node
        # >1 -> skeleton pixel attached to node with corresponding id
        # <0 -> visited skeleton pixel attached to node with corresponding id (just negated)
        node_ids = itertools.count(2)

        dist = cv2.distanceTransform(gray, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        fixed_seeds = frozenset(seeds)
        seeds.clear()
        for sx, sy, seed_id in sorted(fixed_seeds, key=lambda e: dist[e[0], e[1]], reverse=True):

            if wiggle_radius > 0.1:
                mask = circular_mask(sx, sy, wiggle_radius, gray.shape)
                masked_dist = np.ma.array(dist[mask.slice], mask=np.invert(mask.mask))
                max_coordinates = np.unravel_index(masked_dist.argmax(fill_value=0.0), masked_dist.shape)
                (ax, ay) = mask.mind_offset(max_coordinates)
            else:
                (ax, ay) = (sx, sy)

            radius = dist[ax, ay]
            if radius < min_radius * seeded_radius_factor:
                continue

            mask = circular_mask(ax, ay, radius - 1, gray.shape)
            if skeleton[mask.slice][mask.mask].any():
                node_id = next(node_ids)
                graph.add_node(node_id, x=int(ay)-1, y=int(ax)-1, r=float(radius), convert_seed=seed_id)
                node_area = np.ma.masked_array(skeleton[mask.slice], mask=np.invert(mask.mask))
                node_area[node_area == 1] = node_id
                seeds.add((ax, ay, node_id))
                update_image(gray, dist, (ax, ay), dist[ax, ay])

        while True:
            argmax = np.unravel_index(dist.argmax(), dist.shape)

            if dist[argmax] < min_radius:
                break

            ndx = np.asscalar(argmax[0])
            ndy = np.asscalar(argmax[1])
            ndr = dist[argmax] - 1  # reducing the radius prevents pixels associated to multiple nodes
            mask = circular_mask(ndx, ndy, ndr, gray.shape)

            if skeleton[mask.slice][mask.mask].any():
                node_id = next(node_ids)
                graph.add_node(node_id, x=int(ndy)-1, y=int(ndx)-1, r=int(dist[argmax]), convert_seed=0)
                node_area = np.ma.masked_array(skeleton[mask.slice], mask=np.invert(mask.mask))
                node_area[node_area == 1] = node_id
                seeds.add((ndx, ndy, node_id))

            update_image(gray, dist, argmax, dist[argmax])

        logging.debug('{} node(s) found'.format(graph.number_of_nodes()))

        max_id = graph.number_of_nodes() + 2
        assert np.all(skeleton <= max_id), 'maximal value in skeleton is {}, while max_id={}'.format(skeleton.max(), max_id)

        for node_id in graph.nodes(data=False):
            neighbors = explore_neighbors(node_id, skeleton)
            assert all(id > 1 for id in neighbors), 'Found neighbor with id <= 1 {}'.format(neighbors)
            graph.add_edges_from((node_id, int(neighbor)) for neighbor in neighbors)

    return graphs


def explore_neighbors(node_id: int, skeleton: np.array) -> Set[int]:
    """
    Check the neighbours around the seeds and follow the skeleton until another node or a leave is reached.
    Depending on the value of the neighboring element, the following steps are performed

      - value < 1: Element is background or already visited -> Do nothing.
      - value == node_id: Element is already on seed list -> Do nothing.
      - value == 1: Element is unlabeled skeleton -> Label with current node_id and save as new seed.
      - value > 1:
    :param node_id: id of the current node
    :param skeleton: array of the skeleton
    :return: a set containing the node_ids of neighbours
    """
    neighbors = set()
    seeds: List = np.argwhere(skeleton == node_id).tolist()
    max_x = skeleton.shape[0] - 2 # -1 for 0index and -1 for padding
    max_y = skeleton.shape[1] - 2
    while seeds:
        x, y = seeds.pop()
        assert skeleton[x, y] == node_id, 'found value on skeleton in seed list that does not match ({} != {})'.format(skeleton[x,y], node_id)

        skeleton[x, y] *= -1

        for dx, dy in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
            nx = x + dx
            ny = y + dy

            if nx < 1 or nx > max_x or ny < 1 or ny > max_y:
                continue

            value = skeleton[nx, ny]

            if value < 1 or value == node_id:
                # pixel that are background, already visited or already in seeds
                continue

            if value == 1:
                skeleton[nx, ny] = node_id
                seeds.append([nx, ny])
                continue

            assert(value > 1 and value != node_id)

            neighbors.add(value)

    return neighbors


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input_dir', metavar='crop-dir',
                        help='path to folder containing bw images as tif for a single crop')
    parser.add_argument('--output-dir', dest='output_dir', default='converted_bw',
                        help='Path to the output. Existing files will be overwritten')
    parser.add_argument('--reverse', action='store_true',
                        help='reverse file list before converting')
    parser.add_argument('--debug', action='store_true',
                        help='enable more verbose console output')
    parser.add_argument('--min-radius', metavar='R', dest='min_radius', type=float, default=5.0,
                        help='minimum radius for nodes to be considered')
    parser.add_argument('--wiggle-radius', metavar='R', dest='wiggle_radius', type=float, default=3.0,
                        help='radius in which tracked nodes may move to maximize area')
    parser.add_argument('--seeded-radius-factor', metavar='X', dest='seeded_radius_factor', type=float, default=0.5,
                        help='successors to node may be smaller by this factor than the min radius')
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)
    else:

        logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    bw_dir = pathlib.Path(args.input_dir)
    if not bw_dir.exists() or not bw_dir.is_dir():
        raise ValueError('argument \'{}\' is not a directory'.format(args.input_dir))

    start_time = datetime.now()
    logging.info('Starting {} at {}'.format(bw_dir.stem, start_time))

    files = sorted(bw_dir.glob('*.tif'))
    if args.reverse:
        files.reverse()
    logging.info('found {} files'.format(len(files)))

    graphs = convert(files, args.min_radius, args.wiggle_radius, args.seeded_radius_factor)

    output_dir = pathlib.Path(args.output_dir).joinpath(bw_dir.stem)
    output_dir.mkdir(parents=True, exist_ok=True)
    for graph in graphs:
        nx.write_gpickle(graph, output_dir.joinpath(graph.name).with_suffix('.pickle').open("wb"))

    end_time = datetime.now()
    logging.info('finished {} at {} after {}'.format(bw_dir.stem, end_time, end_time-start_time))


if __name__ == '__main__':
    main()
