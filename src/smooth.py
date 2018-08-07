#!/usr/bin/env python3

"""
Take gray scale images, binarize them and save them as tif files.
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Callable

import numpy as np
import skimage.io as io
import skimage.io.collection as ioc
import skimage.filters as sif

from common import LOG_FORMAT


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input_dir', metavar='crop-dir',
                        help='Path to the smoothed output. Existing files will be overwritten')
    parser.add_argument('--output-dir', dest='output_dir', default='results/bwsmooth',
                        help='Path to the output. Existing files will be overwritten')
    parser.add_argument('--debug', action='store_true',
                        help='enable more verbose console output')
    parser.add_argument('--sigma', dest='sigma', default=1.41, type=int,
                        help='Standard deviation for Gaussian kernel')

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)
    else:
        logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    binary_dir = Path(args.input_dir)
    if not binary_dir.exists() or not binary_dir.is_dir():
        raise ValueError('argument \'{}\' is not a directory'.format(args.input_dir))

    start_time = datetime.now()
    logging.info('Starting time {}'.format(start_time))

    files = sorted(binary_dir.glob('*.tif'))
    logging.info('found {} files'.format(len(files)))
    output_dir = Path(args.output_dir).joinpath(binary_dir.stem)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.debug('loading files and creating block')
    files = sorted(output_dir.glob('*.tif'))
    images = ioc.ImageCollection(files, load_func=lambda f: io.imread(f, as_grey=True))
    image_block = images.concatenate()

    logging.debug('smoothin block using sigma={}'.format(args.sigma))
    smoothed_block = sif.gaussian(image_block, sigma=args.sigma)

    logging.debug('writing smoothed files')
    for filename, smooth_image in zip(files, smoothed_block):
        binary = (smooth_image > 0.5).astype(np.uint8) * 255
        io.imsave(str(output_dir.joinpath(filename.name).with_suffix('.tif')), binary, plugin='tifffile', compress=6)

    end_time = datetime.now()
    logging.info('finished {} after {}'.format(end_time, end_time-start_time))


if __name__ == '__main__':
    main()
