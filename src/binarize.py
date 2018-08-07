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
import skimage.filters as sif

from common import LOG_FORMAT

THRESHOLD_METHODS = {
    'otsu': sif.threshold_otsu,
    'mean': sif.threshold_mean,
    'yen':  sif.threshold_yen,
    'local': lambda image: sif.threshold_local(image, 351)
}


def binarize(files: List[Path], threshold_method: Callable[[np.ndarray], float], output_dir: Path) -> None:
    for i, image_path in enumerate(files):
        gray = io.imread(str(image_path), as_grey=True)
        threshold = threshold_method(gray)
        binary = (gray <= threshold).astype(np.uint8) * 255

        io.imsave(str(output_dir.joinpath(image_path.name).with_suffix('.tif')), binary, plugin='tifffile', compress=6)


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input_dir', metavar='crop-dir',
                        help='path to folder containing the original images as tif for a single crop')
    parser.add_argument('--output-dir', dest='output_dir', default='results/bw',
                        help='Path to the output. Existing files will be overwritten')
    parser.add_argument('--debug', action='store_true',
                        help='enable more verbose console output')
    parser.add_argument('--threshold', choices=list(THRESHOLD_METHODS.keys()), default='otsu',
                        help='Select a thresholding method')

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)
    else:
        logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    raw_dir = Path(args.input_dir)
    if not raw_dir.exists() or not raw_dir.is_dir():
        raise ValueError('argument \'{}\' is not a directory'.format(args.input_dir))

    start_time = datetime.now()
    logging.info('Starting time {}'.format(start_time))
    logging.debug('Selected {} as global threshold method'.format(args.threshold))
    threshold = THRESHOLD_METHODS[args.threshold]

    files = sorted(raw_dir.glob('*.tif'))
    logging.info('found {} files'.format(len(files)))

    output_dir = Path(args.output_dir).joinpath(args.threshold, raw_dir.stem)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.debug('binarizing files')
    binarize(files, threshold, output_dir)

    end_time = datetime.now()
    logging.info('finished {} after {}'.format(end_time, end_time-start_time))


if __name__ == '__main__':
    main()
