#!/usr/bin/env python

# Example command:

# ./build_lightcurves.py outputfolder

from vasttools.analysis import BuildLightcurves
from vasttools.utils import get_logger

import argparse
import os

import datetime

runstart = datetime.datetime.now()


def parse_args():
    '''
    Parse arguments

    :returns: Argument namespace
    :rtype: `argparse.Namespace`
    '''

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        'folder',
        type=str,
        help='')
    parser.add_argument(
        '--no-plotting',
        action="store_true",
        help='Write lightcurves to file without plotting')
    parser.add_argument(
        '--quiet',
        action="store_true",
        help='Turn off non-essential terminal output.')
    parser.add_argument(
        '--debug',
        action="store_true",
        help='Turn on debug output.')
    parser.add_argument(
        '--min-points',
        type=int,
        help='Minimum number of epochs a source must be covered by',
        default=2)
    parser.add_argument(
        '--min-detections',
        type=int,
        help='Minimum number of times a source must be detected',
        default=1)
    parser.add_argument(
        '--mjd',
        action="store_true",
        help='Plot lightcurve in MJD rather than datetime.')
    parser.add_argument(
        '--grid',
        action="store_true",
        help="Turn on the 'grid' in the lightcurve plot.")
    parser.add_argument(
        '--yaxis-start',
        type=str,
        choices=["auto", "0"],
        default="auto",
        help=(
            "Define where the y axis on the lightcurve plot starts from."
            " 'auto' will let matplotlib decide the best range and '0' "
            " will start from 0."
        ))
    parser.add_argument(
        '--nice',
        type=int,
        help='Set nice level.',
        default=5)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    os.nice(args.nice)

    logfile = "build_lightcurves_{}.log".format(
        runstart.strftime("%Y%m%d_%H:%M:%S"))
    logger = get_logger(args.debug, args.quiet, logfile=logfile)

    query = BuildLightcurves(args)
    query.run_query()

    runend = datetime.datetime.now()
    runtime = runend - runstart
    logger.info(
        "Processing took {:.1f} minutes.".format(
            runtime.seconds / 60.))
