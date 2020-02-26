#!/usr/bin/env python

# Example command:

# ./build_lightcurves.py

from vasttools.analysis import Lightcurve, BuildLightcurves

import argparse
import sys
import numpy as np
import os
import glob
import datetime
import pandas as pd
import warnings

import logging
import logging.handlers
import logging.config

import matplotlib
import matplotlib.pyplot as plt

from astropy.coordinates import Angle
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.nddata.utils import Cutout2D
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time
from astropy.timeseries import TimeSeries
from astropy.wcs.utils import skycoord_to_pixel
from astropy.utils.exceptions import AstropyWarning, AstropyDeprecationWarning

from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection
from astropy.visualization import ZScaleInterval, ImageNormalize
from astropy.visualization import PercentileInterval
from astropy.visualization import AsymmetricPercentileInterval
from astropy.visualization import LinearStretch
import matplotlib.axes as maxes
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

warnings.filterwarnings('ignore', category=AstropyWarning, append=True)
warnings.filterwarnings('ignore',
                        category=AstropyDeprecationWarning, append=True)

matplotlib.pyplot.switch_backend('Agg')

try:
    import colorlog
    use_colorlog = True
except ImportError:
    use_colorlog = False

# Force nice
# os.nice(5)

runstart = datetime.datetime.now()


def get_logger(args, use_colorlog=False):
    '''
    Set up the logger

    :param args: Arguments namespace
    :type args: `argparse.Namespace`
    :param usecolorlog: Use colourful logging scheme, defaults to False
    :type usecolorlog: bool, optional

    :returns: Logger
    :rtype: `logging.RootLogger`
    '''

    logger = logging.getLogger()
    s = logging.StreamHandler()
    fh = logging.FileHandler(
        "build_lightcurves_{}.log".format(
            runstart.strftime("%Y%m%d_%H:%M:%S")))
    fh.setLevel(logging.DEBUG)
    logformat = '[%(asctime)s] - %(levelname)s - %(message)s'

    if use_colorlog:
        formatter = colorlog.ColoredFormatter(
            "%(log_color)s[%(asctime)s] - %(levelname)s - %(blue)s%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            reset=True,
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white', },
            secondary_log_colors={},
            style='%'
        )
    else:
        formatter = logging.Formatter(logformat, datefmt="%Y-%m-%d %H:%M:%S")

    s.setFormatter(formatter)
    fh.setFormatter(formatter)

    if args.debug:
        s.setLevel(logging.DEBUG)
    else:
        if args.quiet:
            s.setLevel(logging.WARNING)
        else:
            s.setLevel(logging.INFO)

    logger.addHandler(s)
    logger.addHandler(fh)
    logger.setLevel(logging.DEBUG)


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

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    logger = get_logger(args, use_colorlog=use_colorlog)

    query = BuildLightcurves(args)
    query.run_query()
