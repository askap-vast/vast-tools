#!/usr/bin/env python

# Example command:

# ./find_sources.py "16:16:00.22 +22:16:04.83" --create-png --imsize 5.0
# --png-zscale-contrast 0.1 --png-selavy-overlay --use-combined
from vasttools.survey import Fields, Image
from vasttools.survey import RELEASED_EPOCHS
from vasttools.source import Source
from vasttools.query import Query, EpochInfo
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
        'coords',
        type=str,
        help=("Right Ascension and Declination in quotes. Can be formatted as "
              "\"HH:MM:SS [+/-]DD:MM:SS\" (e.g. \"12:00:00 -20:00:00\") or "
              "decimal degrees (e.g. \"12.123 -20.123\"). Multiple "
              "coordinates are supported by separating with a comma (no space)"
              " e.g. \"12.231 -56.56,123.4 +21.3\"."
              " Finally you can also enter coordinates using a .csv file."
              " See example file for format."))

    parser.add_argument(
        '--vast-pilot',
        type=str,
        help=("Select the VAST Pilot Epoch to query. Epoch 0 is RACS. "
              "All available epochs can be queried using "
              "\"--vast-pilot=all\""),
        default="1")
    parser.add_argument(
        '--imsize',
        type=float,
        help='Edge size of the postagestamp in arcmin',
        default=30.)
    parser.add_argument(
        '--maxsep',
        type=float,
        help='Maximum separation of source from beam centre in degrees.',
        default=1.0)
    parser.add_argument(
        '--out-folder',
        type=str,
        help='Name of the output directory to place all results in.',
        default="find_sources_output_{}".format(
            runstart.strftime("%Y%m%d_%H:%M:%S")))
    parser.add_argument(
        '--source-names',
        type=str,
        help=("Only for use when entering coordaintes via the command line. "
              "State the name of the source being searched. "
              "Use quote marks for names that contain a space. "
              "For multiple sources separate with a comma with no "
              "space, e.g. \"SN 1994N,SN 2003D,SN 2019A\"."),
        default="")
    parser.add_argument(
        '--crossmatch-radius',
        type=float,
        help='Crossmatch radius in arcseconds',
        default=15.0)
    parser.add_argument(
        '--crossmatch-radius-overlay',
        action="store_true",
        help=('A circle is placed on all PNG and region/annotation'
              ' files to represent the crossmatch radius.'))
    parser.add_argument(
        '--use-tiles',
        action="store_true",
        help='Use the individual tiles instead of combined mosaics.')
    parser.add_argument(
        '--islands',
        action="store_true",
        help='Search islands instead of components.')
    parser.add_argument(
        '--base-folder',
        type=str,
        help='Path to base folder if using default directory structure')
    parser.add_argument(
        '--img-folder',
        type=str,
        help='Path to folder where images are stored')
    parser.add_argument(
        '--rms-folder',
        type=str,
        help='Path to folder where image RMS estimates are stored')
    parser.add_argument(
        '--cat-folder',
        type=str,
        help='Path to folder where selavy catalogues are stored')
    parser.add_argument(
        '--create-png',
        action="store_true",
        help='Create a png of the fits cutout.')
    parser.add_argument(
        '--png-selavy-overlay',
        action="store_true",
        help='Overlay selavy components onto the png image.')
    parser.add_argument(
        '--png-linear-percentile',
        type=float,
        default=99.9,
        help='Choose the percentile level for the png normalisation.')
    parser.add_argument(
        '--png-use-zscale',
        action="store_true",
        help='Select ZScale normalisation (default is \'linear\').')
    parser.add_argument(
        '--png-zscale-contrast',
        type=float,
        default=0.1,
        help='Select contrast to use for zscale.')
    parser.add_argument(
        '--png-hide-beam',
        action="store_true",
        help='Select to not show the image synthesised beam on the plot.')
    parser.add_argument(
        '--png-no-island-labels',
        action="store_true",
        help='Disable island lables on the png.')
    parser.add_argument(
        '--png-no-colorbar',
        action="store_true",
        help='Do not show the colorbar on the png.')
    parser.add_argument(
        '--ann',
        action="store_true",
        help='Create a kvis annotation file of the components.')
    parser.add_argument(
        '--reg',
        action="store_true",
        help='Create a DS9 region file of the components.')
    parser.add_argument(
        '--stokesv',
        action="store_true",
        help='Use Stokes V images and catalogues if available.')
    parser.add_argument(
        '--quiet',
        action="store_true",
        help='Turn off non-essential terminal output.')
    parser.add_argument(
        '--crossmatch-only',
        action="store_true",
        help='Only run crossmatch, do not generate any fits or png files.')
    parser.add_argument(
        '--selavy-simple',
        action="store_true",
        help='Only include flux density and uncertainty in returned table.')
    parser.add_argument(
        '--process-matches',
        action="store_true",
        help='Only produce data products for sources with a selavy match.')
    parser.add_argument(
        '--debug',
        action="store_true",
        help='Turn on debug output.')
    parser.add_argument(
        '--no-background-rms',
        action="store_true",
        help='Do not estimate the background RMS around each source.')
    parser.add_argument(
        '--find-fields',
        action="store_true",
        help='Only return the associated field for each source.')
    parser.add_argument(
        '--clobber',
        action="store_true",
        help=("Overwrite the output directory if it already exists."))
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

    logfile = "find_sources_{}.log".format(
        runstart.strftime("%Y%m%d_%H:%M:%S"))
    logger = get_logger(args.debug, args.quiet, logfile=logfile)
    logger.debug("Available epochs: {}".format(sorted(RELEASED_EPOCHS.keys())))

    query = Query(args)
    query.run_query()

    runend = datetime.datetime.now()
    runtime = runend - runstart
    logger.info(
        "Processing took {:.1f} minutes.".format(
            runtime.seconds / 60.))
