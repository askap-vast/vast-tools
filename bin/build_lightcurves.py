#!/usr/bin/env python

# Example command:

# ./build_lightcurves.py outputfolder

from vasttools.utils import get_logger

import argparse
import os
# import numexpr
import datetime
import glob
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u
from vasttools.source import Source

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
        '--use-int-flux',
        action="store_true",
        help='Use the integrated flux, rather than peak flux')
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
        default="0",
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


def load_sources(folder):
    """docstring for get_source_files"""

    files = sorted(
        glob.glob(os.path.join(folder, '*_measurements.csv'))
    )

    sources = [
        source_from_measurements_file(i, folder) for i in files
    ]

    return sources


def source_from_measurements_file(measurement_file, outdir):
    measurements = pd.read_csv(measurement_file)

    m = measurements.iloc[0]

    source_coord = SkyCoord(m.ra, m.dec, unit=(u.deg, u.deg))
    source_name = m['name']
    source_epochs = measurements['epoch'].to_list()
    source_fields = measurements['field'].to_list()
    source_stokes = m['stokes']
    source_primary_field = m['primary_field']
    source_base_folder = None
    source_crossmatch_radius = None
    source_outdir = outdir
    source_image_type = "UNKNOWN"

    thesource = Source(
        source_coord,
        source_name,
        source_epochs,
        source_fields,
        source_stokes,
        source_primary_field,
        source_crossmatch_radius,
        measurements,
        source_base_folder,
        source_image_type,
        outdir=source_outdir
    )

    return thesource


if __name__ == '__main__':
    args = parse_args()
    os.nice(args.nice)

    # numexpr.set_num_threads(2)
    args = parse_args()

    logfile = "build_lightcurves_{}.log".format(
        runstart.strftime("%Y%m%d_%H:%M:%S"))
    logger = get_logger(args.debug, args.quiet, logfile=logfile)

    sources = load_sources(args.folder)

    for s in sources:
        s.plot_lightcurve(
            min_points=args.min_points,
            min_detections=args.min_detections,
            mjd=args.mjd,
            grid=args.grid,
            yaxis_start=args.yaxis_start,
            peak_flux=(not args.use_int_flux),
            save=True
        )
        logger.info('Saving %s lightcurve.', s.name)

    runend = datetime.datetime.now()
    runtime = runend - runstart
    logger.info(
        "Processing took {:.1f} minutes.".format(
            runtime.seconds / 60.))
