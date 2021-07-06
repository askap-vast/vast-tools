#!/usr/bin/env python

# Example command:

# ./build_lightcurves.py outputfolder

from vasttools.query import FieldQuery
from vasttools.utils import get_logger

import argparse
import os
import pandas as pd
import datetime
import sys

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
        'fields',
        type=str,
        nargs="+",
        help='Fields to query (or csv file containing fields).')
    parser.add_argument(
        '--psf',
        action="store_true",
        help=(
            'Include the used PSF of the 36'
            ' beams that make up the field.'
            ' Usually set from beam 00.'
            ))
    parser.add_argument(
        '--largest-psf',
        action="store_true",
        help=(
            'Include the largest PSF of the 36'
            ' beams that make up the field.'
            ))
    parser.add_argument(
        '--common-psf',
        action="store_true",
        help=(
            'Include the common PSF of the 36'
            ' beams that make up the field.'
            ))
    parser.add_argument(
        '--all-psf',
        action="store_true",
        help='Include all the PSF information for the field.')
    parser.add_argument(
        '--save',
        action="store_true",
        help=(
            "Save the resulting information."
            " Files will be saved to the current working directory"
            " in the form of 'VAST_XXXX+/-XXA_field_info.csv'."
        ))
    parser.add_argument(
        '--quiet',
        action="store_true",
        help='Turn off non-essential terminal output.')
    parser.add_argument(
        '--debug',
        action="store_true",
        help='Turn on debug output.')
    parser.add_argument(
        '--nice',
        type=int,
        help='Set nice level.',
        default=5)

    args = parser.parse_args()

    return args


def read_fields(fields_file):
    fields = pd.read_csv(fields_file, comment='#')
    try:
        fields = fields.field_name.to_list()
    except Exception as e:
        logger.error("Could not find column 'field_name' in file.")
        fields = []
    return fields


def main():
    args = parse_args()
    os.nice(args.nice)

    logfile = "pilot_fields_info_{}.log".format(
        runstart.strftime("%Y%m%d_%H:%M:%S"))
    logger = get_logger(args.debug, args.quiet, logfile=logfile)

    if len(args.fields) == 1:
        file_name = args.fields[0]
        if os.path.isfile(file_name):
            logger.info("Input file detected - reading file...")
            fields = read_fields(file_name)
            if len(fields) == 0:
                logger.error(
                    "Failed to read any fields from {}!".format(
                        file_name
                    )
                )
                sys.exit()
        else:
            fields = args.fields
    else:
        fields = args.fields

    if sum([args.psf, args.largest_psf, args.common_psf]) > 1:
        logger.warning(
            "More than one psf option has been selected."
            " Please correct and re-run."
        )
        sys.exit()

    logger.info("Will find information for the following fields:")
    [logger.info(i) for i in fields]

    for i, field in enumerate(fields):
        query = FieldQuery(field)
        if i == 0:
            query.run_query(
                psf=args.psf,
                largest_psf=args.largest_psf,
                common_psf=args.common_psf,
                all_psf=args.all_psf,
                save=args.save
            )
            try:
                pilot_info = query.pilot_info
            except Exception as e:
                continue
        else:
            query.run_query(
                psf=args.psf,
                largest_psf=args.largest_psf,
                common_psf=args.common_psf,
                all_psf=args.all_psf,
                save=args.save,
                _pilot_info=pilot_info
            )

    runend = datetime.datetime.now()
    runtime = runend - runstart
    logger.info(
        "Processing took {:.1f} minutes.".format(
            runtime.seconds / 60.))


if __name__ == '__main__':
    main()
