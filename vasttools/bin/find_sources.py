#!/usr/bin/env python
"""
A script find sources in the VAST Pilot Survey.

Includes options to provide light curves, cutouts and overlay files for
viewers such as kvis and DS9.

Example:
    ```terminal
    find_sources "16:16:00.22 +22:16:04.83" --create-png --imsize 5.0 \
--png-zscale-contrast 0.1 --png-selavy-overlay --use-combined
    ```

Attributes:
    runstart (datetime.datetime): The running start time of the script.
"""


from astropy import units as u
from astropy.coordinates import Angle
from vasttools import RELEASED_EPOCHS, ALLOWED_PLANETS
from vasttools.query import Query
from vasttools.utils import (
    get_logger,
    build_catalog,
    build_SkyCoord,
    create_source_directories
)
import argparse
import os
import datetime
import shutil
import logging
import pandas as pd
import sys

runstart = datetime.datetime.now()


def parse_args() -> argparse.Namespace:
    """
    Parse the arguments.

    Returns:
        The argument namespace.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--coords',
        type=str,
        help=("Right Ascension and Declination in quotes. Can be formatted as"
              " \"HH:MM:SS [+/-]DD:MM:SS\" (e.g. \"12:00:00 -20:00:00\") or "
              "decimal degrees (e.g. \"12.123 -20.123\"). Multiple "
              "coordinates are supported by separating with a comma (no space)"
              " e.g. \"12.231 -56.56,123.4 +21.3\"."
              " Finally you can also enter coordinates using a .csv file."
              " See example file for format."),
        default=None)
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
        '--ncpu',
        type=int,
        help="Number of cpus to use in queries",
        default=2)
    parser.add_argument(
        '--epochs',
        type=str,
        help=("Select the VAST Pilot Epoch to query. Epoch 0 is RACS. "
              "All available epochs can be queried using "
              "'all' or all VAST Epochs using 'all-vast'. Otherwise"
              " enter as a comma separated list with no spaces, e.g."
              " '1,2,3x,4x'."),
        default="1")
    parser.add_argument(
        '--imsize',
        type=float,
        help='Edge size of the postagestamp in arcmin',
        default=5.)
    parser.add_argument(
        '--maxsep',
        type=float,
        help='Maximum separation of source from beam centre in degrees.',
        default=1.5)
    parser.add_argument(
        '--out-folder',
        type=str,
        help='Name of the output directory to place all results in.',
        default="find_sources_output_{}".format(
            runstart.strftime("%Y%m%d_%H:%M:%S")))
    parser.add_argument(
        '--crossmatch-radius',
        type=float,
        help='Crossmatch radius in arcseconds',
        default=10.0)
    parser.add_argument(
        '--search-all-fields',
        action="store_true",
        help='Return all data at the requested location(s) regardless of field'
    )
    parser.add_argument(
        '--use-combined',
        action="store_true",
        help='Use the combined mosaics instead of individual tiles.')
    parser.add_argument(
        '--uncorrected-data',
        action="store_true",
        help='Use the uncorrected data. Only relevant with --use-tiles')
    parser.add_argument(
        '--corrected-data',
        action="store_true",
        help='Use the corrected data. Only relevant with --use-tiles. '
             'Note: this is distinct from the post-processed data, which '
             'will be used by default if no data arguments are passed.'
    )
    parser.add_argument(
        '--islands',
        action="store_true",
        help='Search islands instead of components.')
    parser.add_argument(
        '--base-folder',
        type=str,
        help=(
            'Path to base folder if using default directory structure.'
            ' Not required if the `VAST_DATA_DIR` environment variable'
            ' has been set.'
        ))
    parser.add_argument(
        '--stokes',
        type=str,
        choices=["I", "Q", "U", "V"],
        help='Select the Stokes parameter.',
        default="I")
    parser.add_argument(
        '--quiet',
        action="store_true",
        help='Turn off non-essential terminal output.')
    parser.add_argument(
        '--forced-fits',
        action="store_true",
        help=(
            'Perform forced fits at the locations requested.'
        )
    )
    parser.add_argument(
        '--forced-cluster-threshold',
        type=float,
        default=1.5,
        help=(
            'Multiple of `major_axes` to use for identifying clusters,'
            ' when performing forced fits.'
        )
    )
    parser.add_argument(
        '--forced-allow-nan',
        action="store_true",
        help=(
            'When used, forced fits are attempted even when NaN'
            ' values are present.'
        )
    )
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
        '--planets',
        default=[],
        help=(
            "Also search for solar system objects. "
            "Enter as a comma separated list, e.g. 'jupiter,venus,moon'. "
            "Allowed choices are: {}".format(ALLOWED_PLANETS)))
    parser.add_argument(
        '--find-fields',
        action="store_true",
        help='Only return the associated field for each source.')
    parser.add_argument(
        '--search-around-coordinates',
        action="store_true",
        help=(
            'Return all crossmatches within the queried crossmatch radius.'
            'Plotting options will be unavailable.'
        ))
    parser.add_argument(
        '--clobber',
        action="store_true",
        help=("Overwrite the output directory if it already exists."))
    parser.add_argument(
        '--scheduler',
        default='processes',
        choices=['processes', 'single-threaded'],
        help=("Dask scheduling option to use. Options are 'processes' "
              "(parallel processing) or 'single-threaded'.")
    )
    parser.add_argument(
        '--sort-output',
        action="store_true",
        help=(
            "Place results into individual source directories within the "
            "main output directory."
        ))
    parser.add_argument(
        '--nice',
        type=int,
        help='Set nice level.',
        default=5)
    parser.add_argument(
        '--crossmatch-radius-overlay',
        action="store_true",
        help=('A circle is placed on all PNG and region/annotation'
              ' files to represent the crossmatch radius.'))
    parser.add_argument(
        '--no-fits',
        action="store_true",
        help='Do not save the FITS cutouts.')
    parser.add_argument(
        '--rms-cutouts',
        action="store_true",
        help='Create FITS files containing noisemap cutouts.')
    parser.add_argument(
        '--bkg-cutouts',
        action="store_true",
        help='Create FITS files containing meanmap cutouts.')
    parser.add_argument(
        '--plot-dpi',
        type=int,
        help="Specify the DPI of all saved figures.",
        default=150)
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
        '--png-disable-autoscaling',
        action="store_true",
        help=(
            'Do not use the auto normalisation and instead apply'
            ' scale settings to each epoch individually.'
        ))
    parser.add_argument(
        '--png-absolute-axes',
        action="store_true",
        help=(
            'Create PNGs with axis labels in absolute coordinates,'
            ' rather than offsets from the central position.'
        ))
    parser.add_argument(
        '--ann',
        action="store_true",
        help='Create a kvis annotation file of the components.')
    parser.add_argument(
        '--reg',
        action="store_true",
        help='Create a DS9 region file of the components.')
    parser.add_argument(
        '--lightcurves',
        action="store_true",
        help='Create lightcurve plots.')
    parser.add_argument(
        '--lc-use-int-flux',
        action="store_true",
        help='Use the integrated flux, rather than peak flux')
    parser.add_argument(
        '--lc-no-plotting',
        action="store_true",
        help='Write lightcurves to file without plotting')
    parser.add_argument(
        '--lc-min-points',
        type=int,
        help='Minimum number of epochs a source must be covered by',
        default=2)
    parser.add_argument(
        '--lc-min-detections',
        type=int,
        help='Minimum number of times a source must be detected',
        default=0)
    parser.add_argument(
        '--lc-mjd',
        action="store_true",
        help='Plot lightcurve in MJD rather than datetime.')
    parser.add_argument(
        '--lc-start-date',
        type=str,
        help=(
            "Plot lightcurve in days from some start date, "
            "formatted as YYYY-MM-DD HH:MM:SS or any other form "
            "that is accepted by pd.to_datetime()"
        ))
    parser.add_argument(
        '--lc-grid',
        action="store_true",
        help="Turn on the 'grid' in the lightcurve plot.")
    parser.add_argument(
        '--lc-yaxis-start',
        type=str,
        choices=["auto", "0"],
        default="0",
        help=(
            "Define where the y axis on the lightcurve plot starts from."
            " 'auto' will let matplotlib decide the best range and '0' "
            " will start from 0."
        ))
    parser.add_argument(
        '--lc-use-forced-for-limits',
        action="store_true",
        help="Use the forced fits values instead of upper limits.")
    parser.add_argument(
        '--lc-use-forced-for-all',
        action="store_true",
        help="Use the forced fits for all datapoints.")
    parser.add_argument(
        '--lc-hide-legend',
        action="store_true",
        help="Don't show the legend on the final lightcurve plot.")

    args = parser.parse_args()

    return args


def check_output_directory(args: argparse.Namespace) -> bool:
    """Creates the output directory while checking if path already exists.

    Will overwrite if user has requested the clobber option.

    Args:
        args: The argparse.Namespace object.

    Returns:
        'True' if the output directory has been created successfully.
        Otherwise, 'False'.
    """

    logger = logging.getLogger()

    output_dir = args.out_folder

    if os.path.isdir(output_dir):
        if args.clobber:
            logger.warning((
                "Directory {} already exists "
                "but clobber selected. "
                "Removing current directory."
            ).format(output_dir))
            shutil.rmtree(output_dir)
        else:
            logger.critical(
                ("Requested output directory '{}' already exists! "
                 "Will not overwrite.").format(output_dir))
            return False

    logger.info("Creating directory '{}'.".format(output_dir))
    os.mkdir(output_dir)

    return True


def main() -> None:
    """The main function.

    Returns:
        None
    """
    args = parse_args()

    post_processed_data = True
    if args.corrected_data or args.uncorrected_data:
        post_processed_data = False

    os.nice(args.nice)

    logfile = "find_sources_{}.log".format(
        runstart.strftime("%Y%m%d_%H:%M:%S")
    )

    logger = get_logger(args.debug, args.quiet, logfile=logfile)
    logger.debug(
        "Available epochs: {}".format(sorted(RELEASED_EPOCHS.keys()))
    )

    if args.corrected_data and args.uncorrected_data:
        logger.error(
            "Uncorrected and corrected data cannot be selected simultaneously"
        )
    if (args.uncorrected_data or post_processed_data) and args.use_combined:
        logger.error(
            "Uncorrected or post-processed data has been selected "
            "with COMBINED data! These modes cannot be used together, "
            "please check input and try again."
        )
    if len(args.planets) > 0:
        args.planets = args.planets.lower().replace(" ", "").split(",")

    if (args.coords is None and
            args.source_names == "" and len(args.planets) == 0):
        logger.error(
            "No coordinates or source names have been provided!"
        )
        logger.error(
            "Please check input and try again."
        )
        sys.exit()

    if args.forced_fits and args.search_around_coordinates:
        logger.error(
            "Forced fits and search around mode are both selected!"
        )
        logger.error(
            "These modes cannot be used together, "
            "please check input and try again."
        )
        sys.exit()

    if args.forced_fits and not args.use_combined:
        logger.error(
            "Forced fits and TILES are both selected!"
        )
        logger.error(
            "These modes cannot be used together, "
            "please check input and try again."
        )
        sys.exit()

    if args.lc_use_forced_for_limits or args.lc_use_forced_for_all:
        if not args.forced_fits:
            logger.error(
                "Forced fits requested for lightcurve, "
                "please use --forced-fits flag and try again."
            )
            sys.exit()

    if args.lc_start_date:
        try:
            args.lc_start_date = pd.to_datetime(args.lc_start_date)
        except Exception as e:
            logger.error(
                "Invalid lightcurve start date, "
                "please check input and try again"
            )
            logger.error("Error: %s", e)
            sys.exit()

    output_ok = check_output_directory(args)

    if not output_ok:
        logger.critical("Exiting.")
        sys.exit()

    # if this is None we'll let the Query search Simbad
    if args.coords is not None:
        catalog = build_catalog(args.coords, args.source_names)

        sky_coords = build_SkyCoord(catalog)
        source_names = catalog.name.to_list()
    elif args.source_names != "":
        catalog = pd.DataFrame(
            args.source_names.split(","),
            columns=['name']
        )
        sky_coords = None
        source_names = catalog.name.to_list()

    else:
        sky_coords = None
        source_names = ""
    logger.debug(args.epochs)

    query = Query(
        coords=sky_coords,
        source_names=source_names,
        planets=args.planets,
        epochs=args.epochs,
        stokes=args.stokes,
        crossmatch_radius=args.crossmatch_radius,
        max_sep=args.maxsep,
        use_tiles=~args.use_combined,
        use_islands=args.islands,
        base_folder=args.base_folder,
        matches_only=args.process_matches,
        no_rms=args.no_background_rms,
        output_dir=args.out_folder,
        ncpu=args.ncpu,
        search_around_coordinates=args.search_around_coordinates,
        sort_output=args.sort_output,
        forced_fits=args.forced_fits,
        forced_cluster_threshold=args.forced_cluster_threshold,
        forced_allow_nan=args.forced_allow_nan,
        incl_observed=args.find_fields,
        corrected_data=args.corrected_data,
        post_processed_data=post_processed_data,
        search_all_fields=args.search_all_fields,
        scheduler=args.scheduler,
    )

    if args.find_fields:
        query.find_fields()
        query.write_find_fields()

    # else if find sources or else find surrounding sources?
    else:
        try:
            query.find_sources()
        except Exception as e:
            logger.error(e)
            sys.exit()

        if args.search_around_coordinates:
            logger.info(
                'Search around coordinates mode selected.'
                ' No other output will be wrtten apart from the'
                ' matches csv files.'
            )
            create_source_directories(
                args.out_folder,
                query.results.name.unique()
            )
            query.save_search_around_results(args.sort_output)
        else:
            if args.sort_output:
                create_source_directories(
                    args.out_folder,
                    query.results.index.values
                )
            if args.crossmatch_only:
                fits = False
                rms = False
                bkg = False
                png = False
                ann = False
                reg = False
                lightcurve = False
            else:
                fits = (not args.no_fits)
                rms = args.rms_cutouts
                bkg = args.bkg_cutouts
                png = args.create_png
                ann = args.ann
                reg = args.reg
                lightcurve = args.lightcurves

            query._gen_all_source_products(
                fits=fits,
                rms=rms,
                bkg=bkg,
                png=png,
                ann=ann,
                reg=reg,
                lightcurve=lightcurve,
                measurements=True,
                fits_outfile=None,
                png_selavy=args.png_selavy_overlay,
                png_percentile=args.png_linear_percentile,
                png_zscale=args.png_use_zscale,
                png_contrast=args.png_zscale_contrast,
                png_no_islands=args.png_no_island_labels,
                png_no_colorbar=args.png_no_colorbar,
                png_crossmatch_overlay=args.crossmatch_radius_overlay,
                png_hide_beam=args.png_hide_beam,
                png_disable_autoscaling=args.png_disable_autoscaling,
                png_offset_axes=(not args.png_absolute_axes),
                ann_crossmatch_overlay=args.crossmatch_radius_overlay,
                reg_crossmatch_overlay=args.crossmatch_radius_overlay,
                lc_sigma_thresh=5,
                lc_figsize=(8, 4),
                lc_min_points=args.lc_min_points,
                lc_min_detections=args.lc_min_detections,
                lc_mjd=args.lc_mjd,
                lc_start_date=args.lc_start_date,
                lc_grid=args.lc_grid,
                lc_yaxis_start=args.lc_yaxis_start,
                lc_peak_flux=(not args.lc_use_int_flux),
                lc_use_forced_for_limits=args.lc_use_forced_for_limits,
                lc_use_forced_for_all=args.lc_use_forced_for_all,
                lc_hide_legend=args.lc_hide_legend,
                measurements_simple=args.selavy_simple,
                imsize=Angle(args.imsize * u.arcmin),
                plot_dpi=args.plot_dpi
            )

    query._summary_log()

    runend = datetime.datetime.now()
    runtime = runend - runstart
    logger.info(
        "Processing took {:.1f} minutes.".format(
            runtime.seconds / 60.
        )
    )


if __name__ == '__main__':
    main()
