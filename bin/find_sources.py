#!/usr/bin/env python

# Example command:

# ./find_sources.py "16:16:00.22 +22:16:04.83" --create-png --imsize 5.0
# --png-zscale-contrast 0.1 --png-selavy-overlay --use-combined
from vasttools.survey import Fields, Image
from vasttools.survey import RELEASED_EPOCHS
from vasttools.source import Source

import argparse
import sys
import numpy as np
import os
import datetime
import pandas as pd
import warnings
import shutil
import io

import logging
import logging.handlers
import logging.config

import matplotlib.pyplot as plt

from astropy.coordinates import Angle
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.nddata.utils import Cutout2D
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from astropy.utils.exceptions import AstropyWarning, AstropyDeprecationWarning

from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection
from astropy.visualization import ZScaleInterval, ImageNormalize
from astropy.visualization import PercentileInterval
from astropy.visualization import AsymmetricPercentileInterval
from astropy.visualization import LinearStretch
import matplotlib.axes as maxes
from mpl_toolkits.axes_grid1 import make_axes_locatable

warnings.filterwarnings('ignore', category=AstropyWarning, append=True)
warnings.filterwarnings('ignore',
                        category=AstropyDeprecationWarning, append=True)


try:
    import colorlog
    use_colorlog = True
except ImportError:
    use_colorlog = False

# Force nice
os.nice(5)

runstart = datetime.datetime.now()

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    'coords',
    metavar="\"HH:MM:SS [+/-]DD:MM:SS\" OR input.csv",
    type=str,
    help=("Right Ascension and Declination in format "
          "\"HH:MM:SS [+/-]DD:MM:SS\", in quotes. "
          "E.g. \"12:00:00 -20:00:00\". Degrees is also acceptable, "
          "e.g. \"12.123 -20.123\". Multiple coordinates are supported "
          "by separating with a comma (no space) e.g. "
          "\"12.231 -56.56,123.4 +21.3\"."
          " Finally you can also enter coordinates using a .csv file."
          " See example file for format."))

parser.add_argument(
    '--vast-pilot',
    choices=["0", ]+sorted(RELEASED_EPOCHS),
    help=("Select the VAST Pilot Epoch to query. "
          "Epoch 0 is RACS."),
    default=1)
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
    '--png-no-island-labels',
    action="store_true",
    help='Disable island lables on the png.')
parser.add_argument(
    '--png-ellipse-pa-corr',
    type=float,
    help=("Correction to apply to ellipse position angle if needed (in deg). "
          "Angle is from x-axis from left to right."),
    default=0.0)
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
    help='Only produce data products for sources that have a selavy match.')
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

args = parser.parse_args()

logger = logging.getLogger()
s = logging.StreamHandler()
fh = logging.FileHandler(
    "find_sources_{}.log".format(
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

# Sort out output directory
output_name = args.out_folder
if os.path.isdir(output_name):
    if args.clobber:
        logger.warning(("Directory {} already exists "
                        "but clobber selected. "
                        "Removing current directory."
                        ).format(output_name))
        shutil.rmtree(output_name)
    else:
        logger.critical(
            ("Requested output directory '{}' already exists! "
             "Will not overwrite.").format(output_name))
        logger.critical("Exiting.")
        sys.exit()

logger.info("Creating directory '{}'.".format(output_name))
os.mkdir(output_name)

if " " not in args.coords:
    logger.info("Loading file {}".format(args.coords))
    # Give explicit check to file existence
    user_file = os.path.abspath(args.coords)
    if not os.path.isfile(user_file):
        logger.critical("{} not found!".format(user_file))
        logger.critical("Exiting.")
        sys.exit()
    try:
        catalog = pd.read_csv(user_file, comment="#")
        catalog.columns = map(str.lower, catalog.columns)
        if ("ra" not in catalog.columns) or ("dec" not in catalog.columns):
            logger.critical("Cannot find one of 'ra' or 'dec' in input file.")
            logger.critical("Please check column headers!")
            sys.exit()
        if "name" not in catalog.columns:
            catalog["name"] = [
                "{}_{}".format(
                    i, j) for i, j in zip(
                    catalog['ra'], catalog['dec'])]
    except Exception as e:
        logger.critical("Pandas reading of {} failed!".format(args.coords))
        logger.critical("Check format!")
        sys.exit()
else:
    catalog_dict = {'ra': [], 'dec': []}
    coords = args.coords.split(",")
    for i in coords:
        ra_str, dec_str = i.split(" ")
        catalog_dict['ra'].append(ra_str)
        catalog_dict['dec'].append(dec_str)

    if args.source_names != "":
        source_names = args.source_names.split(",")
        if len(source_names) != len(catalog_dict['ra']):
            logger.critical(
                "All sources must be named when using '--source-names'.")
            logger.critical("Please check inputs.")
            sys.exit()
    else:
        source_names = [
            "{}_{}".format(
                i, j) for i, j in zip(
                catalog_dict['ra'], catalog_dict['dec'])]

    catalog_dict['name'] = source_names

    catalog = pd.DataFrame.from_dict(catalog_dict)

catalog['name'] = catalog['name'].astype(str)

imsize = Angle(args.imsize, unit=u.arcmin)

max_sep = args.maxsep

if args.use_tiles:
    outfile_prefix = "tile"
else:
    outfile_prefix = "combined"
    if args.stokesv:
        outfile_prefix += "_stokesv"

crossmatch_radius = Angle(args.crossmatch_radius, unit=u.arcsec)

if args.stokesv and args.use_tiles:
    logger.critical(
        "Stokes V can only be used with combined mosaics at the moment.")
    logger.critical("Run again but remove the option '--use-tiles'.")
    sys.exit()

if args.stokesv:
    stokes_param = "V"
else:
    stokes_param = "I"

FIND_FIELDS = args.find_fields
if FIND_FIELDS:
    logger.info("find-fields selected, only outputting field catalogue")

pilot_epoch = args.vast_pilot
if pilot_epoch == "0":
    survey = "racs"
    survey_folder = "RACS/aug2019_reprocessing"
else:
    # This currently works, but we should include a csv for each epoch to
    # ensure complete correctness
    survey = "vast_pilot"
    epoch_str = "EPOCH{}".format(pilot_epoch)
    survey_folder = "PILOT/release/{}".format(epoch_str)

default_base_folder = "/import/ada1/askap/"

IMAGE_FOLDER = args.img_folder
if not IMAGE_FOLDER:
    if args.use_tiles:
        image_dir = "FLD_IMAGES/"
        stokes_dir = "stokesI"
    else:
        if args.vast_pilot:
            image_dir = "COMBINED"
            stokes_dir = "STOKES{}_IMAGES".format(stokes_param)
        else:
            image_dir = "COMBINED_MOSAICS"
            stokes_dir = "{}_mosaic_1.0".format(stokes_param)

    IMAGE_FOLDER = os.path.join(
        default_base_folder,
        survey_folder,
        image_dir,
        stokes_dir)

if not os.path.isdir(IMAGE_FOLDER):
    if not FIND_FIELDS:
        logger.critical(
            "{} does not exist. Only finding fields".format(IMAGE_FOLDER))
        FIND_FIELDS = True


SELAVY_FOLDER = args.cat_folder
if not SELAVY_FOLDER:
    if args.use_tiles:
        SELAVY_FOLDER = ("/import/ada1/askap/RACS/aug2019_reprocessing/"
                         "SELAVY_OUTPUT/stokesI_cat/")
    else:
        if args.vast_pilot:
            image_dir = "COMBINED"
            selavy_dir = "STOKES{}_SELAVY".format(stokes_param)
        else:
            image_dir = "COMBINED_MOSAICS"
            selavy_dir = "racs_cat"
            if args.stokesv:
                selavy_dir += "v"

    SELAVY_FOLDER = os.path.join(
        default_base_folder,
        survey_folder,
        image_dir,
        selavy_dir)

if not os.path.isdir(SELAVY_FOLDER):
    if not FIND_FIELDS:
        logger.critical(
            "{} does not exist. Only finding fields".format(SELAVY_FOLDER))
        FIND_FIELDS = True

RMS_FOLDER = args.rms_folder
if not RMS_FOLDER:
    if args.use_tiles:
        logger.warning(
            "Background noise estimates are not supported for tiles.")
        logger.warning("Estimating background from mosaics instead.")
    if args.vast_pilot:
        image_dir = "COMBINED"
        rms_dir = "STOKES{}_RMSMAPS".format(stokes_param)
    else:
        image_dir = "COMBINED_MOSAICS"
        rms_dir = "{}_mosaic_1.0_BANE".format(stokes_param)

    RMS_FOLDER = os.path.join(
        default_base_folder,
        survey_folder,
        image_dir,
        rms_dir)

if not os.path.isdir(RMS_FOLDER):
    if not FIND_FIELDS:
        logger.critical(
            "{} does not exist. Only finding fields".format(RMS_FOLDER))
        FIND_FIELDS = True

if catalog['ra'].dtype == np.float64:
    hms = False
    deg = True

elif ":" in catalog['ra'].iloc[0]:
    hms = True
    deg = False
else:
    deg = True
    hms = False

if hms:
    src_coords = SkyCoord(
        catalog['ra'],
        catalog['dec'],
        unit=(
            u.hourangle,
            u.deg))
else:
    src_coords = SkyCoord(catalog['ra'], catalog['dec'], unit=(u.deg, u.deg))

logger.info("Finding fields for {} sources...".format(len(src_coords)))
logger.debug("Using epoch {}".format(pilot_epoch))
fields = Fields(pilot_epoch)
src_fields, coords_mask = fields.find(src_coords, max_sep, catalog)

src_coords = src_coords[coords_mask]

uniq_fields = src_fields['field_name'].unique().tolist()

if len(uniq_fields) == 0:
    logger.error("Source(s) not in Survey!")
    sys.exit()

if FIND_FIELDS:
    if survey == "racs":
        fields_cat_file = "{}_racs_fields.csv".format(output_name)
    else:
        fields_cat_file = "{}_VAST_{}_fields.csv".format(
            output_name, pilot_epoch)

    fields_cat_file = os.path.join(output_name, fields_cat_file)
    fields.write_fields_cat(fields_cat_file)
    sys.exit()

crossmatch_output_check = False

logger.info("Performing crossmatching for sources, please wait...")

for uf in uniq_fields:
    logger.info("-----------------------------------------------------------")

    mask = src_fields["field_name"] == uf
    srcs = src_fields[mask]
    indexes = srcs.index
    srcs = srcs.reset_index()
    field_src_coords = src_coords[mask]

    if args.vast_pilot:
        fieldname = "{}.{}.{}".format(uf, epoch_str, stokes_param)
    else:
        fieldname = uf

    image = Image(srcs["sbid"].iloc[0], fieldname,
                  IMAGE_FOLDER, RMS_FOLDER, tiles=args.use_tiles)

    if not args.no_background_rms:
        image.get_rms_img()

    for i, row in srcs.iterrows():
        SBID = row['sbid']

        number = row["original_index"] + 1

        label = row["name"]

        logger.info("Searching for crossmatch to source {}".format(label))

        outfile = "{}_{}_{}.fits".format(
            label.replace(" ", "_"), fieldname, outfile_prefix)
        outfile = os.path.join(output_name, outfile)

        src_coord = field_src_coords[i]

        source = Source(
            fieldname,
            src_coord,
            SBID,
            SELAVY_FOLDER,
            vast_pilot=args.vast_pilot,
            tiles=args.use_tiles,
            stokesv=args.stokesv)

        source.extract_source(crossmatch_radius, args.stokesv)
        if not args.no_background_rms and not image.rms_fail:
            source.get_background_rms(image.rms_data, image.rms_wcs)

        if args.process_matches and not source.has_match:
            logger.info("Source does not have a selavy match, not "
                        "continuing processing")
            continue
        else:
            if not args.crossmatch_only and not image.image_fail:
                source.make_postagestamp(
                    image.data,
                    image.header,
                    image.wcs,
                    imsize,
                    outfile)

            # not ideal but line below has to be run after those above
            if source.selavy_fail is False:
                source.filter_selavy_components(imsize)
                if args.ann:
                    source.write_ann(
                        outfile,
                        crossmatch_overlay=args.crossmatch_radius_overlay)
                if args.reg:
                    source.write_reg(
                        outfile,
                        crossmatch_overlay=args.crossmatch_radius_overlay)
            else:
                logger.error(
                    "Selavy failed! No region or annotation files "
                    "will be made if requested.")

            if args.create_png:
                if not args.crossmatch_only and not image.image_fail:
                    if survey == "racs":
                        png_title = "{} RACS {}".format(
                            label,
                            uf.split("_")[-1]
                        )
                    else:
                        png_title = "{} VAST Pilot {} Epoch {}".format(
                            label,
                            uf.split("_")[-1],
                            pilot_epoch
                        )
                    source.make_png(
                        args.png_selavy_overlay,
                        args.png_linear_percentile,
                        args.png_use_zscale,
                        args.png_zscale_contrast,
                        outfile,
                        args.png_ellipse_pa_corr,
                        no_islands=args.png_no_island_labels,
                        label=label,
                        no_colorbar=args.png_no_colorbar,
                        title=png_title,
                        crossmatch_overlay=args.crossmatch_radius_overlay)

        if not crossmatch_output_check:
            crossmatch_output = source.selavy_info
            crossmatch_output.index = [indexes[i]]
            crossmatch_output_check = True
        else:
            temp_crossmatch_output = source.selavy_info
            temp_crossmatch_output.index = [indexes[i]]
            buffer = io.StringIO()
            crossmatch_output.info(buf=buffer)
            df_info = buffer.getvalue()
            logger.debug("Crossmatch df:\n{}".format(df_info))
            buffer = io.StringIO()
            source.selavy_info.info(buf=buffer)
            df_info = buffer.getvalue()
            logger.debug("Selavy info df:\n{}".format(df_info))
            crossmatch_output = crossmatch_output.append(
                source.selavy_info, sort=False)
        logger.info(
            "-----------------------------------------------------------")

runend = datetime.datetime.now()
runtime = runend - runstart

logger.info("-----------------------------------------------------------")
logger.info("Summary")
logger.info("-----------------------------------------------------------")
logger.info("Number of sources searched for: {}".format(len(catalog.index)))
logger.info("Number of sources in survey: {}".format(len(src_fields.index)))
logger.info("Number of sources with matches < {} arcsec: {}".format(
    crossmatch_radius.arcsec,
    len(crossmatch_output[~crossmatch_output["island_id"].isna()].index)))

logger.info("Processing took {:.1f} minutes.".format(runtime.seconds / 60.))

# Create and write final crossmatch csv
if args.selavy_simple:
    crossmatch_output = crossmatch_output.filter(
        items=["flux_int", "rms_image", "BANE_rms"])
    crossmatch_output = crossmatch_output.rename(
        columns={"flux_int": "S_int", "rms_image": "S_err"})

final = src_fields.join(crossmatch_output)

output_crossmatch_name = "{}_crossmatch.csv".format(output_name)
output_crossmatch_name = os.path.join(output_name, output_crossmatch_name)
final.to_csv(output_crossmatch_name, index=False)
logger.info("Written {}.".format(output_crossmatch_name))
logger.info("All results in {}.".format(output_name))
