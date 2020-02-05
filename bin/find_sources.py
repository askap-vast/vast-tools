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
import socket

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

HOST = socket.gethostname()
HOST_ADA = 'ada.physics.usyd.edu.au'

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
        type=str,
        help=("Select the VAST Pilot Epoch to query. "
              "Epoch 0 is RACS."),
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

    args = parser.parse_args()

    return args


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

    return logger


class Query:
    '''
    This is a class representation of various information about a particular
    query including the catalogue of target sources, the Stokes parameter,
    crossmatch radius and output parameters.

    :param args: Arguments namespace
    :type args: `argparse.Namespace`
    '''

    def __init__(self, args):
        '''Constructor method
        '''
        self.logger = logging.getLogger('vasttools.find_sources.Query')

        self.args = args

        self.epochs = self.get_epochs()

        self.catalog = self.build_catalog()
        self.src_coords = self.build_SkyCoord()
        self.logger.info(
            "Finding fields for {} sources...".format(len(self.src_coords)))

        self.set_stokes_param()
        self.set_outfile_prefix()
        self.set_output_directory()

        self.imsize = Angle(args.imsize, unit=u.arcmin)
        self.max_sep = args.maxsep
        self.crossmatch_radius = Angle(args.crossmatch_radius, unit=u.arcsec)

    def build_catalog(self):
        '''
        Build the catalogue of target sources

        :returns: Catalogue of target sources
        :rtype: `pandas.core.frame.DataFrame`
        '''

        if " " not in self.args.coords:
            self.logger.info("Loading file {}".format(self.args.coords))
            # Give explicit check to file existence
            user_file = os.path.abspath(self.args.coords)
            if not os.path.isfile(user_file):
                self.logger.critical("{} not found!".format(user_file))
                self.logger.critical("Exiting.")
                sys.exit()
            try:
                catalog = pd.read_csv(user_file, comment="#")
                catalog.columns = map(str.lower, catalog.columns)
                no_ra_col = "ra" not in catalog.columns
                no_dec_col = "dec" not in catalog.columns
                if no_ra_col or no_dec_col:
                    self.logger.critical(
                        "Cannot find one of 'ra' or 'dec' in input file.")
                    self.logger.critical("Please check column headers!")
                    sys.exit()
                if "name" not in catalog.columns:
                    catalog["name"] = [
                        "{}_{}".format(
                            i, j) for i, j in zip(
                            catalog['ra'], catalog['dec'])]
            except Exception as e:
                self.logger.critical(
                    "Pandas reading of {} failed!".format(self.args.coords))
                self.logger.critical("Check format!")
                sys.exit()
        else:
            catalog_dict = {'ra': [], 'dec': []}
            coords = self.args.coords.split(",")
            for i in coords:
                ra_str, dec_str = i.split(" ")
                catalog_dict['ra'].append(ra_str)
                catalog_dict['dec'].append(dec_str)

            if self.args.source_names != "":
                source_names = self.args.source_names.split(",")
                if len(source_names) != len(catalog_dict['ra']):
                    self.logger.critical(
                        ("All sources must be named "
                         "when using '--source-names'."))
                    self.logger.critical("Please check inputs.")
                    sys.exit()
            else:
                source_names = [
                    "{}_{}".format(
                        i, j) for i, j in zip(
                        catalog_dict['ra'], catalog_dict['dec'])]

            catalog_dict['name'] = source_names

            catalog = pd.DataFrame.from_dict(catalog_dict)

        catalog['name'] = catalog['name'].astype(str)

        return catalog

    def build_SkyCoord(self):
        '''
        Create a SkyCoord array for each target source

        :returns: Target source SkyCoord
        :rtype: `astropy.coordinates.sky_coordinate.SkyCoord`
        '''

        if self.catalog['ra'].dtype == np.float64:
            hms = False
            deg = True

        elif ":" in self.catalog['ra'].iloc[0]:
            hms = True
            deg = False
        else:
            deg = True
            hms = False

        if hms:
            src_coords = SkyCoord(
                self.catalog['ra'],
                self.catalog['dec'],
                unit=(
                    u.hourangle,
                    u.deg))
        else:
            src_coords = SkyCoord(
                self.catalog['ra'],
                self.catalog['dec'],
                unit=(
                    u.deg,
                    u.deg))

        return src_coords

    def get_epochs(self):
        '''
        Parse the list of epochs to query.

        :returns: Epochs to query, as a list of string
        :rtype: list
        '''

        available_epochs = ["0", ] + sorted(RELEASED_EPOCHS)
        epochs = []

        for epoch in self.args.vast_pilot.split(','):
            if epoch in available_epochs:
                epochs.append(epoch)
            else:
                self.logger.info(
                    "Epoch {} is not available. Ignoring.".format(epoch))

        if len(epochs) == 0:
            self.logger.critical("No requested epochs are available")
            sys.exit()

        return epochs

    def set_output_directory(self):
        '''
        Build the output directory and store the path
        '''

        output_dir = self.args.out_folder
        if os.path.isdir(output_dir):
            if self.args.clobber:
                self.logger.warning(("Directory {} already exists "
                                     "but clobber selected. "
                                     "Removing current directory."
                                     ).format(output_dir))
                shutil.rmtree(output_dir)
            else:
                self.logger.critical(
                    ("Requested output directory '{}' already exists! "
                     "Will not overwrite.").format(output_dir))
                self.logger.critical("Exiting.")
                sys.exit()

        self.logger.info("Creating directory '{}'.".format(output_dir))
        os.mkdir(output_dir)

        self.output_dir = output_dir

    def set_stokes_param(self):
        '''
        Set the stokes Parameter
        '''

        if self.args.stokesv:
            stokes_param = "V"
        else:
            stokes_param = "I"

        self.stokes_param = stokes_param

    def set_outfile_prefix(self):
        '''
        Return general parameters of the requested survey

        :returns: prefix for output file
        :rtype: str
        '''

        if self.args.stokesv and self.args.use_tiles:
            self.logger.critical(
                ("Stokes V can only be used "
                 "with combined mosaics at the moment."))
            self.logger.critical(
                "Run again but remove the option '--use-tiles'.")
            sys.exit()

        if self.args.use_tiles:
            outfile_prefix = "tile"
        else:
            outfile_prefix = "combined"
            if self.args.stokesv:
                outfile_prefix += "_stokesv"

        self.outfile_prefix = outfile_prefix

    def run_query(self):
        '''
        Run the requested query
        '''

        for epoch in self.epochs:
            self.run_epoch(epoch)

    def run_epoch(self, epoch):
        '''
        Query a specific epoch

        :param epoch: The epoch to query
        :type epoch: str
        '''

        EPOCH_INFO = EpochInfo(self.args, epoch, self.stokes_param)
        survey = EPOCH_INFO.survey
        epoch_str = EPOCH_INFO.epoch_str
        self.logger.info("Querying {}".format(epoch_str))

        fields = Fields(epoch)
        src_fields, coords_mask = fields.find(
            self.src_coords, self.max_sep, self.catalog)

        src_coords_field = self.src_coords[coords_mask]

        uniq_fields = src_fields['field_name'].unique().tolist()

        if len(uniq_fields) == 0:
            self.logger.error("Source(s) not in Survey!")
            return

        if EPOCH_INFO.FIND_FIELDS:
            if survey == "racs":
                fields_cat_file = "{}_racs_fields.csv".format(self.output_dir)
            else:
                fields_cat_file = "{}_VAST_{}_fields.csv".format(
                    self.output_dir, epoch)

            fields_cat_file = os.path.join(self.output_dir, fields_cat_file)
            fields.write_fields_cat(fields_cat_file)

            return

        crossmatch_output_check = False

        self.logger.info(
            "Performing crossmatching for sources, please wait...")

        for uf in uniq_fields:
            self.logger.info(
                "-----------------------------------------------------")

            mask = src_fields["field_name"] == uf
            srcs = src_fields[mask]
            indexes = srcs.index
            srcs = srcs.reset_index()
            field_src_coords = src_coords_field[mask]

            if survey == "vast_pilot":
                fieldname = "{}.EPOCH{}.{}".format(
                    uf, RELEASED_EPOCHS[epoch], self.stokes_param)
            else:
                fieldname = uf

            image = Image(srcs["sbid"].iloc[0],
                          fieldname,
                          EPOCH_INFO.IMAGE_FOLDER,
                          EPOCH_INFO.RMS_FOLDER,
                          epoch,
                          tiles=self.args.use_tiles)

            if not self.args.no_background_rms:
                image.get_rms_img()

            for i, row in srcs.iterrows():
                SBID = row['sbid']

                number = row["original_index"] + 1

                label = row["name"]

                self.logger.info(
                    "Searching for crossmatch to source {}".format(label))

                outfile = "{}_{}_{}.fits".format(
                    label.replace(" ", "_"), fieldname, self.outfile_prefix)
                outfile = os.path.join(self.output_dir, outfile)

                src_coord = field_src_coords[i]

                source = Source(
                    fieldname,
                    src_coord,
                    SBID,
                    EPOCH_INFO.SELAVY_FOLDER,
                    vast_pilot=epoch,
                    tiles=self.args.use_tiles,
                    stokesv=self.args.stokesv)

                source.extract_source(
                    self.crossmatch_radius, self.args.stokesv)
                if not self.args.no_background_rms and not image.rms_fail:
                    source.get_background_rms(image.rms_data, image.rms_wcs)

                if self.args.process_matches and not source.has_match:
                    self.logger.info("Source does not have a selavy match, "
                                     "not continuing processing")
                    continue
                else:
                    crossmatch_only = self.args.crossmatch_only
                    if not crossmatch_only and not image.image_fail:
                        source.make_postagestamp(
                            image.data,
                            image.header,
                            image.wcs,
                            self.imsize,
                            outfile)

                    # not ideal but line below has to be run after those above
                    crossmatch_overlay = self.args.crossmatch_radius_overlay
                    if source.selavy_fail is False:
                        source.filter_selavy_components(self.imsize)
                        if self.args.ann:
                            source.write_ann(
                                outfile,
                                crossmatch_overlay=crossmatch_overlay)
                        if self.args.reg:
                            source.write_reg(
                                outfile,
                                crossmatch_overlay=crossmatch_overlay)
                    else:
                        self.logger.error(
                            "Selavy failed! No region or annotation files "
                            "will be made if requested.")

                    if self.args.create_png:
                        if not crossmatch_only and not image.image_fail:
                            if survey == "racs":
                                png_title = "{} RACS {}".format(
                                    label,
                                    uf.split("_")[-1]
                                )
                            else:
                                png_title = "{} VAST Pilot {} Epoch {}".format(
                                    label,
                                    uf.split("_")[-1],
                                    epoch
                                )
                            source.make_png(
                                self.args.png_selavy_overlay,
                                self.args.png_linear_percentile,
                                self.args.png_use_zscale,
                                self.args.png_zscale_contrast,
                                outfile,
                                image.beam,
                                no_islands=self.args.png_no_island_labels,
                                label=label,
                                no_colorbar=self.args.png_no_colorbar,
                                title=png_title,
                                crossmatch_overlay=crossmatch_overlay,
                                hide_beam=self.args.png_hide_beam)

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
                    self.logger.debug("Crossmatch df:\n{}".format(df_info))
                    buffer = io.StringIO()
                    source.selavy_info.info(buf=buffer)
                    df_info = buffer.getvalue()
                    self.logger.debug("Selavy info df:\n{}".format(df_info))
                    crossmatch_output = crossmatch_output.append(
                        source.selavy_info, sort=False)
                logger.info(
                    "-----------------------------------------------------")

        runend = datetime.datetime.now()
        runtime = runend - runstart

        self.logger.info(
            "-----------------------------------------------------")
        self.logger.info("Summary")
        self.logger.info(
            "-----------------------------------------------------")
        self.logger.info("Number of sources searched for: {}".format(
            len(self.catalog.index)))
        self.logger.info("Number of sources in survey: {}".format(
            len(src_fields.index)))

        matched = crossmatch_output[~crossmatch_output["island_id"].isna()]
        num_matched = len(matched.index)
        self.logger.info((
            "Number of sources with matches"
            " < {} arcsec: {}").format(
                                    self.crossmatch_radius.arcsec,
                                    num_matched))

        logger.info(
            "Processing took {:.1f} minutes.".format(
                runtime.seconds / 60.))

        # Create and write final crossmatch csv
        if self.args.selavy_simple:
            crossmatch_output = crossmatch_output.filter(
                items=["flux_int", "rms_image", "BANE_rms"])
            crossmatch_output = crossmatch_output.rename(
                columns={"flux_int": "S_int", "rms_image": "S_err"})

        final = src_fields.join(crossmatch_output)

        output_crossmatch_name = "{}_crossmatch_{}.csv".format(
            self.output_dir, epoch_str)
        output_crossmatch_name = os.path.join(
            self.output_dir, output_crossmatch_name)
        final.to_csv(output_crossmatch_name, index=False)
        logger.info("Written {}.".format(output_crossmatch_name))
        logger.info("All results in {}.".format(self.output_dir))


class EpochInfo:
    '''
    This is a class representation of various information about a particular
    epoch query including the relevant folders, whether to only find fields,
    the survey and epoch.

    :param args: Arguments namespace
    :type args: `argparse.Namespace`
    :param pilot_epoch: Pilot epoch (0 for RACS)
    :type pilot_epoch: str
    :param stokes_param: Stokes parameter (I or V)
    :type stokes_param: str
    '''

    def __init__(self, args, pilot_epoch, stokes_param):
        self.logger = logging.getLogger('vasttools.find_sources.EpochInfo')

        FIND_FIELDS = args.find_fields
        if FIND_FIELDS:
            self.logger.info(
                "find-fields selected, only outputting field catalogue")

        BASE_FOLDER = args.base_folder
        IMAGE_FOLDER = args.img_folder
        SELAVY_FOLDER = args.cat_folder
        RMS_FOLDER = args.rms_folder

        self.use_tiles = args.use_tiles
        self.pilot_epoch = pilot_epoch
        self.stokes_param = stokes_param

        racsv = False

        if pilot_epoch == "0":
            survey = "racs"
            epoch_str = "RACS"
            if not BASE_FOLDER:
                survey_folder = "RACS/release/racs_v3/"
            else:
                survey_folder = "racs_v3"

            if stokes_param == "V":
                self.logger.critical(
                    "Stokes V is currently unavailable for RACS V3."
                    "Using V2 instead")
                racsv = True
        else:
            survey = "vast_pilot"
            epoch_str = "EPOCH{}".format(RELEASED_EPOCHS[pilot_epoch])
            if not BASE_FOLDER:
                survey_folder = "PILOT/release/{}".format(epoch_str)
            else:
                survey_folder = epoch_str

        self.survey = survey
        self.epoch_str = epoch_str
        self.survey_folder = survey_folder
        self.racsv = racsv

        if not BASE_FOLDER:
            if HOST != HOST_ADA:
                self.logger.critical(
                    "Base folder must be specified if not running on ada")
                sys.exit()
            BASE_FOLDER = "/import/ada1/askap/"

        if not IMAGE_FOLDER:
            if self.use_tiles:
                image_dir = "FLD_IMAGES/"
                stokes_dir = "stokesI"
            else:
                image_dir = "COMBINED"
                stokes_dir = "STOKES{}_IMAGES".format(stokes_param)

            IMAGE_FOLDER = os.path.join(
                BASE_FOLDER,
                survey_folder,
                image_dir,
                stokes_dir)

            if self.racsv:
                IMAGE_FOLDER = ("/import/ada1/askap/RACS/aug2019_reprocessing/"
                                "COMBINED_MOSAICS/V_mosaic_1.0")

        if not os.path.isdir(IMAGE_FOLDER):
            if not FIND_FIELDS:
                self.logger.critical(
                    ("{} does not exist. "
                     "Only finding fields").format(IMAGE_FOLDER))
                FIND_FIELDS = True

        if not SELAVY_FOLDER:
            image_dir = "COMBINED"
            selavy_dir = "STOKES{}_SELAVY".format(stokes_param)

            SELAVY_FOLDER = os.path.join(
                BASE_FOLDER,
                survey_folder,
                image_dir,
                selavy_dir)
            if self.use_tiles:
                SELAVY_FOLDER = ("/import/ada1/askap/RACS/aug2019_"
                                 "reprocessing/SELAVY_OUTPUT/stokesI_cat/")

            if racsv:
                SELAVY_FOLDER = ("/import/ada1/askap/RACS/aug2019_"
                                 "reprocessing/COMBINED_MOSAICS/racs_catv")

        if not os.path.isdir(SELAVY_FOLDER):
            if not FIND_FIELDS:
                self.logger.critical(
                    ("{} does not exist. "
                     "Only finding fields").format(SELAVY_FOLDER))
                FIND_FIELDS = True

        if not RMS_FOLDER:
            if self.use_tiles:
                self.logger.warning(
                    "Background noise estimates are not supported for tiles.")
                self.logger.warning(
                    "Estimating background from mosaics instead.")
            image_dir = "COMBINED"
            rms_dir = "STOKES{}_RMSMAPS".format(stokes_param)

            RMS_FOLDER = os.path.join(
                BASE_FOLDER,
                survey_folder,
                image_dir,
                rms_dir)

            if racsv:
                RMS_FOLDER = ("/import/ada1/askap/RACS/aug2019_reprocessing/"
                              "COMBINED_MOSAICS/V_mosaic_1.0_BANE")

        if not os.path.isdir(RMS_FOLDER):
            if not FIND_FIELDS:
                self.logger.critical(
                    ("{} does not exist. "
                     "Only finding fields").format(RMS_FOLDER))
                FIND_FIELDS = True

        self.FIND_FIELDS = FIND_FIELDS
        self.IMAGE_FOLDER = IMAGE_FOLDER
        self.SELAVY_FOLDER = SELAVY_FOLDER
        self.RMS_FOLDER = RMS_FOLDER


if __name__ == '__main__':
    args = parse_args()
    logger = get_logger(args, use_colorlog=use_colorlog)
    logger.debug("Available epochs: {}".format(RELEASED_EPOCHS.keys()))

    query = Query(args)
    query.run_query()
