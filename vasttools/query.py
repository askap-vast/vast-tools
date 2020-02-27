from vasttools.survey import Fields, Image
from vasttools.survey import RELEASED_EPOCHS
from vasttools.survey import FIELD_FILES
from vasttools.source import Source

import sys
import numpy as np
import os
import datetime
import pandas as pd
import warnings
import shutil
import io
import socket
import re

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

from radio_beam import Beams

from tabulate import tabulate

warnings.filterwarnings('ignore', category=AstropyWarning, append=True)
warnings.filterwarnings('ignore',
                        category=AstropyDeprecationWarning, append=True)

HOST = socket.gethostname()
HOST_ADA = 'ada.physics.usyd.edu.au'


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
                self.logger.debug(catalog)
                catalog.columns = map(str.lower, catalog.columns)
                self.logger.debug(catalog.columns)
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
            catalog = catalog[['name', 'ra', 'dec']]

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

        available_epochs = sorted(RELEASED_EPOCHS, key=RELEASED_EPOCHS.get)
        self.logger.debug(available_epochs)

        if HOST == HOST_ADA:
            available_epochs.insert(0, "0")

        epochs = []

        epoch_arg = self.args.vast_pilot
        if epoch_arg == 'all':
            return available_epochs

        for epoch in epoch_arg.split(','):
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
        self.logger.info(
            "-----------------------------------------------------")
        self.logger.info("Query executed successfully!")
        self.logger.info("All results in {}.".format(self.output_dir))

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
            self.logger.error(
                "Source(s) not in {}!".format(EPOCH_INFO.epoch_str))
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
                    self.logger.debug(src_coord)
                    self.logger.debug(image.rmspath)
                    self.logger.debug(image.imgpath)
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
                self.logger.info(
                    "-----------------------------------------------------")

        self.logger.info(
            "-----------------------------------------------------")
        self.logger.info("Epoch Summary ({})".format(epoch_str))
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
        self.logger.info("Written {}.".format(output_crossmatch_name))
        self.logger.info(
            "-----------------------------------------------------")


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


class FieldQuery:
    '''
    This is a class representation of a query of the VAST Pilot survey
    fields, returning basic information such as observation dates and psf
    information.

    :param args: Arguments namespace
    :type args: `argparse.Namespace`
    '''

    def __init__(self, field):
        '''Constructor method
        '''
        self.logger = logging.getLogger('vasttools.query.FieldQuery')

        self.field = field
        self.valid = self._check_field()

    def _check_field(self):
        '''
        Check that the field is a valid pilot survey field.
        Epoch 1 is checked against as it is a complete observation.

        :returns: Bool representing if field is valid.
        :rtype: bool.
        '''

        epoch_01 = pd.read_csv(FIELD_FILES["1"])
        self.logger.debug("Field name: {}".format(self.field))
        result = epoch_01['FIELD_NAME'].str.contains(
            re.escape(self.field)
        ).any()
        self.logger.debug("Field found: {}".format(result))
        if result is False:
            self.logger.error(
                "Field {} is not a valid field name!".format(self.field)
            )
        del epoch_01
        return result

    def _get_beams(self):
        '''
        Processes all the beams of a field per epoch and initialises
        radio_beam.Beams objects.

        :returns: Dictionary of 'radio_beam.Beams' objects.
        :rtype: dict.
        '''
        epoch_beams = {}
        for e in self.epochs:
            epoch_cut = self.field_info[self.field_info.EPOCH == e]
            epoch_beams[e] = Beams(
                epoch_cut.BMAJ.values * u.arcsec,
                epoch_cut.BMIN.values * u.arcsec,
                epoch_cut.BPA.values * u.deg
            )
        return epoch_beams

    def run_query(
            self,
            largest_psf=False,
            common_psf=False,
            all_psf=False,
            save=False,
            _pilot_info=None):
        '''
        Running the field query.

        :param largest_psf: If true the largest psf  is calculated
            of the field per epoch. Defaults to False.
        :type largest_psf: bool, optional
        :param common_psf: If true the common psf is calculated
            of the field per epoch. Defaults to False.
        :type common_psf: bool, optional
        :param all_psf: If true the common psf is calculated
            of the field per epoch and all the beam information of
            the field is shown. Defaults to False.
        :type all_psf: bool, optional
        :param save: Save the output tables to a csv file. Defaults
            to False.
        :type save: bool, optional
        :param _pilot_info: Allows for the pilot info to be provided
            rather than the function building it locally. If not provided
            then the dataframe is built. Defaults to None.
        :type _pilot_info: pandas.DataFrame, optional
        '''
        if not self.valid:
            self.logger.error("Field doesn't exist.")
            return

        if _pilot_info is not None:
            self.pilot_info = _pilot_info
        else:
            self.logger.debug("Building pilot info file.")
            for i, val in enumerate(sorted(RELEASED_EPOCHS)):
                if i == 0:
                    self.pilot_info = pd.read_csv(FIELD_FILES[val])
                    self.pilot_info["EPOCH"] = RELEASED_EPOCHS[val]
                else:
                    to_append = pd.read_csv(FIELD_FILES[val])
                    to_append["EPOCH"] = RELEASED_EPOCHS[val]
                    self.pilot_info = self.pilot_info.append(
                        to_append, sort=False
                    )

        self.field_info = self.pilot_info[
            self.pilot_info.FIELD_NAME == self.field
        ]

        self.field_info.reset_index(drop=True, inplace=True)

        self.field_info = self.field_info.filter([
            "EPOCH",
            "FIELD_NAME",
            "SBID",
            "BEAM",
            "RA_HMS",
            "DEC_DMS",
            "DATEOBS",
            "DATEEND",
            "BMAJ",
            "BMIN",
            "BPA"
        ])

        self.field_info.sort_values(by=["EPOCH", "BEAM"], inplace=True)

        self.epochs = self.field_info.EPOCH.unique()

        if largest_psf or common_psf or all_psf:
            self.logger.info("Getting psf information.")
            epoch_beams = self._get_beams()

        if all_psf:
            common_beams = {}
            self.logger.info("Calculating common psfs...")
            for i in sorted(epoch_beams):
                common_beams[i] = epoch_beams[i].common_beam()

            self.logger.info("{} information:".format(self.field))

            print(tabulate(
                self.field_info,
                headers=self.field_info.columns,
                showindex=False
            ))

            table = []

            for i in sorted(epoch_beams):
                table.append([
                    self.field,
                    i,
                    common_beams[i].major.to(u.arcsec).value,
                    common_beams[i].minor.to(u.arcsec).value,
                    common_beams[i].pa.to(u.deg).value
                ])

            self.logger.info("Common psf for {}".format(self.field))

            print(tabulate(table, headers=[
                "FIELD",
                "EPOCH",
                "BMAJ (arcsec)",
                "BMIN (arcsec)",
                "BPA (degree)"
            ]))

            if save:
                common_df = pd.DataFrame(table, columns=[
                    "FIELD",
                    "EPOCH",
                    "BMAJ (arcsec)",
                    "BMIN (arcsec)",
                    "BPA (degree)"
                ])
                savename = "{}_field_info_common_psf.csv".format(self.field)
                common_df.to_csv(savename, index=False)
                self.logger.info("Saved common psf output to {}.".format(
                    savename
                ))

        else:
            self.field_info = self.field_info.filter([
                "EPOCH",
                "FIELD_NAME",
                "SBID",
                "RA_HMS",
                "DEC_DMS",
                "DATEOBS",
                "DATEEND",
            ])

            self.field_info.rename(columns={
                "RA_HMS": "RA_HMS (Beam 0)",
                "DEC_DMS": "DEC_DMS (Beam 0)",
            }, inplace=True)

            self.field_info.drop_duplicates("EPOCH", inplace=True)
            if largest_psf:
                largest_beams = []
                for i in sorted(epoch_beams):
                    largest_beams.append(epoch_beams[i].largest_beam())

                self.field_info["L_BMAJ (arcsec)"] = [
                    b.major.value for b in largest_beams
                ]
                self.field_info["L_BMIN (arcsec)"] = [
                    b.minor.value for b in largest_beams
                ]
                self.field_info["L_BPA (deg)"] = [
                    b.pa.value for b in largest_beams
                ]

            elif common_psf:
                common_beams = []
                self.logger.info("Calculating common psfs...")
                for i in sorted(epoch_beams):
                    common_beams.append(epoch_beams[i].common_beam())

                self.field_info["C_BMAJ (arcsec)"] = [
                    b.major.value for b in common_beams
                ]
                self.field_info["C_BMIN (arcsec)"] = [
                    b.minor.value for b in common_beams
                ]
                self.field_info["C_BPA (deg)"] = [
                    b.pa.value for b in common_beams
                ]

            self.logger.info("{} information:".format(self.field))

            print(tabulate(
                self.field_info,
                headers=self.field_info.columns,
                showindex=False
            ))

        if save:
            savename = "{}_field_info.csv".format(self.field)
            self.field_info.to_csv(savename, index=False)
            self.logger.info("Saved output to {}.".format(savename))
