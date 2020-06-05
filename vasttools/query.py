from vasttools.survey import Fields, Image
from vasttools.survey import (
    RELEASED_EPOCHS, FIELD_FILES, NIMBUS_BASE_DIR, EPOCH_FIELDS
)
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

    def __init__(
        self, coords, source_names=[], epochs="all", stokes="I",
        crossmatch_radius=5.0, max_sep=1.0, use_tiles=False,
        use_islands=False, base_folder=None, matches_only=False,
        no_rms=False, output_dir="."
    ):
        '''Constructor method
        '''
        self.logger = logging.getLogger('vasttools.find_sources.Query')

        self.coords = coords
        self.source_names = source_names

        self.query_df = self.build_catalog_2()

        self.settings = {}

        self.settings['epochs'] = self.get_epochs(epochs)
        self.settings['stokes'] = self.get_stokes(stokes)

        self.settings['crossmatch_radius'] = Angle(
            crossmatch_radius, unit=u.arcsec
        )
        self.settings['max_sep'] = max_sep

        self.settings['islands'] = use_islands
        self.settings['tiles'] = use_tiles
        self.settings['no_rms'] = no_rms

        self.settings['output_dir'] = output_dir

        if base_folder is None:
            self.base_folder = NIMBUS_BASE_DIR
        else:
            self.base_folder = base_folder

        if not os.path.isdir(self.base_folder):
            raise ValueError("The base directory {} does not exist!".format(
                self.base_folder
            ))

        self.fields_found = False

        # self.catalog = self.build_catalog()
        # self.src_coords = self.build_SkyCoord()
        # self.logger.info(
        #     "Finding fields for {} sources...".format(len(self.src_coords)))

        # if output is not None:
        #     self.set_outfile_prefix()
        #     self.set_output_directory()

        # self.imsize = Angle(args.imsize, unit=u.arcmin)
        # self.max_sep = args.maxsep
        # self.crossmatch_radius = Angle(args.crossmatch_radius, unit=u.arcsec)


    def run_new_query(self):
        self.find_fields()

        self.query_df[["result"]] = self.query_df.apply(
            self.find_sources,
            axis=1
        )


    def create_all_cutouts(self, ):
        # first get cutout data and selavy sources per image
        # group by image to do this

        grouped_query = self.query_df.groupby('image')
        cutouts = grouped_query.apply(
            lambda x: self._grouped_fetch_cutouts(x.name, x)
        )

        cutouts.index = cutouts.index.droplevel()

        self.query_df = self.query_df.join(
            cutouts
        )

        self.results = self.results.apply(
            self._add_source_cutout_data,
        )

        self.results.apply(
            self._save_all_png_cutouts,
        )



    def _save_all_png_cutouts(self, s):

        s.save_all_png_cutouts()



    def _add_source_cutout_data(self, s):
        s_name = s.name
        s_cutout = self.query_df[[
            'data',
            'wcs',
            'header',
            'selavy_overlay',
            'beam'
        ]][self.query_df.name == s_name].reset_index(drop=True)

        s.cutout_df = s_cutout
        s._cutouts_got = True

        del s_cutout

        return s


    def _grouped_fetch_cutouts(self, image_file, group):

        image = Image(
            group.iloc[0].field,
            group.iloc[0].epoch,
            self.settings['stokes'],
            self.base_folder,
            sbid=group.iloc[0].sbid
        )

        cutout_data = group.apply(
            self._get_cutout,
            args=(image,),
            axis=1,
            result_type='expand'
        ).rename(columns={
            0: "data",
            1: "wcs",
            2: "header",
            3: "selavy_overlay",
            4: "beam"
        })

        del image

        return cutout_data

    def _get_cutout(self, row, image, size=Angle(5. * u.arcmin)):

        cutout = Cutout2D(
            image.data,
            position=row.skycoord,
            size=size,
            wcs=image.wcs
        )

        selavy_components = pd.read_fwf(row.selavy, skiprows=[1,], usecols=[
            'island_id',
            'ra_deg_cont',
            'dec_deg_cont',
            'maj_axis',
            'min_axis',
            'pos_ang'
        ])

        selavy_coords = SkyCoord(
            selavy_components.ra_deg_cont.values * u.deg,
            selavy_components.dec_deg_cont.values * u.deg
        )

        selavy_components = self.filter_selavy_components_2(
            selavy_components,
            selavy_coords,
            size,
            row.skycoord
        )

        header = image.header.copy()
        header.update(cutout.wcs.to_header())

        beam = image.beam

        del selavy_coords

        return (
            cutout.data, cutout.wcs, header, selavy_components, beam
        )

    #this needs to be moved to a general script!
    def filter_selavy_components_2(self, selavy_df, selavy_sc, imsize, target):
        '''
        Create a shortened catalogue by filtering out selavy components
        outside of the image

        :param imsize: Size of the image along each axis
        :type imsize: `astropy.coordinates.angles.Angle` or tuple of two
            `Angle` objects
        '''

        seps = target.separation(selavy_sc)
        mask = seps <= imsize / 1.4
        return selavy_df[mask].reset_index(drop=True)


    def find_sources(self):
        if self.fields_found is False:
            self.find_fields()

        self.query_df = self.query_df.explode(
            'field_per_epoch'
        ).reset_index(drop=True)
        self.query_df[['epoch', 'field', 'sbid', 'dateobs']] = self.query_df.field_per_epoch.apply(pd.Series)
        self.query_df[['selavy', 'image', 'rms']] = self.query_df[['field_per_epoch']].apply(
            self._add_files,
            axis=1,
            result_type='expand'
        )

        grouped_query = self.query_df.groupby('selavy')
        results = grouped_query.apply(
            lambda x: self._get_components(x.name, x)
        )
        results.index = results.index.droplevel()
        self.crossmatch_results = self.query_df.merge(
            results, how='left', left_index=True, right_index=True
        )

        grouped_source_query = self.crossmatch_results.groupby('name')

        self.results = grouped_source_query.apply(
            lambda x: self._init_sources(x.name, x)
        )


    def _init_sources(self, source_name, group):
        # master = {}

        m = group.iloc[0]

        source_coord = m.skycoord
        source_name = m['name']
        source_epochs = m.epochs
        source_fields = m.fields
        source_stokes = self.settings['stokes']
        source_primary_field = m.primary_field
        source_base_folder = self.base_folder
        source_crossmatch_radius = self.settings['crossmatch_radius']
        source_outdir = self.settings['output_dir']
        if self.settings['tiles']:
            source_image_type = "TILES"
        else:
            source_image_type = "COMBINED"
        source_islands=self.settings['islands']

        source_df = group.drop(
            columns=[
                'ra',
                'dec',
                'fields',
                'epochs',
                'field_per_epoch',
                'sbids',
                'dates',
                '#'
            ]
        )

        source_df = source_df.reset_index(drop=True)

        thesource = Source(
            source_coord,
            source_name,
            source_epochs,
            source_fields,
            source_stokes,
            source_primary_field,
            source_crossmatch_radius,
            source_df,
            source_base_folder,
            source_image_type,
            islands=source_islands,
            outdir=source_outdir
        )

        return thesource


    def _get_components(self, selavy_file, group):
        master = pd.DataFrame()

        selavy_df = pd.read_fwf(
            selavy_file, skiprows=[1,]
        )

        selavy_coords = SkyCoord(
            selavy_df.ra_deg_cont * u.deg,
            selavy_df.dec_deg_cont * u.deg
        )
        group_coords = SkyCoord(
            group.ra * u.deg,
            group.dec * u.deg
        )

        idx, d2d, _ = group_coords.match_to_catalog_sky(selavy_coords)
        mask = d2d < self.settings['crossmatch_radius']
        idx_matches = idx[mask]

        copy = selavy_df.iloc[idx_matches].reset_index(drop=True)
        copy["detection"] = True
        copy.index = group[mask].index.values

        master = master.append(copy, sort=False)

        missing = group_coords[~mask]
        if missing.shape[0] > 0 and self.settings['no_rms'] == False:
            image = Image(
                group.iloc[0].field,
                group.iloc[0].epoch,
                self.settings['stokes'],
                self.base_folder,
                sbid=group.iloc[0].sbid
            )
            rms_values = image.measure_coord_pixel_values(
                missing, rms=True
            )
            rms_df = pd.DataFrame(rms_values, columns=['rms_image'])

            # to mJy
            rms_df['rms_image'] = rms_df['rms_image'] * 1.e3
            rms_df['detection'] = False

            rms_df.index = group[~mask].index.values

            master = master.append(rms_df, sort=False)

        return master


    def _add_files(self, row):
        if self.settings['islands']:
            cat_type = 'islands'
        else:
            cat_type = 'components'
        if self.settings['tiles']:
            pass
        else:
            selavy_file = os.path.join(
                self.base_folder,
                (
                    "EPOCH{0}/"
                    "COMBINED/"
                    "STOKES{1}_SELAVY/"
                    "{2}.EPOCH{0}.{1}.selavy.{3}.txt".format(
                        RELEASED_EPOCHS[row.field_per_epoch[0]],
                        self.settings['stokes'],
                        row.field_per_epoch[1],
                        cat_type
                    )
                )
            )
            image_file = os.path.join(
                self.base_folder,
                (
                    "EPOCH{0}/"
                    "COMBINED/"
                    "STOKES{1}_IMAGES/"
                    "{2}.EPOCH{0}.{1}.fits".format(
                        RELEASED_EPOCHS[row.field_per_epoch[0]],
                        self.settings['stokes'],
                        row.field_per_epoch[1]
                    )
                )
            )
            rms_file = os.path.join(
                self.base_folder,
                (
                    "EPOCH{0}/"
                    "COMBINED/"
                    "STOKES{1}_RMSMAPS/"
                    "{2}.EPOCH{0}.{1}_rms.fits".format(
                        RELEASED_EPOCHS[row.field_per_epoch[0]],
                        self.settings['stokes'],
                        row.field_per_epoch[1]
                    )
                )
            )

        return selavy_file, image_file, rms_file


    def find_fields(self):
        fields = Fields('1')

        self.query_df[[
            'fields',
            'primary_field',
            'epochs',
            'field_per_epoch',
            'sbids',
            'dates'
        ]] = self.query_df.apply(
            self._field_matching,
            args=(fields.direction, fields.fields.FIELD_NAME),
            axis=1,
            result_type='expand'
        )

        self.query_df = self.query_df.dropna()
        self.fields_found = True


    def _field_matching(self, row, fields_coords, fields_names):
        seps = row.skycoord.separation(fields_coords)
        accept = seps.deg < self.settings['max_sep']
        fields = np.unique(fields_names[accept])
        if fields.shape[0] == 0:
            warnings.warn(
                "Source '{}' not in selected footprint. Dropping source.".format(
                    row['name']
                )
            )
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        primary_field = fields[0]
        epochs = []
        field_per_epochs = []
        sbids = []
        dateobs = []
        for i in self.settings['epochs']:
            epoch_fields = EPOCH_FIELDS[i].keys()
            for j in fields:
                if j in epoch_fields:
                    epochs.append(i)
                    sbid = EPOCH_FIELDS[i][j]["SBID"]
                    date = EPOCH_FIELDS[i][j]["DATEOBS"]
                    sbids.append(sbid)
                    dateobs.append(date)
                    field_per_epochs.append([i,j,sbid,date])
                    break

        return fields, primary_field, epochs, field_per_epochs, sbids, dateobs


    def build_catalog_2(self):
        cols = ['ra', 'dec', 'name', 'skycoord']
        if self.coords.shape == ():
            print('here')
            catalog = pd.DataFrame(
                [[
                    self.coords.ra.deg,
                    self.coords.dec.deg,
                    self.source_names[0],
                    self.coords
                ]], columns = cols
            )
        else:
            catalog = pd.DataFrame(
                self.source_names,
                columns = ['name']
            )
            catalog['ra'] = self.coords.ra.deg
            catalog['dec'] = self.coords.dec.deg
            catalog['skycoord'] = self.coords

        return catalog


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
                catalog.dropna(how="all", inplace=True)
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

        ra_str = self.catalog['ra'].iloc[0]
        if self.catalog['ra'].dtype == np.float64:
            hms = False
            deg = True

        elif ":" in ra_str or " " in ra_str:
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

    def get_epochs(self, req_epochs):
        '''
        Parse the list of epochs to query.

        :returns: Epochs to query, as a list of string
        :rtype: list
        '''

        available_epochs = sorted(RELEASED_EPOCHS, key=RELEASED_EPOCHS.get)
        self.logger.debug(available_epochs)

        # if HOST == HOST_ADA or self.args.find_fields:
        #     available_epochs.insert(0, "0")

        if req_epochs == 'all':
            return available_epochs

        epochs = []
        for epoch in req_epochs.split(','):
            if epoch in available_epochs:
                epochs.append(epoch)
            else:
                if self.logger is None:
                    self.logger.info(
                        "Epoch {} is not available. Ignoring.".format(epoch)
                    )
                else:
                    warnings.warn(
                        "Removing Epoch {} as it"
                        " is not a valid epoch.".format(epoch),
                        stacklevel=2
                    )

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

    def get_stokes(self, req_stokes):
        '''
        Set the stokes Parameter
        '''
        valid = ["I", "Q", "U", "V"]

        if req_stokes.upper() not in valid:
            raise ValueError(
                "Stokes {} is not valid!".format(req_stokes.upper())
            )
        else:
            return req_stokes.upper()


    def set_outfile_prefix(self):
        '''
        Return general parameters of the requested survey

        :returns: prefix for output file
        :rtype: str
        '''

        if self.stokes_param != "I" and self.args.use_tiles:
            self.logger.critical(
                ("Only Stokes I tiles can be queried right now."))
            self.logger.critical(
                "Run again but remove the option '--use-tiles'.")
            sys.exit()

        if self.args.use_tiles:
            outfile_prefix = "tile"
        else:
            outfile_prefix = "combined"
            if self.stokes_param != "I":
                outfile_prefix += "_stokes{}".format(
                    self.stokes_param.lower())

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

        EPOCH_INFO = EpochInfo(
             epoch, self.base_folder, self.stokes, self.tiles
        )
        survey = EPOCH_INFO.survey
        epoch_str = EPOCH_INFO.epoch_str
        self.logger.info("Querying {}".format(epoch_str))
        crossmatch_only = False

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

            if crossmatch_only:
                self.logger.warning(
                    "Crossmatch only mode selected."
                    " Ignore any possible image errors below."
                )
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
                    stokes=self.stokes_param,
                    islands=self.args.islands)

                source.extract_source(
                    self.crossmatch_radius)
                if not self.args.no_background_rms and not image.rms_fail:
                    self.logger.debug(src_coord)
                    self.logger.debug(image.rmspath)
                    self.logger.debug(image.imgpath)
                    source.get_background_rms(image.rms_data, image.rms_wcs)
                else:
                    source.selavy_info["SELAVY_rms"] = 0.0

                if self.args.process_matches and not source.has_match:
                    self.logger.info("Source does not have a selavy match, "
                                     "not continuing processing")
                    continue
                else:
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

    def __init__(
        self, pilot_epoch, base_folder, stokes, tiles
    ):
        self.logger = logging.getLogger('vasttools.find_sources.EpochInfo')

        # FIND_FIELDS = find_fields
        # CROSSMATCH_ONLY = args.crossmatch_only
        # if FIND_FIELDS:
        #     self.logger.info(
        #         "find-fields selected, only outputting field catalogue."
        #     )
        # elif CROSSMATCH_ONLY:
        #     self.logger.info(
        #         "crossmatch only selected, only outputting crossmatches."
        #     )

        BASE_FOLDER = base_folder

        self.use_tiles = tiles
        self.pilot_epoch = pilot_epoch
        self.stokes_param = stokes

        racsv = False

        if pilot_epoch == "0":
            survey = "racs"
            epoch_str = "RACS"
            if not BASE_FOLDER:
                survey_folder = "RACS/release/racs_v3/"
            else:
                survey_folder = "racs_v3"

            # if stokes_param == "V":
            #    self.logger.critical(
            #        "Stokes V is currently unavailable for RACS V3. "
            #        "Using V2 instead")
            #    racsv = True
            if stokes_param != "I":
                self.logger.critical(
                    "Stokes {} is currently unavailable for RACS".format(
                        self.stokes_param))
                sys.exit()

        else:
            survey = "vast_pilot"
            epoch_str = "EPOCH{}".format(RELEASED_EPOCHS[pilot_epoch])
            survey_folder = os.path.join(
                base_folder, "{}".format(epoch_str)
            )

        self.survey = survey
        self.epoch_str = epoch_str
        self.survey_folder = survey_folder
        self.racsv = racsv

        # already checked
        # if not BASE_FOLDER:
        #     if HOST != HOST_ADA:
        #         if not FIND_FIELDS:
        #             self.logger.critical(
        #                 "Base folder must be specified if not running on ada")
        #             sys.exit()
        #     BASE_FOLDER = "/import/ada1/askap/"


        if self.use_tiles:
            image_dir = "TILES"
            stokes_dir = "STOKES{}_IMAGES".format(self.stokes_param)
        else:
            image_dir = "COMBINED"
            stokes_dir = "STOKES{}_IMAGES".format(self.stokes_param)

        IMAGE_FOLDER = os.path.join(
            BASE_FOLDER,
            survey_folder,
            image_dir,
            stokes_dir)

            # if self.racsv:
            #     IMAGE_FOLDER = ("/import/ada1/askap/RACS/aug2019_reprocessing/"
            #                     "COMBINED_MOSAICS/V_mosaic_1.0")

        if not os.path.isdir(IMAGE_FOLDER):
            # if not CROSSMATCH_ONLY:
            self.logger.warning(
                "{} does not exist. "
                "Can only do crossmatching.".format(IMAGE_FOLDER)
            )
                # CROSSMATCH_ONLY = True

        if self.use_tiles:
            self.logger.warning(
                "Background noise estimates are not supported for tiles.")
            self.logger.warning(
                "Estimating background from mosaics instead.")
        image_dir = "COMBINED"
        rms_dir = "STOKES{}_RMSMAPS".format(self.stokes_param)

        RMS_FOLDER = os.path.join(
            BASE_FOLDER,
            survey_folder,
            image_dir,
            rms_dir)

            # if racsv:
            #     RMS_FOLDER = ("/import/ada1/askap/RACS/aug2019_reprocessing/"
            #                   "COMBINED_MOSAICS/V_mosaic_1.0_BANE")

        if not os.path.isdir(RMS_FOLDER):
            # if not CROSSMATCH_ONLY:
            self.logger.critical(
                ("{} does not exist. "
                 "Switching to crossmatch only.").format(RMS_FOLDER))
                # CROSSMATCH_ONLY = True

        image_dir = "COMBINED"
        selavy_dir = "STOKES{}_SELAVY".format(self.stokes_param)

        SELAVY_FOLDER = os.path.join(
            BASE_FOLDER,
            survey_folder,
            image_dir,
            selavy_dir
        )

            # if self.use_tiles:
            #     SELAVY_FOLDER = ("/import/ada1/askap/RACS/aug2019_"
            #                      "reprocessing/SELAVY_OUTPUT/stokesI_cat/")
            #
            # if racsv:
            #     SELAVY_FOLDER = ("/import/ada1/askap/RACS/aug2019_"
            #                      "reprocessing/COMBINED_MOSAICS/racs_catv")

        if not os.path.isdir(SELAVY_FOLDER):
            # if not FIND_FIELDS and not CROSSMATCH_ONLY:
            self.logger.critical(
                ("{} does not exist. "
                 "Only finding fields").format(SELAVY_FOLDER))
                # FIND_FIELDS = True

        # self.FIND_FIELDS = FIND_FIELDS
        # self.CROSSMATCH_ONLY = CROSSMATCH_ONLY
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

        epoch_01 = pd.read_csv(FIELD_FILES["1"], , comment='#')
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
                    self.pilot_info = pd.read_csv(
                        FIELD_FILES[val], comment='#'
                    )
                    self.pilot_info["EPOCH"] = RELEASED_EPOCHS[val]
                else:
                    to_append = pd.read_csv(
                        FIELD_FILES[val], comment='#'
                    )
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
                    b.major.to(u.arcsec).value for b in common_beams
                ]
                self.field_info["C_BMIN (arcsec)"] = [
                    b.minor.to(u.arcsec).value for b in common_beams
                ]
                self.field_info["C_BPA (deg)"] = [
                    b.pa.to(u.deg).value for b in common_beams
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
