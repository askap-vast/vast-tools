from vasttools.survey import Fields, Image
from vasttools.survey import (
    RELEASED_EPOCHS, FIELD_FILES, ADA_BASE_DIR, NIMBUS_BASE_DIR,
    EPOCH_FIELDS, FIELD_CENTRES, OBSERVING_LOCATION, ALLOWED_PLANETS
)
from vasttools.source import Source
from vasttools.utils import (
    filter_selavy_components, simbad_search, match_planet_to_field
)

import sys
import numpy as np
import os
import datetime
import pandas as pd
import warnings
import io
import socket
import re
import signal
import numexpr

from multiprocessing import Pool, cpu_count
from multiprocessing_logging import install_mp_handler
from functools import partial
import dask.dataframe as dd

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
HOST_NIMBUS = 'nimbus.pawsey.org.au'

HOST_NCPU = cpu_count()
numexpr.set_num_threads(int(HOST_NCPU / 4))


class Query:
    '''
    This is a class representation of various information about a particular
    query including the catalogue of target sources, the Stokes parameter,
    crossmatch radius and output parameters.
    
    :param coords: List of coordinates to query, defaults to None
    :type coords: `astropy.coordinates.sky_coordinate.SkyCoord`, optional
    :param source_names: List of source names, defaults to []
    :type source_names: list, optional
    :param epochs: Comma-separated list of epochs to query. All available epochs can be queried by passsing "all". Defaults to "all"
    :type epochs: str, optional
    :param stokes: Stokes parameter to query, defaults to "I"
    :type stokes: str, optional
    :param crossmatch_radius: Crossmatch radius in arcsec, defaults to 5.0
    :type crossmatch_radius: float, optional
    :param max_sep: Maximum separation of source from beam centre in degrees, defaults to 1.0
    :type max_sep: float, optional
    :param use_tiles: Query tiles rather than combined mosaics, defaults to `False`
    :type use_tiles: bool, optional
    :param use_islands: Use selavy islands rather than components, defaults to `False`
    :type use_islands: bool, optional
    :param base_folder: Path to base folder if using default directory structure, defaults to None
    :type base_folder: str, optional
    :param matches_only: Only produce data products for sources with a selavy match, defaults to `False`
    :type matches_only: bool, optional
    :param no_rms: Estimate the background RMS around each source, defaults to `False`
    :type no_rms: bool, optional
    :param output_dir: Output directory to place all results in, defaults to "."
    :type output_dir: str, optional
    :param planets: List of planets to search for, defaults to []
    :type planets: list, optional
    :param ncpu: Number of CPUs to use, defaults to 2
    :type ncpu: float, optional
    '''

    def __init__(
        self, coords=None, source_names=[], epochs="all", stokes="I",
        crossmatch_radius=5.0, max_sep=1.0, use_tiles=False,
        use_islands=False, base_folder=None, matches_only=False,
        no_rms=False, output_dir=".", planets=[], ncpu=2
    ):
        '''Constructor method
        '''
        self.logger = logging.getLogger('vasttools.find_sources.Query')

        install_mp_handler(logger=self.logger)

        self.coords = coords
        self.source_names = source_names
        if ncpu > HOST_NCPU:
            raise ValueError(
                "Number of CPUs requested ({}) "
                "exceeds number available ({})".format(
                    ncpu,
                    HOST_NCPU
                )
            )
        self.ncpu = ncpu

        if self.coords is None and len(self.source_names) == 0 and len(self.planets) == 0:
            if self.logger is None:
                raise ValueError(
                    "No coordinates or source names have been provided!"
                    " Check inputs and try again!"
                )

        if self.coords is None:
            if len(source_names) != 0:
                pre_simbad = len(source_names)
                self.coords, self.source_names = simbad_search(
                    source_names, logger=self.logger
                )
                if self.coords is not None:
                    simbad_msg = "SIMBAD search found {}/{} source(s)".format(
                        len(self.source_names),
                        pre_simbad
                    )
                    self.logger.info(simbad_msg)
                    self.logger.info('Found:')
                    for i in self.source_names:
                        self.logger.info(i)
                    warnings.warn(simbad_msg)
                else:
                    self.logger.error(
                        "SIMBAD search failed!"
                    )
                    raise ValueError(
                        "SIMBAD search failed!"
                    )
        if len(planets) != 0:
            valid_planets = sum([i in ALLOWED_PLANETS for i in planets])

            if valid_planets != len(planets):
                self.logger.error(
                    "Invalid planet object provided!"
                )
                raise ValueError(
                    "Invalid planet object provided!"
                )
            else:
                self.planets = planets
        else:
            self.planets = None

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
            if HOST == HOST_ADA:
                self.base_folder = ADA_BASE_DIR
            elif HOST == HOST_NIMBUS:
                self.base_folder = NIMBUS_BASE_DIR
        else:
            self.base_folder = base_folder

        if not os.path.isdir(self.base_folder):
            self.logger.critical(
                "The base directory {} does not exist!".format(
                    self.base_folder
                )
            )
            raise ValueError("The base directory {} does not exist!".format(
                self.base_folder
            ))

        settings_ok = self._validate_settings()

        if not settings_ok:
            self.logger.critical("Problems found in query settings!")
            self.logger.critical("Please address and try again.")
            raise ValueError((
                "Problems found in query settings!"
                "\nPlease address and try again."
            ))

        if self.coords is not None:
            self.query_df = self.build_catalog()
        else:
            self.query_df = None

        self.fields_found = False

        self.cutout_data_got = False

    def _validate_settings(self):
        '''Use to check misc details'''

        if self.settings['tiles'] and self.settings['stokes'].lower() != "i":
            self.logger.critital("Only Stokes I are supported with tiles!")
            return False

        if self.settings['tiles'] and self.settings['islands']:
            self.logger.critital(
                "Only component catalogues are supported with tiles!"
            )
            return False

        if self.settings['tiles'] and not self.settings['no_rms']:
            self.logger.warning(
                "RMS measurements are not supported with tiles!"
            )
            self.logger.warning("Turning RMS measurements off.")
            self.settings['no_rms'] = True

        return True

    def get_all_cutout_data(self):
        '''Get cutout data'''
        
        # first get cutout data and selavy sources per image
        # group by image to do this

        meta = {
            'data': 'O',
            'wcs': 'O',
            'header': 'O',
            'selavy_overlay': 'O',
            'beam': 'O'
        }

        cutouts = (
            dd.from_pandas(self.sources_df, self.ncpu)
            .groupby('image')
            .apply(
                self._grouped_fetch_cutouts,
                meta=meta,
            ).compute(num_workers=self.ncpu, scheduler='processes')
        )

        cutouts.index = cutouts.index.droplevel()

        self.sources_df = self.sources_df.join(
            cutouts
        )

        for s in self.results:
            s_name = s.name
            s_cutout = self.sources_df[[
                'data',
                'wcs',
                'header',
                'selavy_overlay',
                'beam'
            ]][self.sources_df.name == s_name].reset_index(drop=True)

            s.cutout_df = s_cutout
            s._cutouts_got = True

        self.cutout_data_got = True

    def gen_all_source_products(
        self,
        fits=True,
        png=False,
        ann=False,
        reg=False,
        lightcurve=False,
        measurements=False,
        fits_outfile=None,
        png_selavy=True,
        png_percentile=99.9,
        png_zscale=False,
        png_contrast=0.2,
        png_no_islands=True,
        png_no_colorbar=False,
        png_crossmatch_overlay=False,
        png_hide_beam=False,
        ann_crossmatch_overlay=False,
        reg_crossmatch_overlay=False,
        lc_sigma_thresh=5,
        lc_figsize=(8, 4),
        lc_min_points=2,
        lc_min_detections=1,
        lc_mjd=False,
        lc_grid=False,
        lc_yaxis_start="auto",
        lc_peak_flux=True,
        measurements_simple=False

    ):
        """
        This function is not intended to be used interactively.
        Script only.
        
        :param fits: Create and save fits cutouts, defaults to `True`
        :type fits: bool, optional
        :param png: Create and save png postagestamps, defaults to `False`
        :type png: bool, optional
        :param ann: Create and save kvis annotation files for all components, defaults to `False`
        :type ann: bool, optional
        :param reg: Create and save DS9 annotation files for all components, defaults to `False`
        :type reg: bool, optional
        :param lightcurve: Create and save lightcurves for all sources, defaults to `False`
        :type lightcurve: bool, optional
        :param measurements: Create and save measurements for all sources, defaults to `False`
        :type measurements: bool, optional
        :param fits_outfile: File to save fits cutout to, defaults to None
        :type fits_outfile: str, optional
        :param png_selavy: Overlay selavy components onto png postagestamp, defaults to `True`
        :type png_selavy: bool, optional
        :param png_percentile: Percentile level for the png normalisation, defaults to 99.9
        :type png_percentile: float, optional
        :param png_zscale: Use z-scale normalisation rather than linear, defaults to `False`
        :type png_zscale: bool, optional
        :param png_contrast: Z-scale constrast, defaults to 0.2
        :type png_contrast: float, optional
        :param png_no_islands: Don't overlay selavy islands on png postagestamps, defaults to `True`
        :type png_no_islands: bool, optional
        :param png_no_colorbar: , defaults to `False`
        :type png_no_colorbar: bool, optional
        :param png_crossmatch_overlay: , defaults to `False`
        :type png_crossmatch_overlay: bool, optional
        :param png_hide_beam: , defaults to `False`
        :type png_hide_beam: bool, optional
        :param ann_crossmatch_overlay: , defaults to `False`
        :type ann_crossmatch_overlay: bool, optional
        :param reg_crossmatch_overlay: , defaults to `False`
        :type reg_crossmatch_overlay: bool, optional
        :param lc_sigma_thresh: , defaults to 5
        :type lc_sigma_thresh: float, optional
        :param lc_figsize: , defaults to (8, 4)
        :type lc_figsize: , optional
        :param lc_min_points: , defaults to 2
        :type lc_min_points: float, optional
        :param lc_min_detections: , defaults to 1
        :type lc_min_detections: float, optional
        :param lc_mjd: , defaults to `False`
        :type lc_mjd: bool, optional
        :param lc_grid: , defaults to `False`
        :type lc_grid: bool, optional
        :param lc_yaxis_start: , defaults to "auto"
        :type lc_yaxis_start: str, optional
        :param lc_peak_flux: , defaults to `True`
        :type lc_peak_flux: bool, optional
        :param measurements_simple: , defaults to `False`
        :type measurements_simple: bool, optional
        """

        if not self.cutout_data_got:
            self.get_all_cutout_data()

        original_sigint_handler = signal.signal(
            signal.SIGINT, signal.SIG_IGN
        )

        workers = Pool(processes=self.ncpu)

        signal.signal(signal.SIGINT, original_sigint_handler)

        if png:
            multi_png = partial(
                self._save_all_png_cutouts,
                selavy=png_selavy,
                percentile=png_percentile,
                zscale=png_zscale,
                contrast=png_contrast,
                no_islands=png_no_islands,
                no_colorbar=png_no_colorbar,
                crossmatch_overlay=png_crossmatch_overlay,
                hide_beam=png_hide_beam
            )

        if ann:
            multi_ann = partial(
                self._save_all_ann,
                crossmatch_overlay=ann_crossmatch_overlay
            )

        if reg:
            multi_reg = partial(
                self._save_all_reg,
                crossmatch_overlay=reg_crossmatch_overlay
            )

        if lightcurve:
            multi_lc = partial(
                self._save_all_lc,
                lc_sigma_thresh=lc_sigma_thresh,
                lc_figsize=lc_figsize,
                lc_min_points=lc_min_points,
                lc_min_detections=lc_min_detections,
                lc_mjd=lc_mjd,
                lc_grid=lc_grid,
                lc_yaxis_start=lc_yaxis_start,
                lc_peak_flux=lc_peak_flux,
                lc_save=True,
                lc_outfile=None,
            )

        if measurements:
            multi_measurements = partial(
                self._save_all_measurements,
                simple=measurements_simple,
                outfile=None,
            )

        try:
            if fits:
                self.logger.info("Saving FITS cutouts...")
                workers.map(self._save_all_fits_cutouts, self.results)
                self.logger.info("Done")
            if png:
                self.logger.info("Saving PNG cutouts...")
                workers.map(multi_png, self.results)
                self.logger.info("Done")
            if ann:
                self.logger.info("Saving .ann files...")
                workers.map(multi_ann, self.results)
                self.logger.info("Done")
            if reg:
                self.logger.info("Saving .reg files...")
                workers.map(multi_reg, self.results)
                self.logger.info("Done")
            if lightcurve:
                self.logger.info("Saving lightcurves...")
                workers.map(multi_lc, self.results)
                self.logger.info("Done")
            if measurements:
                self.logger.info("Saving measurements...")
                workers.map(multi_measurements, self.results)
                self.logger.info("Done")

        except KeyboardInterrupt:
            self.logger.error(
                "Caught KeyboardInterrupt, terminating workers."
            )
            workers.terminate()
            sys.exit()

        else:
            self.logger.debug("Normal termination")
            workers.close()
            workers.join()

        self.logger.info("-------------------------")
        self.logger.info("Summary:")
        self.logger.info("-------------------------")
        for s in self.results:
            self.logger.info(
                "Source %s - Matches: %i, Limits %i.",
                s.name,
                s.detections,
                s.limits
            )
        self.logger.info("-------------------------")

    def save_fields(self, out_dir=None):
        pass

    def _save_all_png_cutouts(
        self, s, selavy, percentile,
        zscale, contrast, no_islands, no_colorbar,
        crossmatch_overlay, hide_beam
    ):

        s.save_all_png_cutouts(
            selavy=selavy,
            percentile=percentile,
            zscale=zscale,
            contrast=contrast,
            islands=no_islands,
            no_colorbar=no_colorbar,
            crossmatch_overlay=crossmatch_overlay,
            hide_beam=hide_beam
        )

    def _save_all_fits_cutouts(self, s):

        s.save_all_fits_cutouts()

    def _save_all_ann(self, s, crossmatch_overlay=False):

        s.save_all_ann(crossmatch_overlay=crossmatch_overlay)

    def _save_all_reg(self, s, crossmatch_overlay=False):

        s.save_all_ann(crossmatch_overlay=crossmatch_overlay)

    def _save_all_measurements(self, s, simple=False, outfile=None):

        s.write_measurements(simple=simple, outfile=outfile)

    def _save_all_lc(
        self,
        s,
        lc_sigma_thresh=5,
        lc_figsize=(8, 4),
        lc_min_points=2,
        lc_min_detections=1,
        lc_mjd=False,
        lc_grid=False,
        lc_yaxis_start="auto",
        lc_peak_flux=True,
        lc_save=True,
        lc_outfile=None
    ):
        s.plot_lightcurve(
            sigma_thresh=lc_sigma_thresh,
            figsize=lc_figsize,
            min_points=lc_min_points,
            min_detections=lc_min_detections,
            mjd=lc_mjd,
            grid=lc_grid,
            yaxis_start=lc_yaxis_start,
            peak_flux=lc_peak_flux,
            save=lc_save,
            outfile=lc_outfile,
        )

    def _add_source_cutout_data(self, s):
        s_name = s.name
        s_cutout = self.sources_df[[
            'data',
            'wcs',
            'header',
            'selavy_overlay',
            'beam'
        ]][self.sources_df.name == s_name].reset_index(drop=True)

        s.cutout_df = s_cutout
        s._cutouts_got = True

        del s_cutout

        return s

    def _grouped_fetch_cutouts(self, group):

        image_file = group.iloc[0]['image']

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

        selavy_components = pd.read_fwf(row.selavy, skiprows=[1, ], usecols=[
            'island_id',
            'ra_deg_cont',
            'dec_deg_cont',
            'maj_axis',
            'min_axis',
            'pos_ang'
        ])

        selavy_coords = SkyCoord(
            selavy_components.ra_deg_cont.values,
            selavy_components.dec_deg_cont.values,
            unit=(u.deg, u.deg)
        )

        selavy_components = filter_selavy_components(
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

    def find_sources(self):
        if self.fields_found is False:
            self.find_fields()

        self.sources_df = self.fields_df.copy()

        self.sources_df[
            ['selavy', 'image', 'rms']
        ] = self.sources_df[['epoch', 'field', 'sbid']].apply(
            self._add_files,
            axis=1,
            result_type='expand'
        )

        meta = {
            '#': 'f',
            'island_id': 'U',
            'component_id': 'U',
            'component_name': 'U',
            'ra_hms_cont': 'U',
            'dec_dms_cont': 'U',
            'ra_deg_cont': 'f',
            'dec_deg_cont': 'f',
            'ra_err': 'f',
            'dec_err': 'f',
            'freq': 'f',
            'flux_peak': 'f',
            'flux_peak_err': 'f',
            'flux_int': 'f',
            'flux_int_err': 'f',
            'maj_axis': 'f',
            'min_axis': 'f',
            'pos_ang': 'f',
            'maj_axis_err': 'f',
            'min_axis_err': 'f',
            'pos_ang_err': 'f',
            'maj_axis_deconv': 'f',
            'min_axis_deconv': 'f',
            'pos_ang_deconv': 'f',
            'maj_axis_deconv_err': 'f',
            'min_axis_deconv_err': 'f',
            'pos_ang_deconv_err': 'f',
            'chi_squared_fit': 'f',
            'rms_fit_gauss': 'f',
            'spectral_index': 'f',
            'spectral_curvature': 'f',
            'spectral_index_err': 'f',
            'spectral_curvature_err': 'f',
            'rms_image': 'f',
            'has_siblings': 'f',
            'fit_is_estimate': 'f',
            'spectral_index_from_TT': 'f',
            'flag_c4': 'f',
            'comment': 'f',
            'detection': '?',
        }

        results = (
            dd.from_pandas(self.sources_df, self.ncpu)
            .groupby('selavy')
            .apply(
                self._get_components,
                meta=meta,
            ).compute(num_workers=self.ncpu, scheduler='processes')
        )

        results.index = results.index.droplevel()
        self.crossmatch_results = self.sources_df.merge(
            results, how='left', left_index=True, right_index=True
        )

        meta = {'name': 'O'}

        self.results = (
            dd.from_pandas(self.crossmatch_results, self.ncpu)
            .groupby('name')
            .apply(
                self._init_sources,
                meta=meta,
            ).compute(num_workers=self.ncpu, scheduler='processes')
        )

    def _init_sources(self, group):

        m = group.iloc[0]

        if m['planet']:
            source_coord = group.skycoord
            source_primary_field = group.primary_field
        else:
            source_coord = m.skycoord
            source_primary_field = m.primary_field
        source_name = m['name']
        source_epochs = group['epoch'].to_list()
        source_fields = group['field'].to_list()
        source_stokes = self.settings['stokes']
        source_base_folder = self.base_folder
        source_crossmatch_radius = self.settings['crossmatch_radius']
        source_outdir = self.settings['output_dir']
        if self.settings['tiles']:
            source_image_type = "TILES"
        else:
            source_image_type = "COMBINED"
        source_islands = self.settings['islands']

        source_df = group.drop(
            columns=[
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

    def _get_components(self, group):
        selavy_file = group.iloc[0]['selavy']
        master = pd.DataFrame()

        selavy_df = pd.read_fwf(
            selavy_file, skiprows=[1, ]
        )

        selavy_coords = SkyCoord(
            selavy_df.ra_deg_cont,
            selavy_df.dec_deg_cont,
            unit=(u.deg, u.deg)
        )
        group_coords = SkyCoord(
            group.ra,
            group.dec,
            unit=(u.deg, u.deg)
        )

        idx, d2d, _ = group_coords.match_to_catalog_sky(selavy_coords)
        mask = d2d < self.settings['crossmatch_radius']
        idx_matches = idx[mask]

        copy = selavy_df.iloc[idx_matches].reset_index(drop=True)
        copy["detection"] = True
        copy.index = group[mask].index.values

        master = master.append(copy, sort=False)

        missing = group_coords[~mask]
        if missing.shape[0] > 0:
            if not self.settings['no_rms']:
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
            else:
                rms_df = pd.DataFrame(
                    [-99 for i in range(missing.shape[0])],
                    columns=['rms_image']
                )

            rms_df['detection'] = False

            rms_df.index = group[~mask].index.values

            master = master.append(rms_df, sort=False)

        return master

    def _add_files(self, row):

        epoch_string = "EPOCH{}".format(
            RELEASED_EPOCHS[row.epoch]
        )

        if self.settings['islands']:
            cat_type = 'islands'
        else:
            cat_type = 'components'

        if self.settings['tiles']:

            dir_name = "TILES"

            selavy_file_fmt = (
                "selavy-image.i.SB{}.cont.{}."
                "linmos.taylor.0.restored.components.txt".format(
                    row.sbid, row.field
                )
            )

            image_file_fmt = (
                "image.i.SB{}.cont.{}"
                ".linmos.taylor.0.restored.fits".format(
                    row.sbid, row.field
                )
            )

        else:

            dir_name = "COMBINED"

            selavy_file_fmt = "{}.EPOCH{}.{}.selavy.{}.txt".format(
                row.field,
                RELEASED_EPOCHS[row.epoch],
                self.settings['stokes'],
                cat_type
            )

            image_file_fmt = "{}.EPOCH{}.{}.fits".format(
                row.field,
                RELEASED_EPOCHS[row.epoch],
                self.settings['stokes'],
            )

            rms_file_fmt = "{}.EPOCH{}.{}_rms.fits".format(
                row.field,
                RELEASED_EPOCHS[row.epoch],
                self.settings['stokes'],
            )

        selavy_file = os.path.join(
            self.base_folder,
            epoch_string,
            dir_name,
            "STOKES{}_SELAVY".format(self.settings['stokes']),
            selavy_file_fmt
        )

        image_file = os.path.join(
            self.base_folder,
            epoch_string,
            dir_name,
            "STOKES{}_IMAGES".format(self.settings['stokes']),
            image_file_fmt
        )

        if self.settings['tiles']:
            rms_file = "N/A"
        else:
            rms_file = os.path.join(
                self.base_folder,
                epoch_string,
                dir_name,
                "STOKES{}_RMSMAPS".format(self.settings['stokes']),
                rms_file_fmt
            )

        return selavy_file, image_file, rms_file

    def write_find_fields(self, outname=None):
        if self.fields_found is False:
            self.find_fields()

        if outname is None:
            name = 'find_fields_result.csv'
        else:
            name = outname+'.pkl'

        outdir = self.settings['output_dir']
        if outdir is not None:
            name = os.path.join(outdir, name)

        self.fields_df[[
            'name',
            'ra',
            'dec',
            'field',
            'epoch',
            'sbid',
            'dateobs',
            'primary_field'
        ]].to_csv(name, index=False)

        self.logger.info('Find fields output saved as {}.'.format(
            name
        ))

    def find_fields(self):
        fields = Fields('1')
        field_centres_sc = SkyCoord(
            FIELD_CENTRES["centre-ra"],
            FIELD_CENTRES["centre-dec"],
            unit=(u.deg, u.deg)
        )

        field_centre_names = FIELD_CENTRES.field

        if self.query_df is not None:
            self.fields_df = self.query_df.copy()

            meta = {
                0: 'O',
                1: 'U',
                2: 'O',
                3: 'O',
                4: 'O',
                5: 'O',
            }

            self.fields_df[[
                'fields',
                'primary_field',
                'epochs',
                'field_per_epoch',
                'sbids',
                'dates'
            ]] = (
                dd.from_pandas(self.fields_df, self.ncpu)
                .apply(
                    self._field_matching,
                    args=(
                        fields.direction,
                        fields.fields.FIELD_NAME,
                        field_centres_sc,
                        field_centre_names
                    ),
                    meta=meta,
                    axis=1,
                    result_type='expand'
                ).compute(num_workers=self.ncpu, scheduler='processes')
            )

            self.fields_df = self.fields_df.dropna()

            self.fields_df = self.fields_df.explode(
                'field_per_epoch'
            ).reset_index(drop=True)
            self.fields_df[
                ['epoch', 'field', 'sbid', 'dateobs']
            ] = self.fields_df.field_per_epoch.apply(pd.Series)

            to_drop = [
                'field_per_epoch',
                'epochs',
                'sbids',
                'dates'
            ]

            self.fields_df = self.fields_df.drop(
                labels=to_drop, axis=1
            ).sort_values(
                by=['name', 'dateobs']
            ).reset_index(drop=True)

            self.fields_df['planet'] = False
        else:
            self.fields_df = None

        if self.planets is not None:
            planet_fields = self.search_planets()

            if self.fields_df is None:
                self.fields_df = planet_fields
            else:
                self.fields_df = self.fields_df.append(
                    planet_fields
                ).reset_index(drop=True)

        if self.query_df is None:
            prev_num = 0
        else:
            prev_num = self.query_df.shape[0]

        if self.planets is not None:
            prev_num += len(self.planets)

        self.logger.info(
            "%i/%i sources in VAST Pilot footprint.",
            self.fields_df.name.unique().shape[0],
            prev_num
        )

        self.fields_df['dateobs'] = pd.to_datetime(
            self.fields_df['dateobs']
        )

        self.fields_found = True

    def _field_matching(
        self,
        row,
        fields_coords,
        fields_names,
        field_centres,
        field_centre_names
    ):
        seps = row.skycoord.separation(fields_coords)
        accept = seps.deg < self.settings['max_sep']
        fields = np.unique(fields_names[accept])

        if fields.shape[0] == 0:
            self.logger.warning(
                "Source '%s' not in VAST Pilot footprint.",
                row['name']
            )
            warnings.warn(
                "Source '{}' not in selected footprint. "
                "Dropping source.".format(
                    row['name']
                )
            )
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        centre_seps = row.skycoord.separation(field_centres)
        primary_field = field_centre_names[np.argmin(centre_seps.deg)]
        epochs = []
        field_per_epochs = []
        sbids = []
        dateobs = []

        for i in self.settings['epochs']:
            epoch_fields = EPOCH_FIELDS[i].keys()
            available_fields = [f for f in fields if f in epoch_fields]

            if len(available_fields) == 0:
                continue

            elif primary_field in available_fields:
                field = primary_field

            elif len(available_fields) == 1:
                field = available_fields[0]

            else:
                field_indexes = [
                    field_centre_names[
                        field_centre_names == f
                    ].index[0] for f in available_fields
                ]
                min_field_index = np.argmin(
                    centre_seps[field_indexes].deg
                )

                field = available_fields[min_field_index]

            epochs.append(i)
            sbid = EPOCH_FIELDS[i][field]["SBID"]
            date = EPOCH_FIELDS[i][field]["DATEOBS"]
            sbids.append(sbid)
            dateobs.append(date)
            field_per_epochs.append([i, field, sbid, date])

        return fields, primary_field, epochs, field_per_epochs, sbids, dateobs

    def _get_planets_epoch_df_template(self):
        epochs = self.settings['epochs']

        planet_epoch_fields = pd.DataFrame.from_dict(
            EPOCH_FIELDS[epochs[0]], orient='index'
        ).reset_index().rename(columns={'index': 'FIELD_NAME'})

        planet_epoch_fields['epoch'] = epochs[0]

        for e in epochs[1:]:
            temp = pd.DataFrame.from_dict(
                EPOCH_FIELDS[e], orient='index'
            ).reset_index().rename(columns={'index': 'FIELD_NAME'})
            temp['epoch'] = e
            planet_epoch_fields = planet_epoch_fields.append(
                temp
            )
            del temp

        planet_epoch_fields = planet_epoch_fields.merge(
            FIELD_CENTRES, left_on='FIELD_NAME',
            right_on='field', how='left'
        ).drop('field', axis=1)

        return planet_epoch_fields

    def search_planets(self):

        template = self._get_planets_epoch_df_template()

        template['planet'] = [self.planets for i in range(template.shape[0])]

        template = template.explode('planet')
        template['planet'] = template['planet'].str.capitalize()

        meta = {
            'FIELD_NAME': 'U',
            'SBID': 'i',
            'DATEOBS': 'datetime64[ns]',
            'epoch': 'U',
            'centre-ra': 'f',
            'centre-dec': 'f',
            'planet': 'U',
            'ra': 'f',
            'dec': 'f',
            'sep': 'f'
        }

        results = (
            dd.from_pandas(template, self.ncpu)
            .groupby('planet')
            .apply(
                match_planet_to_field,
                meta=meta,
            ).compute(num_workers=self.ncpu, scheduler='processes')
        )

        results = results.reset_index(drop=True).drop(
            ['centre-ra', 'centre-dec', 'sep'], axis=1
        ).rename(columns={
            'planet': 'name',
            'FIELD_NAME': 'field',
            'DATEOBS': 'dateobs',
            'SBID': 'sbid',
        }).sort_values(by=['name', 'dateobs'])

        results['stokes'] = self.settings['stokes'].upper()
        results['primary_field'] = results['field']
        results['skycoord'] = SkyCoord(
            results['ra'], results['dec'], unit=(u.deg, u.deg)
        )
        results['fields'] = [[i] for i in results['field']]
        results['planet'] = True

        return results

    def build_catalog(self):
        cols = ['ra', 'dec', 'name', 'skycoord', 'stokes']
        if self.coords.shape == ():
            catalog = pd.DataFrame(
                [[
                    self.coords.ra.deg,
                    self.coords.dec.deg,
                    self.source_names[0],
                    self.coords,
                    self.settings['stokes']
                ]], columns=cols
            )
        else:
            catalog = pd.DataFrame(
                self.source_names,
                columns=['name']
            )
            catalog['ra'] = self.coords.ra.deg
            catalog['dec'] = self.coords.dec.deg
            catalog['skycoord'] = self.coords
            catalog['stokes'] = self.settings['stokes']

        return catalog

    def get_epochs(self, req_epochs):
        '''
        Parse the list of epochs to query.

        :returns: Epochs to query, as a list of string
        :rtype: list
        '''

        available_epochs = sorted(RELEASED_EPOCHS, key=RELEASED_EPOCHS.get)
        self.logger.debug("Available epochs: " + str(available_epochs))

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

        if not os.path.isdir(IMAGE_FOLDER):
            # if not CROSSMATCH_ONLY:
            self.logger.warning(
                "{} does not exist. "
                "Can only do crossmatching.".format(IMAGE_FOLDER)
            )

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

        if not os.path.isdir(RMS_FOLDER):
            # if not CROSSMATCH_ONLY:
            self.logger.critical((
                "{} does not exist. "
                "Switching to crossmatch only."
            ).format(RMS_FOLDER))

        image_dir = "COMBINED"
        selavy_dir = "STOKES{}_SELAVY".format(self.stokes_param)

        SELAVY_FOLDER = os.path.join(
            BASE_FOLDER,
            survey_folder,
            image_dir,
            selavy_dir
        )

        if not os.path.isdir(SELAVY_FOLDER):
            # if not FIND_FIELDS and not CROSSMATCH_ONLY:
            self.logger.critical((
                "{} does not exist. "
                "Only finding fields"
            ).format(SELAVY_FOLDER))

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

        epoch_01 = pd.read_csv(FIELD_FILES["1"], comment='#')
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
            psf=False,
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

        if psf or largest_psf or common_psf or all_psf:
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
            if psf:
                beams_zero = []
                for i in sorted(epoch_beams):
                    beams_zero.append(epoch_beams[i][0])

                self.field_info["BMAJ (arcsec)"] = [
                    b.major.value for b in beams_zero
                ]
                self.field_info["BMIN (arcsec)"] = [
                    b.minor.value for b in beams_zero
                ]
                self.field_info["BPA (deg)"] = [
                    b.pa.value for b in beams_zero
                ]
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
