"""Class to perform queries on the VAST observational data.

Attributes:
    HOST_NCPU (int): The number of CPU found on the host using 'cpu_count()'.

"""

import sys
import numpy as np
import os
import datetime
import pandas as pd
import warnings
import io
import re
import signal
import numexpr
import gc
import time
import dask.dataframe as dd
import logging
import logging.handlers
import logging.config
import matplotlib.pyplot as plt
import matplotlib.axes as maxes

from astropy.coordinates import Angle
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.nddata.utils import Cutout2D
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from astropy.utils.exceptions import AstropyWarning, AstropyDeprecationWarning
from astropy.visualization import ZScaleInterval, ImageNormalize
from astropy.visualization import PercentileInterval
from astropy.visualization import AsymmetricPercentileInterval
from astropy.visualization import LinearStretch

from functools import partial

from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection
from multiprocessing import Pool, cpu_count
from multiprocessing_logging import install_mp_handler

from mpl_toolkits.axes_grid1 import make_axes_locatable

from radio_beam import Beams, Beam

from tabulate import tabulate

from typing import Optional, List, Tuple, Dict

from vasttools import RELEASED_EPOCHS, ALLOWED_PLANETS
from vasttools.survey import Fields, Image
from vasttools.survey import (
    load_fields_file, load_field_centres, get_fields_per_epoch_info
)
from vasttools.source import Source
from vasttools.utils import (
    filter_selavy_components, simbad_search, match_planet_to_field,
    check_racs_exists, epoch12_user_warning
)
from vasttools.moc import VASTMOCS
from forced_phot import ForcedPhot

warnings.filterwarnings('ignore', category=AstropyWarning, append=True)
warnings.filterwarnings('ignore',
                        category=AstropyDeprecationWarning, append=True)

HOST_NCPU = cpu_count()
numexpr.set_num_threads(int(HOST_NCPU / 4))

class QueryInitError(Exception):
    """
    A defined error for a problem encountered in the initialisation.
    """
    pass


class Query:
    """
    This is a class representation of various information about a particular
    query including the catalogue of target sources, the Stokes parameter,
    crossmatch radius and output parameters.

    Attributes:
        coords (astropy.coordinates.sky_coordinate.SkyCoord):
            The sky coordinates to be queried.
        source_names (List[str]):
            The names of the sources (coordinates) being queried.
        ncpu (int): The number of cpus available.
        planets (bool): Set to 'True' when planets are to be queried.
        settings (Dict):
            Dictionary that contains the various settings of the query.
            TODO: This dictionary typing needs better defining.
        base_folder (str): The base folder of the VAST data.
        fields_found (bool): Set to 'True' once 'find_fields' has been
            run on the query.
        racs (bool): Set to 'True' if RACS (Epoch 00) is included in
            the query.
        query_df (pandas.core.frame.DataFrame):
            The dataframe that is constructed to perform the query.
        sources_df (pandas.core.frame.DataFrame):
            The dataframe that contains the found sources when 'find_sources'
            is run.
        results (pandas.core.frame.Series):
            Series that contains each result in the form of a
            vasttools.source.Source object, with the source name
            as the index.
    """

    def __init__(
        self,
        coords: Optional[SkyCoord] = None,
        source_names: Optional[List[str]] = None,
        epochs: str = "all",
        stokes: str = "I",
        crossmatch_radius: float = 5.0,
        max_sep: float = 1.0,
        use_tiles: bool = False,
        use_islands: bool = False,
        base_folder: Optional[str] = None,
        matches_only: bool = False,
        no_rms: bool = False,
        search_around_coordinates: bool = False,
        output_dir: str = ".",
        planets: Optional[List[str]] = None,
        ncpu: int = 2,
        sort_output: bool = False,
        forced_fits: bool = False,
        forced_cluster_threshold: float = 1.5,
        forced_allow_nan: bool = False
    ) -> None:
        """
        Constructor method.

        Args:
            coords: List of coordinates to query, defaults to None.
            source_names: List of source names, defaults to None.
            epochs: Comma-separated list of epochs to query.
                All available epochs can be queried by passing "all".
                Defaults to "all".
            stokes: Stokes parameter to query.
            crossmatch_radius: Crossmatch radius in arcsec, defaults to 5.0.
            max_sep: Maximum separation of source from beam centre
                in degrees, defaults to 1.0.
            use_tiles: Query tiles rather than combined mosaics,
                defaults to `False`.
            use_islands: Use selavy islands rather than components,
                defaults to `False`.
            base_folder: Path to base folder if using default directory
                structure, defaults to 'None'.
            matches_only: Only produce data products for sources with a
                selavy match, defaults to `False`.
            no_rms: When set to `True` the estimate of the background RMS
                around each source, will not be performed,
                defaults to `False`.
            search_around_coordinates: When set to `True`, all matches to a
                searched coordinate are returned, instead of only the closest
                match.
            output_dir: Output directory to place all results in,
                defaults to ".".
            planets: List of planets to search for, defaults to None.
            ncpu: Number of CPUs to use, defaults to 2.
            sort_output: Sorts the output into individual source
                directories, defaults to `False`.
            forced_fits: Turns on the option to perform forced fits
                on the locations queried, defaults to `False`.
            forced_cluster_threshold: The cluster_threshold value passed to
                the forced photometry. Beam width distance limit for which a
                cluster is formed for extraction, defaults to 3.0.
            forced_allow_nan: `allow_nan` value passed to the
                forced photometry. If False then any cluster containing a
                NaN is ignored. Defaults to False.

        Returns:
            None

        Raises:
            ValueError: If the number of CPUs requested exceeds the total
                available.
            QueryInitError: No coordinates or source names have been provided.
            QueryInitError: Forced fits and search around coordinates options
                have both been selected.
            QueryInitError: Number of source names provided does not match the
                number of coordinates.
            ValueError: Planet provided is not a valid planet.
            QueryInitError: Base folder could not be determined.
            QueryInitError: SIMBAD search failed.
            QueryInitError: Base folder cannot be found.
            QueryInitError: Base folder cannot be found.
            QueryInitError: Problems found in query settings.
        """
        self.logger = logging.getLogger('vasttools.find_sources.Query')

        install_mp_handler(logger=self.logger)

        if source_names is None:
            source_names = []

        self.source_names = np.array(source_names)
        self.simbad_names = None

        if coords is None:
            self.coords = coords
        elif coords.isscalar:
            self.coords = SkyCoord([coords.ra], [coords.dec])
        else:
            self.coords = coords

        if self.coords is None:
            len_coords = 0
        else:
            len_coords = self.coords.shape[0]

        if ncpu > HOST_NCPU:
            raise ValueError(
                "Number of CPUs requested ({}) "
                "exceeds number available ({})".format(
                    ncpu,
                    HOST_NCPU
                )
            )
        self.ncpu = ncpu

        if coords is None and len(source_names) == 0 and planets is None:
            raise QueryInitError(
                "No coordinates or source names have been provided!"
                " Check inputs and try again!"
            )

        if forced_fits and search_around_coordinates:
            raise QueryInitError(
                "Forced fits and search around coordinates mode cannot be"
                " used together! Check inputs and try again."
            )

        if (
            self.coords is not None and
            len(self.source_names) > 0 and
            len(self.source_names) != len_coords
        ):
            raise QueryInitError(
                "The number of entered source names ({}) does not match the"
                " number of coordinates ({})!".format(
                    len(self.source_names),
                    len_coords
                )
            )

        if self.coords is not None and len(self.source_names) == 0:
            source_names = [
                'source_' + i.to_string(
                    'hmsdms', sep="", precision=1
                ).replace(" ", "") for i in self.coords
            ]
            self.source_names = np.array(source_names)

        if self.coords is None:
            if len(source_names) != 0:
                num_sources = len(source_names)
                self.coords, self.simbad_names = simbad_search(
                    source_names, logger=self.logger
                )
                num_simbad = len(list(filter(None, self.simbad_names)))
                if self.coords is not None:
                    simbad_msg = "SIMBAD search found {}/{} source(s):".format(
                        num_simbad,
                        num_sources
                    )
                    self.logger.info(simbad_msg)
                    names = zip(self.simbad_names, self.source_names)
                    for simbad_name, query_name in names:
                        if simbad_name:
                            self.logger.info(
                                '{}: {}'.format(query_name, simbad_name)
                            )
                        else:
                            self.logger.info(
                                '{}: No match.'.format(query_name)
                            )
                    if self.logger is None:
                        warnings.warn(simbad_msg)
                else:
                    self.logger.error(
                        "SIMBAD search failed!"
                    )
                    raise QueryInitError(
                        "SIMBAD search failed!"
                    )

        if planets is not None:
            planets = [i.lower() for i in planets]
            valid_planets = sum([i in ALLOWED_PLANETS for i in planets])

            if valid_planets != len(planets):
                self.logger.error(
                    "Invalid planet object provided!"
                )
                raise ValueError(
                    "Invalid planet object provided!"
                )

        self.planets = planets

        self.settings = {}

        if base_folder is None:
            the_base_folder = os.getenv('VAST_DATA_DIR')
            if the_base_folder is None:
                raise QueryInitError(
                    "The base folder directory could not be determined!"
                    " Either the system environment 'VAST_DATA_DIR' must be"
                    " defined or the 'base_folder' argument defined when"
                    " initialising the query."
                )
        else:
            the_base_folder = os.path.abspath(str(base_folder))

        if not os.path.isdir(the_base_folder):
            raise QueryInitError(
                "Base folder {} not found!".format(
                    the_base_folder
                )
            )

        self.base_folder = the_base_folder

        self.settings['epochs'] = self._get_epochs(epochs)
        self.settings['stokes'] = self._get_stokes(stokes)

        self.settings['crossmatch_radius'] = Angle(
            crossmatch_radius, unit=u.arcsec
        )
        self.settings['max_sep'] = max_sep

        self.settings['islands'] = use_islands
        self.settings['tiles'] = use_tiles
        self.settings['no_rms'] = no_rms
        self.settings['matches_only'] = matches_only
        self.settings['search_around'] = search_around_coordinates
        self.settings['sort_output'] = sort_output
        self.settings['forced_fits'] = forced_fits
        self.settings[
            'forced_cluster_threshold'
        ] = forced_cluster_threshold
        self.settings['forced_allow_nan'] = forced_allow_nan

        self.settings['output_dir'] = output_dir

        # Going to need this so load it now
        self._epoch_fields = get_fields_per_epoch_info()

        if not os.path.isdir(self.base_folder):
            self.logger.critical(
                "The base directory {} does not exist!".format(
                    self.base_folder
                )
            )
            raise QueryInitError(
                "The base directory {} does not exist!".format(
                    self.base_folder
                )
            )

        settings_ok = self._validate_settings()

        if not settings_ok:
            self.logger.critical("Problems found in query settings!")
            self.logger.critical("Please address and try again.")
            raise QueryInitError((
                "Problems found in query settings!"
                "\nPlease address and try again."
            ))

        if self.coords is not None:
            self.query_df = self._build_catalog()
            if self.query_df.empty:
                raise QueryInitError(
                    'No sources remaining. None of the entered coordinates'
                    ' are found in the VAST Pilot survey footprint!'
                )
        else:
            self.query_df = None

        self.fields_found = False

        # TODO: Remove warning in future release.
        epoch12_user_warning()

    def _validate_settings(self) -> bool:
        """
        Used to check that the settings are valid.

        Returns:
            `True` if settings are acceptable, `False` otherwise.
        """

        if self.settings['tiles'] and self.settings['stokes'].lower() != "i":
            self.logger.critical("Only Stokes I are supported with tiles!")
            return False

        if self.settings['tiles'] and self.settings['islands']:
            self.logger.critical(
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

    def _get_all_cutout_data(self, imsize: Angle) -> pd.DataFrame:
        """
        Get cutout data and selavy components for all sources.

        Args:
            imsize: Size of the requested cutout.

        Returns:
            Dataframe containing the cutout data of all measurements in
            the query. Cutout data specifically is the image data, header,
            wcs, and selavy sources present in the cutout.

        Raises:
            Exception: Function cannot be run when 'search_around_coordinates'
                mode has been selected.
        """
        # first get cutout data and selavy sources per image
        # group by image to do this

        if self.settings['search_around']:
            raise Exception(
                'Getting cutout data cannot be run when'
                ' search around coordinates mode has been'
                ' used.'
            )

        meta = {
            'data': 'O',
            'wcs': 'O',
            'header': 'O',
            'selavy_overlay': 'O',
            'beam': 'O',
            'name': 'U',
            'dateobs': 'datetime64[ns]'
        }

        cutouts = (
            dd.from_pandas(self.sources_df, self.ncpu)
            .groupby('image')
            .apply(
                self._grouped_fetch_cutouts,
                imsize=imsize,
                meta=meta,
            ).compute(num_workers=self.ncpu, scheduler='processes')
        )

        if not cutouts.empty:
            if isinstance(cutouts.index, pd.MultiIndex):
                cutouts.index = cutouts.index.droplevel()

        return cutouts

    def _gen_all_source_products(
        self,
        fits: bool = True,
        png: bool = False,
        ann: bool = False,
        reg: bool = False,
        lightcurve: bool = False,
        measurements: bool = False,
        fits_outfile: Optional[str] = None,
        png_selavy: bool = True,
        png_percentile: float = 99.9,
        png_zscale: bool = False,
        png_contrast: float = 0.2,
        png_no_islands: bool = True,
        png_no_colorbar: bool = False,
        png_crossmatch_overlay: bool = False,
        png_hide_beam: bool = False,
        png_disable_autoscaling: bool = False,
        ann_crossmatch_overlay: bool = False,
        reg_crossmatch_overlay: bool = False,
        lc_sigma_thresh: int = 5,
        lc_figsize: Tuple[int, int] = (8, 4),
        lc_min_points: int = 2,
        lc_min_detections: int = 1,
        lc_mjd: bool = False,
        lc_start_date: Optional[pd.Timestamp] = None,
        lc_grid: bool = False,
        lc_yaxis_start: str = "auto",
        lc_peak_flux: bool = True,
        lc_use_forced_for_limits: bool = False,
        lc_use_forced_for_all: bool = False,
        lc_hide_legend: bool = False,
        measurements_simple: bool = False,
        imsize: Angle = Angle(5. * u.arcmin),
        plot_dpi: int = 150
    ) -> None:
        """
        Generate products for all sources.
        This function is not intended to be used interactively - script only.

        Args:
            fits: Create and save fits cutouts, defaults to `True`.
            png: Create and save png postagestamps, defaults to `False`.
            ann: Create and save kvis annotation files for all components,
                defaults to `False`
            reg: Create and save DS9 annotation files for all components,
                defaults to `False`
            lightcurve: Create and save lightcurves for all sources,
                defaults to `False`
            measurements: Create and save measurements for all sources,
                defaults to `False`
            fits_outfile: File to save fits cutout to, defaults to None.
            png_selavy: Overlay selavy components onto png postagestamp,
                defaults to `True`
            png_percentile: Percentile level for the png normalisation,
                defaults to 99.9.
            png_zscale: Use z-scale normalisation rather than linear,
                defaults to `False`.
            png_contrast: Z-scale constrast, defaults to 0.2.
            png_no_islands: Don't overlay selavy islands on png
                postagestamps, defaults to `True`.
            png_no_colorbar: Don't include colourbar on png output,
                defaults to `False`
            png_crossmatch_overlay: Overlay the crossmatch radius on png
                postagestamps, defaults to `False`.
            png_hide_beam: Do not show the beam shape on png postagestamps,
                defaults to `False`.
            ann_crossmatch_overlay: Include crossmatch radius in ann,
                defaults to `False`.
            reg_crossmatch_overlay: Include crossmatch radius in reg,
                defaults to `False`.
            lc_sigma_thresh: Detection threshold (in sigma) for
                lightcurves, defaults to 5.
            lc_figsize: Size of lightcurve figure, defaults to (8, 4).
            lc_min_points: Minimum number of source observations required
                for a lightcurve to be generated, defaults to 2.
            lc_min_detections: Minimum number of source detections required
                for a lightcurve to be generated, defaults to 1.
            lc_mjd: Use MJD for lightcurve x-axis, defaults to `False`.
            lc_start_date: Plot lightcurve in days from start date,
                defaults to None.
            lc_grid: Include grid on lightcurve plot, defaults to `False`.
            lc_yaxis_start: Start the lightcurve y-axis at 0 ('0') or use
                the matpotlib default ('auto'). Defaults to 'auto'.
            lc_peak_flux: Generate lightcurve using peak flux density
                rather than integrated flux density, defaults to `True`.
            measurements_simple: Use simple schema for measurement output,
                defaults to `False`.
            imsize: Size of the requested cutout.
            plot_dpi: Specify the DPI of saved figures, defaults to 150.

        Returns:
            None

        Raises:
            Exception: Function cannot be run when 'search_around_coordinates'
                option has been selected.
        """
        if self.settings['search_around']:
            raise Exception(
                'Getting source products cannot be run when'
                ' search around coordinates mode has been'
                ' used.'
            )

        if sum([fits, png, ann, reg]) > 0:
            self.logger.info(
                "Fetching cutout data for sources..."
            )
            cutouts_df = self._get_all_cutout_data(imsize)
            self.logger.info('Done.')
            if cutouts_df.empty:
                fits = False
                png = False
                self.logger.warning(
                    'Cutout data could not be fetched, turning off fits and'
                    ' png production.'
                )
                to_process = [(s, None) for s in self.results.values]
                cutouts_df = None
            else:
                to_process = [(s, cutouts_df.loc[
                    cutouts_df['name'] == s.name
                ].sort_values(
                    by='dateobs'
                ).reset_index()) for s in self.results.values]

                del cutouts_df
                gc.collect()
        else:
            to_process = [(s, None) for s in self.results.values]
            cutouts_df = None

        self.logger.info(
            'Saving source products, please be paitent for large queries...'
        )

        produce_source_products_multi = partial(
            self._produce_source_products,
            fits=fits,
            png=png,
            ann=ann,
            reg=reg,
            lightcurve=lightcurve,
            measurements=measurements,
            png_selavy=png_selavy,
            png_percentile=png_percentile,
            png_zscale=png_zscale,
            png_contrast=png_contrast,
            png_no_islands=png_no_islands,
            png_no_colorbar=png_no_colorbar,
            png_crossmatch_overlay=png_crossmatch_overlay,
            png_hide_beam=png_hide_beam,
            png_disable_autoscaling=png_disable_autoscaling,
            ann_crossmatch_overlay=ann_crossmatch_overlay,
            reg_crossmatch_overlay=reg_crossmatch_overlay,
            lc_sigma_thresh=lc_sigma_thresh,
            lc_figsize=lc_figsize,
            lc_min_points=lc_min_points,
            lc_min_detections=lc_min_detections,
            lc_mjd=lc_mjd,
            lc_start_date=lc_start_date,
            lc_grid=lc_grid,
            lc_yaxis_start=lc_yaxis_start,
            lc_peak_flux=lc_peak_flux,
            lc_use_forced_for_limits=lc_use_forced_for_limits,
            lc_use_forced_for_all=lc_use_forced_for_all,
            lc_hide_legend=lc_hide_legend,
            measurements_simple=measurements_simple,
            calc_script_norms=~(png_disable_autoscaling),
            plot_dpi=plot_dpi
        )

        original_sigint_handler = signal.signal(
            signal.SIGINT, signal.SIG_IGN
        )
        signal.signal(signal.SIGINT, original_sigint_handler)
        workers = Pool(processes=self.ncpu, maxtasksperchild=100)

        try:
            workers.map(
                produce_source_products_multi,
                to_process,
            )
        except KeyboardInterrupt:
            self.logger.error(
                "Caught KeyboardInterrupt, terminating workers."
            )
            workers.terminate()
            sys.exit()
        except Exception as e:
            self.logger.error(
                "Encountered error!."
            )
            self.logger.error(
                e
            )
            workers.terminate()
            sys.exit()
        finally:
            self.logger.debug("Closing workers.")
            # Using terminate below as close was prone to hanging
            # when join is called. I believe the hang comes from
            # a child processes not getting returned properly
            # because of the large number of file I/O.
            workers.terminate()
            workers.join()

    def _produce_source_products(
        self,
        i: Tuple[Source, pd.DataFrame],
        fits: bool = True,
        png: bool = False,
        ann: bool = False,
        reg: bool = False,
        lightcurve: bool = False,
        measurements: bool = False,
        png_selavy: bool = True,
        png_percentile: float = 99.9,
        png_zscale: bool = False,
        png_contrast: float = 0.2,
        png_no_islands: bool = True,
        png_no_colorbar: bool = False,
        png_crossmatch_overlay: bool = False,
        png_hide_beam: bool = False,
        png_disable_autoscaling: bool = False,
        ann_crossmatch_overlay: bool = False,
        reg_crossmatch_overlay: bool = False,
        lc_sigma_thresh: int = 5,
        lc_figsize: Tuple[int, int] = (8, 4),
        lc_min_points: int = 2,
        lc_min_detections: int = 1,
        lc_mjd: bool = False,
        lc_start_date: Optional[pd.Timestamp] = None,
        lc_grid: bool = False,
        lc_yaxis_start: str = "auto",
        lc_peak_flux: bool = True,
        lc_use_forced_for_limits: bool = False,
        lc_use_forced_for_all: bool = False,
        lc_hide_legend: bool = False,
        measurements_simple: bool = False,
        calc_script_norms: bool = False,
        plot_dpi: int = 150
    ) -> None:
        """
        Produce source products for one source.

        Args:
            i: Tuple containing source and cutout data.
            fits: Create and save fits cutouts, defaults to `True`.
            png: Create and save png postagestamps, defaults to `False`.
            ann: Create and save kvis annotation files for all components,
                defaults to `False`.
            reg: Create and save DS9 annotation files for all components,
                defaults to `False`.
            lightcurve: Create and save lightcurves for all sources,
                defaults to `False`.
            measurements: Create and save measurements for all sources,
                defaults to `False`.
            png_selavy: Overlay selavy components onto png postagestamp,
                defaults to `True`.
            png_percentile: Percentile level for the png normalisation,
                defaults to 99.9.
            png_zscale: Use z-scale normalisation rather than linear,
                defaults to `False`.
            png_contrast: Z-scale constrast, defaults to 0.2.
            png_no_islands: Don't overlay selavy islands on png
                postagestamps, defaults to `True`.
            png_no_colorbar: Don't include colourbar on png output,
                defaults to `False`.
            png_crossmatch_overlay: Overlay the crossmatch radius on png
                postagestamps, defaults to `False`.
            png_hide_beam: Do not show the beam shape on png postagestamps,\
                defaults to `False`.
            ann_crossmatch_overlay: Include crossmatch radius in ann,
                defaults to `False`.
            reg_crossmatch_overlay: Include crossmatch radius in reg,
                defaults to `False`.
            lc_sigma_thresh: Detection threshold (in sigma) for
                lightcurves, defaults to 5.
            lc_figsize: Size of lightcurve figure, defaults to (8, 4).
            lc_min_points: Minimum number of source observations required
                for a lightcurve to be generated, defaults to 2.
            lc_min_detections: Minimum number of source detections required
                for a lightcurve to be generated, defaults to 1.
            lc_mjd: Use MJD for lightcurve x-axis, defaults to `False`.
            lc_start_date: Plot lightcurve in days from start date,
                defaults to None.
            lc_grid: Include grid on lightcurve plot, defaults to `False`.
            lc_yaxis_start: Start the lightcurve y-axis at 0 ('0') or use
                the matpotlib default ('auto'). Defaults to 'auto'.
            lc_peak_flux: Generate lightcurve using peak flux density
                rather than integrated flux density, defaults to `True`.
            lc_use_forced_for_limits: Generate lightcurves using forced
                photometry for non-detections only.
            lc_use_forced_for_all: Generate lightcurves using forced
                photometry for all measurements.
            measurements_simple: Use simple schema for measurement output,
                defaults to `False`.
            calc_script_norms: Calculate the png normalisation if it
                hasn't been already.
            plot_dpi: Specify the DPI of saved figures, defaults to 150.

        Returns:
            None
        """

        source, cutout_data = i

        if fits:
            source.save_all_fits_cutouts(cutout_data=cutout_data)

        if png:
            source.save_all_png_cutouts(
                selavy=png_selavy,
                percentile=png_percentile,
                zscale=png_zscale,
                contrast=png_contrast,
                no_islands=png_no_islands,
                no_colorbar=png_no_colorbar,
                crossmatch_overlay=png_crossmatch_overlay,
                hide_beam=png_hide_beam,
                disable_autoscaling=png_disable_autoscaling,
                cutout_data=cutout_data,
                calc_script_norms=calc_script_norms,
                plot_dpi=plot_dpi
            )

        if ann:
            source.save_all_ann(
                crossmatch_overlay=ann_crossmatch_overlay,
                cutout_data=cutout_data
            )

        if reg:
            source.save_all_reg(
                crossmatch_overlay=reg_crossmatch_overlay,
                cutout_data=cutout_data
            )

        if lightcurve:
            source.plot_lightcurve(
                sigma_thresh=lc_sigma_thresh,
                figsize=lc_figsize,
                min_points=lc_min_points,
                min_detections=lc_min_detections,
                mjd=lc_mjd,
                start_date=lc_start_date,
                grid=lc_grid,
                yaxis_start=lc_yaxis_start,
                peak_flux=lc_peak_flux,
                save=True,
                outfile=None,
                use_forced_for_limits=lc_use_forced_for_limits,
                use_forced_for_all=lc_use_forced_for_all,
                hide_legend=lc_hide_legend,
                plot_dpi=plot_dpi
            )

        if measurements:
            source.write_measurements(simple=measurements_simple)

        # attempt to avoid join hangs
        time.sleep(0.2)

        return

    def _summary_log(self) -> None:
        """
        Prints a summary log.

        Returns:
            None
        """
        self.logger.info("-------------------------")
        self.logger.info("Summary:")
        self.logger.info("-------------------------")
        self.logger.info(
            "Number of sources within footprint: %i",
            self.num_sources_searched
        )
        try:
            self.logger.info(
                "Number of sources with detections: %i",
                self.num_sources_detected
            )
        except Exception as e:
            # Means that find sources has not been run
            pass
        self.logger.info("-------------------------")

    def _add_source_cutout_data(self, s: Source) -> Source:
        """
        Add cutout data to the source of interest.

        Args:
            s: Source of interest.

        Returns:
            Updated source of interest containing the cutout data.
        """
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

    def _grouped_fetch_cutouts(
        self, group: pd.DataFrame, imsize: Angle
    ) -> pd.DataFrame:
        """
        Function that handles fetching the cutout data per
        group object, where the requested sources have been
        grouped by image.

        Args:
            group: Catalogue of sources grouped by field.
            imsize: Size of the requested cutout.

        Returns:
            Dataframe containing the cutout data for the group.
        """
        image_file = group.iloc[0]['image']

        try:
            image = Image(
                group.iloc[0].field,
                group.iloc[0].epoch,
                self.settings['stokes'],
                self.base_folder,
                sbid=group.iloc[0].sbid,
                tiles=self.settings['tiles']
            )

            image.get_img_data()

            cutout_data = group.apply(
                self._get_cutout,
                args=(image, imsize),
                axis=1,
                result_type='expand'
            ).rename(columns={
                0: "data",
                1: "wcs",
                2: "header",
                3: "selavy_overlay",
                4: "beam"
            })

            cutout_data['name'] = group['name'].values
            cutout_data['dateobs'] = group['dateobs'].values

            del image
        except Exception as e:
            cutout_data = pd.DataFrame(columns=[
                'data',
                'wcs',
                'header',
                'selavy_overlay',
                'beam',
                'name',
                'dateobs'
            ])

        return cutout_data

    def _get_cutout(
        self, row: pd.Series, image: Image,
        size: Angle = Angle(5. * u.arcmin)
    ) -> Tuple[pd.DataFrame, WCS, fits.Header, pd.DataFrame, Beam]:
        """
        Create cutout centered on a source location

        Args:
            row: Row of query catalogue corresponding to the source of
                interest
            image: Image to create cutout from.
            size: Size of the cutout, defaults to Angle(5.*u.arcmin).

        Returns:
            Tuple containing cutout data, WCS, image header, associated
            selavy components and beam information.
        """

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

    def find_sources(self) -> None:
        """
        Run source search. Results are stored in attributes.

        Steps:
        1. Run find_fields if not already run.
        2. Add the file paths to each measurement point.
        3. Obtain forced fits if requested.
        4. Run selavy matching and upper limit fetching.
        5. Package up results into vasttools.source.Source objects.

        Returns:
            None
        """
        self.logger.debug('Running find_sources...')

        if self.fields_found is False:
            self.find_fields()

        self.logger.info("Finding sources in PILOT data...")

        self.sources_df = self.fields_df.sort_values(
            by=['name', 'dateobs']
        ).reset_index(drop=True)

        self.sources_df[
            ['selavy', 'image', 'rms']
        ] = self.sources_df[['epoch', 'field', 'sbid']].apply(
            self._add_files,
            axis=1,
            result_type='expand'
        )

        if self.settings['forced_fits']:
            self.logger.info("Obtaining forced fits...")
            meta = {
                'f_island_id': 'U',
                'f_component_id': 'U',
                'f_ra_deg_cont': 'f',
                'f_dec_deg_cont': 'f',
                'f_flux_peak': 'f',
                'f_flux_peak_err': 'f',
                'f_flux_int': 'f',
                'f_flux_int_err': 'f',
                'f_chi_squared_fit': 'f',
                'f_rms_image': 'f',
                'f_maj_axis': 'f',
                'f_min_axis': 'f',
                'f_pos_ang': 'f',
            }

            f_results = (
                dd.from_pandas(self.sources_df, self.ncpu)
                .groupby('image')
                .apply(
                    self._get_forced_fits,
                    cluster_threshold=(
                        self.settings['forced_cluster_threshold']
                    ),
                    allow_nan=self.settings['forced_allow_nan'],
                    meta=meta,
                ).compute(num_workers=self.ncpu, scheduler='processes')
            )

            if not f_results.empty:
                if isinstance(f_results.index, pd.MultiIndex):
                    f_results.index = f_results.index.droplevel()
            else:
                self.settings['forced_fits'] = False

            self.logger.info("Done.")

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

        if self.settings['search_around']:
            meta['index'] = 'i'

        results = (
            dd.from_pandas(self.sources_df, self.ncpu)
            .groupby('selavy')
            .apply(
                self._get_components,
                meta=meta,
            ).compute(num_workers=self.ncpu, scheduler='processes')
        )

        if not results.empty:
            if isinstance(results.index, pd.MultiIndex):
                results.index = results.index.droplevel()

        if self.settings['search_around']:
            results = results.set_index('index')

        if self.settings['forced_fits']:
            results = results.merge(
                f_results, left_index=True, right_index=True
            )

            del f_results

        if self.settings['search_around']:
            how = 'inner'
        else:
            how = 'left'

        self.crossmatch_results = self.sources_df.merge(
            results, how=how, left_index=True, right_index=True
        )

        meta = {'name': 'O'}

        self.num_sources_detected = (
            self.crossmatch_results.groupby('name').agg({
                'detection': any
            }).sum()
        )

        if self.settings['search_around']:
            self.results = self.crossmatch_results.rename(
                columns={'#': 'distance'}
            )
        else:
            self.results = (
                dd.from_pandas(self.crossmatch_results, self.ncpu)
                .groupby('name')
                .apply(
                    self._init_sources,
                    meta=meta,
                ).compute(num_workers=self.ncpu, scheduler='processes')
            )
            self.results = self.results.dropna()

        self.logger.info("Done.")

    def save_search_around_results(self, sort_output: bool = False) -> None:
        """
        Save results from cone search.

        Args:
            sort_output: Whether to sort the output, defaults to `False`.

        Returns:
            None
        """
        meta = {}
        # also have the sort output setting as a function
        # input in case of interactive use.
        if self.settings['sort_output']:
            sort_output = True
        result = (
            dd.from_pandas(
                self.results.drop([
                    'fields',
                    'stokes',
                    'skycoord',
                    'selavy',
                    'image',
                    'rms',
                ], axis=1), self.ncpu)
            .groupby('name')
            .apply(
                self._write_search_around_results,
                sort_output=sort_output,
                meta=meta,
            ).compute(num_workers=self.ncpu, scheduler='processes')
        )

    def _write_search_around_results(
        self, group: pd.DataFrame, sort_output: bool
    ) -> None:
        """
        Write cone search results to file

        Args:
            group: The group from the pandas groupby function,
                which in this case is grouped by image.
            sort_output: Whether to sort the output.

        Returns:
            None
        """
        source_name = group['name'].iloc[0].replace(
            " ", "_"
        ).replace("/", "_")

        group = group.sort_values(by=['dateobs', 'component_id'])

        outname = "{}_matches_around.csv".format(
            source_name
        )

        if sort_output:
            base = os.path.join(
                self.settings['output_dir'],
                source_name
            )
        else:
            base = self.settings['output_dir']

        outname = os.path.join(
            base,
            outname
        )

        group.to_csv(outname, index=False)

        time.sleep(0.1)

    def _check_for_duplicate_epochs(self, epochs: pd.Series) -> pd.Series:
        """
        Checks whether a source has been detected in an epoch twice, which
        usually affects planets.

        If a duplicate is detected it adds `-N` to the epoch where N is the
        ith occurance of the epoch. E.g. 0, 0 is converted to 0-1, 0-2.

        Args:
            epochs: The epochs of the source.

        Returns:
            Corrected epochs.
        """
        dup_mask = epochs.duplicated(keep=False)
        if dup_mask.any():
            epochs.loc[dup_mask] = (
                epochs.loc[dup_mask]
                + "-"
                + (epochs[dup_mask].groupby(
                    epochs[dup_mask]
                ).cumcount() + 1).astype(str)
            )

        return epochs

    def _init_sources(self, group: pd.DataFrame) -> Source:
        """
        Initialises the vasttools.source.Source objects
        which are returned by find_sources.

        Args:
            group: The grouped measurements to initialise a source object.

        Returns:
            Source of interest.
        """
        group = group.sort_values(by='dateobs')

        m = group.iloc[0]

        if self.settings['matches_only']:
            if group['detection'].sum() == 0:
                self.logger.warning(
                    f"'{m['name']}' has no detections and 'matches only' "
                    "has been selected. This source will not be in the "
                    "results."
                )
                return
        if m['planet']:
            source_coord = group.skycoord
            source_primary_field = group.primary_field
            group['epoch'] = self._check_for_duplicate_epochs(
                group['epoch']
            )
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
        if self.settings['sort_output']:
            source_outdir = os.path.join(
                source_outdir,
                source_name.replace(" ", "_").replace("/", "_")
            )
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

        source_df = source_df.sort_values('dateobs').reset_index(drop=True)

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
            forced_fits=self.settings['forced_fits'],
            outdir=source_outdir,
        )

        return thesource

    def _get_forced_fits(
        self, group: pd.DataFrame,
        cluster_threshold: float = 1.5, allow_nan: bool = False
    ) -> pd.DataFrame:
        """
        Perform the forced fits on an image, on the coordinates
        supplied by the group.

        Args:
            group: A dataframe of sources/positions which have been
                supplied by grouping the queried sources by image.
            cluster_threshold: The cluster_threshold value passed to
                the forced photometry. Beam width distance limit for which a
                cluster is formed for extraction, defaults to 3.0.
            allow_nan: `allow_nan` value passed to the forced photometry.
                If False then any cluster containing a NaN is ignored.
                Defaults to False.

        Returns:
            Dataframe containing the forced fit measurements for each source.
        """
        image = group.name
        if image is None:
            return

        group = group.sort_values(by='dateobs')

        m = group.iloc[0]
        image_name = image.split("/")[-1]
        rms = m['rms']
        bkg = rms.replace('rms', 'bkg')

        field = m['field']
        epoch = m['epoch']
        stokes = m['stokes']

        try:
            img_beam = Image(
                field,
                epoch,
                stokes,
                self.base_folder
            )
            img_beam.get_img_data()
            img_beam = img_beam.beam
        except Exception as e:
            return pd.DataFrame(columns=[
                'f_island_id',
                'f_component_id',
                'f_ra_deg_cont',
                'f_dec_deg_cont',
                'f_flux_peak',
                'f_flux_peak_err',
                'f_flux_int',
                'f_flux_int_err',
                'f_chi_squared_fit',
                'f_rms_image',
                'f_maj_axis',
                'f_min_axis',
                'f_pos_ang',
            ])

        major = img_beam.major.to(u.arcsec).value
        minor = img_beam.minor.to(u.arcsec).value
        pa = img_beam.pa.to(u.deg).value

        to_fit = SkyCoord(
            group.ra, group.dec, unit=(u.deg, u.deg)
        )

        # make the Forced Photometry object
        FP = ForcedPhot(image, bkg, rms)

        # run the forced photometry
        (
            flux_islands, flux_err_islands,
            chisq_islands, DOF_islands, iscluster
        ) = FP.measure(
            to_fit,
            [major for i in range(to_fit.shape[0])] * u.arcsec,
            [minor for i in range(to_fit.shape[0])] * u.arcsec,
            [pa for i in range(to_fit.shape[0])] * u.deg,
            cluster_threshold=cluster_threshold,
            allow_nan=allow_nan
        )

        flux_islands *= 1.e3
        flux_err_islands *= 1.e3

        source_names = [
            "{}_{:04d}".format(
                image_name, i
            ) for i in range(len(flux_islands))
        ]

        data = {
            'f_island_id': source_names,
            'f_component_id': source_names,
            'f_ra_deg_cont': group.ra,
            'f_dec_deg_cont': group.dec,
            'f_flux_peak': flux_islands,
            'f_flux_peak_err': flux_err_islands,
            'f_flux_int': flux_islands,
            'f_flux_int_err': flux_err_islands,
            'f_chi_squared_fit': chisq_islands,
            'f_rms_image': flux_err_islands,
        }

        df = pd.DataFrame(data)

        df['f_maj_axis'] = major
        df['f_min_axis'] = minor
        df['f_pos_ang'] = pa

        df.index = group.index.values

        return df

    def _get_components(self, group: pd.DataFrame) -> pd.DataFrame:
        """
        Obtains the matches from the selavy catalogue for each coordinate
        in the group. The group is the queried sources grouped by image
        (the result from find_fields). If no component is found then the
        rms is measured at the source location.

        Args:
            group: The grouped coordinates to search in the image.

        Returns:
            The selavy matched component and/or upper limits for the queried
            coordinates.
        """
        selavy_file = str(group.name)

        if selavy_file is None:
            return

        master = pd.DataFrame()

        selavy_df = pd.read_fwf(
            selavy_file, skiprows=[1, ]
        )

        if self.settings['stokes'] != "I":
            head, tail = os.path.split(selavy_file)
            nselavy_file = os.path.join(head, 'n{}'.format(tail))
            nselavy_df = pd.read_fwf(
                nselavy_file, skiprows=[1, ]
            )

            nselavy_df[["flux_peak", "flux_int"]] *= -1.0

            selavy_df = selavy_df.append(
                nselavy_df, ignore_index=True, sort=False
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

        if self.settings['search_around']:
            idxselavy, idxc, d2d, _ = group_coords.search_around_sky(
                selavy_coords, self.settings['crossmatch_radius']
            )
            if idxc.shape[0] == 0:
                return
            copy = selavy_df.iloc[idxselavy].reset_index(drop=True)
            copy['detection'] = True
            copy['#'] = d2d.arcsec
            copy.index = group.iloc[idxc].index.values
            master = master.append(copy, sort=False)
            # reset index and move previous index to the end to match the meta
            master_cols = master.columns.to_list()
            master = master.reset_index()[master_cols + ['index']]
        else:
            idx, d2d, _ = group_coords.match_to_catalog_sky(selavy_coords)
            mask = d2d < self.settings['crossmatch_radius']
            idx_matches = idx[mask]

            copy = selavy_df.iloc[idx_matches].reset_index(drop=True)
            copy['detection'] = True
            copy.index = group[mask].index.values

            master = master.append(copy, sort=False)

            missing = group_coords[~mask]
            if missing.shape[0] > 0:
                if not self.settings['no_rms']:
                    try:
                        image = Image(
                            group.iloc[0].field,
                            group.iloc[0].epoch,
                            self.settings['stokes'],
                            self.base_folder,
                            sbid=group.iloc[0].sbid,
                            tiles=self.settings['tiles']
                        )
                        image.get_img_data()
                        rms_values = image.measure_coord_pixel_values(
                            missing, rms=True
                        )
                        rms_df = pd.DataFrame(
                            rms_values, columns=['rms_image'])

                        # to mJy
                        rms_df['rms_image'] = rms_df['rms_image'] * 1.e3
                    except Exception as e:
                        rms_df = pd.DataFrame(
                            [-99 for i in range(missing.shape[0])],
                            columns=['rms_image']
                        )
                else:
                    rms_df = pd.DataFrame(
                        [-99 for i in range(missing.shape[0])],
                        columns=['rms_image']
                    )

                rms_df['detection'] = False

                rms_df.index = group[~mask].index.values

                master = master.append(rms_df, sort=False)

        return master

    def _add_files(self, row: pd.Series) -> Tuple[str, str, str]:
        """
        Adds the file paths for the image, selavy catalogues and
        rms images for each source to be queried.

        Args:
            row: The input row of the dataframe (this function is called with
                a .apply())

        Returns:
            The paths of the image, selavy catalogue and rms image.
        """
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

    def write_find_fields(self, outname: Optional[str] = None) -> None:
        """
        Write the results of a field search to file.

        Args:
            outname: Name of file to write output to, defaults to None, which
                will name the file 'find_fields_results.csv'.

        Returns:
            None
        """
        if self.fields_found is False:
            self.find_fields()

        if outname is None:
            name = 'find_fields_result.csv'
        else:
            name = outname + '.pkl'

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

    def find_fields(self) -> None:
        """
        Find the corresponding field for each source.

        Planet fields are also found here if any are selected.

        Returns:
            None

        Raises:
            Exception: No sources are found within the requested footprint.
        """
        self.logger.info(
            "Matching queried sources to VAST Pilot fields..."
        )

        if self.racs:
            base_epoch = '0'
            base_fc = 'RACS'
        else:
            base_epoch = '1'
            base_fc = 'VAST'

        fields = Fields(base_epoch)
        field_centres = load_field_centres()
        field_centres = field_centres.loc[
            field_centres['field'].str.contains(base_fc)
        ].reset_index()

        field_centres_sc = SkyCoord(
            field_centres["centre-ra"],
            field_centres["centre-dec"],
            unit=(u.deg, u.deg)
        )

        # if RACS is being use we convert all the names to 'VAST'
        # to match the VAST field names, makes matching easier.
        if self.racs:
            field_centres['field'] = [
                f.replace("RACS", "VAST") for f in field_centres.field
            ]

        field_centre_names = field_centres.field

        if self.query_df is not None:
            self.fields_df = self.query_df.copy()

            meta = {
                0: 'O',
                1: 'U',
                2: 'O',
                3: 'O',
                4: 'O',
                5: 'O',
                6: 'O',
            }
            self.logger.debug("Running field matching...")
            self.fields_df[[
                'fields',
                'primary_field',
                'epochs',
                'field_per_epoch',
                'sbids',
                'dates',
                'freqs'
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
            self.logger.debug("Finished field matching.")
            self.fields_df = self.fields_df.dropna()
            if self.fields_df.empty:
                raise Exception(
                    "No requested sources are within the requested footprint!")

            self.fields_df = self.fields_df.explode(
                'field_per_epoch'
            ).reset_index(drop=True)

            self.fields_df[
                ['epoch', 'field', 'sbid', 'dateobs', 'obs_freq']
            ] = pd.DataFrame(
                self.fields_df['field_per_epoch'].tolist(),
                index=self.fields_df.index
            )

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

        # Handle Planets
        if self.planets is not None:
            planet_fields = self._search_planets()

            if self.fields_df is None:
                self.fields_df = planet_fields
            else:
                self.fields_df = self.fields_df.append(
                    planet_fields
                ).reset_index(drop=True)

        self.logger.debug(self.fields_df)

        if self.query_df is None:
            prev_num = 0
        else:
            prev_num = self.query_df.shape[0]

        if self.planets is not None:
            prev_num += len(self.planets)

        self.num_sources_searched = self.fields_df.name.unique().shape[0]

        if self.racs:
            self.logger.info(
                "%i/%i sources in RACS & VAST Pilot footprint.",
                self.num_sources_searched,
                prev_num
            )
        else:
            self.logger.info(
                "%i/%i sources in VAST Pilot footprint.",
                self.num_sources_searched,
                prev_num
            )

        self.fields_df['dateobs'] = pd.to_datetime(
            self.fields_df['dateobs']
        )

        self.logger.info("Done.")
        self.fields_found = True

    def _field_matching(
        self,
        row: pd.Series,
        fields_coords: SkyCoord,
        fields_names: pd.Series,
        field_centres: SkyCoord,
        field_centre_names: List[str]
    ) -> Tuple[
        str, str, List[str], List[str], List[str], List[datetime.datetime]
    ]:
        """
        This function does the actual field matching for each queried
        coordinate, which is a 'row' here in the function.

        Args:
            row: The row from the query_df, i.e. the coordinates to match
                to a field.
            fields_coords: SkyCoord object representing the beam
                centres of the VAST or RACS survey.
            fields_names: Field names to match with the SkyCoord object.
            field_centres: SkyCoord object representing the field centres
            field_centre_names: Field names matching the field centre
                SkyCoord.

        Returns:
            Tuple containing the field information.
        """
        seps = row.skycoord.separation(fields_coords)
        accept = seps.deg < self.settings['max_sep']
        fields = np.unique(fields_names[accept])
        if self.racs:
            vast_fields = np.array(
                [f.replace("RACS", "VAST") for f in fields]
            )

        if fields.shape[0] == 0:
            if self.racs:
                self.logger.info(
                    "Source '%s' not in RACS & VAST Pilot footprint.",
                    row['name']
                )
            else:
                self.logger.info(
                    "Source '%s' not in VAST Pilot footprint.",
                    row['name']
                )
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        centre_seps = row.skycoord.separation(field_centres)
        primary_field = field_centre_names.iloc[np.argmin(centre_seps.deg)]
        epochs = []
        field_per_epochs = []
        sbids = []
        dateobs = []
        freqs = []

        for i in self.settings['epochs']:

            if i != '0' and self.racs:
                the_fields = vast_fields
            else:
                the_fields = fields

            available_fields = [
                f for f in the_fields if f in self._epoch_fields.loc[
                    i
                ].index.to_list()
            ]

            if i == '0':
                available_fields = [
                    j.replace("RACS", "VAST") for j in available_fields
                ]

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

            # Change VAST back to RACS
            if i == '0':
                field = field.replace("VAST", "RACS")
            epochs.append(i)
            sbid = self._epoch_fields.loc[i, field]["SBID"]
            date = self._epoch_fields.loc[i, field]["DATEOBS"]
            freq = self._epoch_fields.loc[i, field]["OBS_FREQ"]
            sbids.append(sbid)
            dateobs.append(date)
            freqs.append(freq)
            field_per_epochs.append([i, field, sbid, date, freq])

        return_vals = (fields,
                       primary_field,
                       epochs,
                       field_per_epochs,
                       sbids,
                       dateobs,
                       freqs
                       )
        return return_vals

    def _get_planets_epoch_df_template(self) -> pd.DataFrame:
        """
        Generate template df for fields containing planets in all epochs

        Returns:
            Dataframe containing fields and epoch info.
        """
        epochs = self.settings['epochs']
        field_centres = load_field_centres()

        planet_epoch_fields = self._epoch_fields.loc[epochs].reset_index()

        planet_epoch_fields = planet_epoch_fields.merge(
            field_centres, left_on='FIELD_NAME',
            right_on='field', how='left'
        ).drop('field', axis=1).rename(
            columns={'EPOCH': 'epoch'}
        )

        return planet_epoch_fields

    def _search_planets(self) -> pd.DataFrame:
        """
        Search for planets in all requested epochs

        Returns:
            Dataframe containing search results
        """
        template = self._get_planets_epoch_df_template()

        template['planet'] = [self.planets for i in range(template.shape[0])]

        template = template.explode('planet')
        template['planet'] = template['planet'].str.capitalize()

        meta = {
            'epoch': 'U',
            'FIELD_NAME': 'U',
            'SBID': 'i',
            'DATEOBS': 'datetime64[ns]',
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

    def _build_catalog(self) -> pd.DataFrame:
        """
        Generate source catalogue from requested coordinates,
        removing those outside of the VAST pilot fields.

        Returns:
            Catalogue of source positions.
        """
        cols = ['ra', 'dec', 'name', 'skycoord', 'stokes']

        if '0' in self.settings['epochs']:
            mask = self.coords.dec.deg > 42

            if mask.any():
                self.logger.warning(
                    "Removing %i sources outside the RACS area", sum(mask)
                )
                self.coords = self.coords[~mask]
                self.source_names = self.source_names[~mask]
        else:
            mocs = VASTMOCS()
            vast_pilot_moc = mocs.load_pilot_epoch_moc('1')
            mask = vast_pilot_moc.contains(
                self.coords.ra, self.coords.dec, keep_inside=False
            )
            if mask.any():
                self.logger.warning(
                    "Removing %i sources outside"
                    " the VAST Pilot Footprint", sum(mask)
                )
                self.coords = self.coords[~mask]
                self.source_names = self.source_names[~mask]

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

        if self.simbad_names is not None:
            self.simbad_names = self.simbad_names[~mask]
            catalog['simbad_name'] = self.simbad_names

        return catalog

    def _get_epochs(self, req_epochs: str) -> List[str]:
        """
        Parse the list of epochs to query.

        Args:
            req_epochs: Requested epochs to query.

        Returns:
            Epochs to query, as a list of strings.
        """

        available_epochs = sorted(RELEASED_EPOCHS, key=RELEASED_EPOCHS.get)
        self.logger.debug("Available epochs: " + str(available_epochs))

        if req_epochs == 'all':
            epochs = available_epochs
        elif req_epochs == 'all-vast':
            epochs = available_epochs
            epochs.remove('0')
        else:
            epochs = []
            for epoch in req_epochs.split(','):
                if epoch in available_epochs:
                    epochs.append(epoch)
                else:
                    if self.logger is None:
                        self.logger.info(
                            "Epoch {} is not available. Ignoring.".format(
                                epoch
                            )
                        )
                    else:
                        warnings.warn(
                            "Removing Epoch {} as it"
                            " is not a valid epoch.".format(epoch),
                            stacklevel=2
                        )

        # RACS check
        if '0' in epochs:
            if not check_racs_exists(self.base_folder):
                self.logger.warning('RACS EPOCH00 directory not found!')
                self.logger.warning('Removing RACS from requested epochs.')
                epochs.remove('0')
                self.racs = False
            else:
                self.logger.warning('RACS data selected!')
                self.logger.warning(
                    'Remember RACS data supplied by VAST is not final '
                    'and results may vary.'
                )
                self.racs = True
        else:
            self.racs = False

        if len(epochs) == 0:
            self.logger.critical("No requested epochs are available")
            sys.exit()

        return epochs

    def _get_stokes(self, req_stokes: str) -> str:
        """
        Set the stokes Parameter

        Args:
            req_stokes: Requested stokes parameter to check.

        Returns:
            Valid stokes parameter.

        Raises:
            ValueError: Entered Stokes parameter is not valid.
            ValueError: Stokes V is not supported with RACS.
        """

        valid = ["I", "Q", "U", "V"]

        if req_stokes.upper() not in valid:
            raise ValueError(
                "Stokes {} is not valid!".format(req_stokes.upper())
            )
        elif self.racs and req_stokes.upper() == 'V':
            raise ValueError(
                "Stokes V is not supported with RACS!"
            )
        else:
            return req_stokes.upper()


class FieldQuery:
    """
    This is a class representation of a query of the VAST Pilot survey
    fields, returning basic information such as observation dates and psf
    information.

    Attributes:
        field (str): Name of requested field.
        valid (bool): Confirm the requested field exists.
        pilot_info (pandas.core.frame.DataFrame):
            Dataframe describing the pilot survey.
        field_info (pandas.core.frame.DataFrame):
            Dataframe describing properties of the field.
        epochs (pandas.core.frame.DataFrame):
            Dataframe containing epochs this field was observed in.
    """

    def __init__(self, field: str) -> None:
        """Constructor method

        Args:
            field: Name of requested field.

        Returns:
            None
        """
        self.logger = logging.getLogger('vasttools.query.FieldQuery')

        self.field = field
        self.valid = self._check_field()

    def _check_field(self) -> bool:
        """
        Check that the field is a valid pilot survey field.

        Epoch 1 is checked against as it is a complete observation.

        Returns:
            Bool representing if field is valid.
        """

        epoch_01 = load_fields_file("1")
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

    def _get_beams(self) -> Dict[str, Beams]:
        """
        Processes all the beams of a field per epoch and initialises
        radio_beam.Beams objects.

        Returns:
            Dictionary of 'radio_beam.Beams' objects.
        """
        epoch_beams = {}
        for e in self.settings['epochs']:
            epoch_cut = self.field_info[self.field_info.EPOCH == e]
            epoch_beams[e] = Beams(
                epoch_cut.BMAJ.values * u.arcsec,
                epoch_cut.BMIN.values * u.arcsec,
                epoch_cut.BPA.values * u.deg
            )
        return epoch_beams

    def run_query(
            self,
            psf: bool = False,
            largest_psf: bool = False,
            common_psf: bool = False,
            all_psf: bool = False,
            save: bool = False,
            _pilot_info: Optional[pd.DataFrame] = None
    ) -> None:
        """Running the field query.

        Args:
            largest_psf: If true the largest psf is calculated
                of the field per epoch. Defaults to False.
            common_psf: If true the common psf is calculated
                of the field per epoch. Defaults to False.
            all_psf: If true the common psf is calculated of the field
                per epoch and all the beam information of
                the field is shown. Defaults to False.
            save: Save the output tables to a csv file. Defaults to False.
            _pilot_info: Allows for the pilot info to be provided
                rather than the function building it locally. If not provided
                then the dataframe is built. Defaults to None.

        Returns:
            None
        """
        if not self.valid:
            self.logger.error("Field doesn't exist.")
            return

        if _pilot_info is not None:
            self.pilot_info = _pilot_info
        else:
            self.logger.debug("Building pilot info file.")
            for i, val in enumerate(sorted(RELEASED_EPOCHS)):
                if i == 0:
                    self.pilot_info = load_fields_file(val)
                    self.pilot_info["EPOCH"] = RELEASED_EPOCHS[val]
                else:
                    to_append = load_fields_file(val)
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
