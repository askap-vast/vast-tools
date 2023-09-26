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
import re
import signal
import numexpr
import gc
import time
import dask.dataframe as dd
import logging
import logging.handlers
import logging.config

from astropy.coordinates import Angle
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.nddata.utils import Cutout2D
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.exceptions import AstropyWarning, AstropyDeprecationWarning
from astropy.nddata.utils import NoOverlapError

from functools import partial

from multiprocessing import Pool, cpu_count
from multiprocessing_logging import install_mp_handler

from radio_beam import Beams, Beam

from tabulate import tabulate

from typing import Optional, List, Tuple, Dict, Union

from pathlib import Path

from vasttools import (
    RELEASED_EPOCHS, OBSERVED_EPOCHS, ALLOWED_PLANETS, BASE_EPOCHS,
    RACS_EPOCHS, P1_EPOCHS, P2_EPOCHS
)
from vasttools.survey import Fields, Image
from vasttools.survey import (
    load_fields_file, load_field_centres, get_fields_per_epoch_info
)
from vasttools.source import Source
from vasttools.utils import (
    filter_selavy_components, simbad_search, match_planet_to_field,
    read_selavy, strip_fieldnames
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
        epochs: Union[str, List[str], List[int]] = "1",
        stokes: str = "I",
        crossmatch_radius: float = 5.0,
        max_sep: float = 1.5,
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
        forced_allow_nan: bool = False,
        incl_observed: bool = False,
        corrected_data: bool = True,
        search_all_fields: bool = False,
        scheduler: str = 'processes',
    ) -> None:
        """
        Constructor method.

        Args:
            coords: List of coordinates to query, defaults to None.
            source_names: List of source names, defaults to None.
            epochs: Epochs to query. Can be specified with either a list
                or a comma-separated string. All available epochs can be
                queried by passing "all", and all available VAST epochs can be
                queried by passing "all-vast". Defaults to "1".
            stokes: Stokes parameter to query.
            crossmatch_radius: Crossmatch radius in arcsec, defaults to 5.0.
            max_sep: Maximum separation of source from beam centre
                in degrees, defaults to 1.5.
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
            incl_observed: Include epochs that have been observed, but not
                released, in the query. This should only be used when finding
                fields, not querying data. Defaults to False.
            corrected_data: Access the corrected data. Only relevant if
                `tiles` is `True`. Defaults to `True`.
            search_all_fields: If `True`, return all data at the requested
                positions regardless of field. If `False`, only return data
                from the best (closest) field in each epoch.
            scheduler: Dask scheduling option to use. Options are "processes"
                (parallel processing) or "single-threaded". Defaults to
                "processes".

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
            QueryInitError: Invalid scheduler option requested.
        """
        self.logger = logging.getLogger('vasttools.find_sources.Query')

        install_mp_handler(logger=self.logger)

        if source_names is None:
            source_names = []
        if planets is None:
            planets = []

        self.source_names = np.array(source_names)
        self.simbad_names = None

        self.corrected_data = corrected_data

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
        self.logger.debug(f"Using {self.ncpu} CPUs")

        if coords is None and len(source_names) == 0 and len(planets) == 0:
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

        self.settings['incl_observed'] = incl_observed
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
        self.settings['search_all_fields'] = search_all_fields

        scheduler_options = ['processes', 'single-threaded']
        if scheduler not in scheduler_options:
            raise QueryInitError(
                f"{scheduler} is not a suitable scheduler option. Please "
                f"select from {scheduler_options}"
            )
        self.settings['scheduler'] = scheduler

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

        all_data_available = self._check_data_availability()
        if all_data_available:
            self.logger.info("All data available!")
        else:
            self.logger.warning(
                "Not all requested data is available! See above for details."
            )
            self.logger.warning(
                "Query will continue run, but proceed with caution."
            )

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

    def _validate_settings(self) -> bool:
        """
        Used to check that the settings are valid.

        Returns:
            `True` if settings are acceptable, `False` otherwise.
        """

        self.logger.debug("Using settings: ")
        self.logger.debug(self.settings)

        if self.settings['tiles'] and self.settings['stokes'].lower() != "i":
            if self.vast_full:
                self.logger.warning("Stokes V tiles are only available for the"
                                    " full VAST survey. Proceed with caution!"
                                    )
            else:
                self.logger.critical("Stokes V tiles are only available for "
                                     "the full VAST survey."
                                     )
                return False

        if self.settings['tiles'] and self.settings['islands']:
            if self.vast_p1 or self.vast_p2 or self.racs:
                self.logger.critical(
                    "Only component catalogues are supported with tiles "
                    "for the selected epochs."
                )
                return False

        if self.settings['islands']:
            self.logger.warning(
                "Image RMS and peak flux error are not available with islands."
                "Using background_noise as a placeholder for both."
            )

        if self.vast_full and not self.settings['tiles']:
            self.logger.critical("COMBINED images are not available for "
                                 "the full VAST survey."
                                 )
            return False

        if self.settings['tiles'] and self.corrected_data and self.vast_full:
            self.logger.critical(
                "Corrected data does not yet exist for the full VAST survey."
                "Pass corrected_data=False to access full survey data. "
                "Query will continue to run, but proceed with caution."
            )

        return True

    def _check_data_availability(self) -> bool:
        """
        Used to check that the requested data is available.

        Returns:
            `True` if all data is available, `False` otherwise.
        """

        all_available = True

        base_dir = Path(self.base_folder)

        data_type = "COMBINED"
        corrected_str = ""

        if self.settings['tiles']:
            data_type = "TILES"
            if self.corrected_data:
                corrected_str = "_CORRECTED"

        stokes = self.settings['stokes']

        self.logger.info("Checking data availability...")

        for epoch in self.settings['epochs']:
            epoch_dir = base_dir / "EPOCH{}".format(RELEASED_EPOCHS[epoch])
            if not epoch_dir.is_dir():
                self.logger.critical(f"Epoch {epoch} is unavailable.")
                self.logger.debug(f"{epoch_dir} does not exist.")
                all_available = False
                continue

            data_dir = epoch_dir / data_type
            if not data_dir.is_dir():
                self.logger.critical(
                    f"{data_type} data unavailable for epoch {epoch}"
                )
                self.logger.debug(f"{data_dir} does not exist.")
                all_available = False
                continue

            image_dir = data_dir / f"STOKES{stokes}_IMAGES{corrected_str}"
            if not image_dir.is_dir():
                self.logger.critical(
                    f"Stokes {stokes} images unavailable for epoch {epoch}"
                )
                self.logger.debug(f"{image_dir} does not exist.")
                all_available = False

            selavy_dir = data_dir / f"STOKES{stokes}_SELAVY{corrected_str}"
            if not selavy_dir.is_dir():
                self.logger.critical(
                    f"Stokes {stokes} catalogues unavailable for epoch {epoch}"
                )
                self.logger.debug(f"{selavy_dir} does not exist.")
                all_available = False

            rms_dir = data_dir / f"STOKES{stokes}_RMSMAPS{corrected_str}"
            if not rms_dir.is_dir() and not self.settings["no_rms"]:
                self.logger.critical(
                    f"Stokes {stokes} RMS maps unavailable for epoch {epoch}"
                )
                self.logger.debug(f"{rms_dir} does not exist.")
                all_available = False

        if all_available:
            self.logger.info("All requested data is available!")

        return all_available

    def _get_all_cutout_data(self,
                             imsize: Angle,
                             img: bool = True,
                             rms: bool = False,
                             bkg: bool = False
                             ) -> pd.DataFrame:
        """
        Get cutout data and selavy components for all sources.

        Args:
            imsize: Size of the requested cutout.
            img: Fetch image data, defaults to `True`.
            rms: Fetch rms data, defaults to `False`.
            bkg: Fetch bkg data, defaults to `False`.

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
            'rms_data': 'O',
            'rms_wcs': 'O',
            'rms_header': 'O',
            'bkg_data': 'O',
            'bkg_wcs': 'O',
            'bkg_header': 'O',
            'name': 'U',
            'dateobs': 'datetime64[ns]',
        }

        cutouts = (
            dd.from_pandas(self.sources_df, self.ncpu)
            .groupby('image')
            .apply(
                self._grouped_fetch_cutouts,
                imsize=imsize,
                meta=meta,
                img=img,
                rms=rms,
                bkg=bkg,
            ).compute(num_workers=self.ncpu,
                      scheduler=self.settings['scheduler']
                      )
        )

        if not cutouts.empty:
            if isinstance(cutouts.index, pd.MultiIndex):
                cutouts.index = cutouts.index.droplevel()

        return cutouts

    def _gen_all_source_products(
        self,
        fits: bool = True,
        rms: bool = False,
        bkg: bool = False,
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
        png_offset_axes: bool = True,
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
            rms: Create and save fits cutouts of the rms images,
                defaults to `True`.
            bkg: Create and save fits cutouts of the bkg images,
                defaults to `True`.
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
            png_disable_autoscaling: Disable autoscaling.
            png_offset_axes: Use offset, rather than absolute, axis labels.
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

        if sum([fits, rms, bkg, png, ann, reg]) > 0:
            self.logger.info(
                "Fetching cutout data for sources..."
            )
            cutouts_df = self._get_all_cutout_data(imsize,
                                                   img=fits,
                                                   rms=rms,
                                                   bkg=bkg,
                                                   )
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
            rms=rms,
            bkg=bkg,
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
            png_offset_axes=png_offset_axes,
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

        if self.settings['scheduler'] == 'processes':
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
        elif self.settings['scheduler'] == 'single-threaded' or self.ncpu == 1:
            for result in map(produce_source_products_multi, to_process):
                pass

    def _produce_source_products(
        self,
        i: Tuple[Source, pd.DataFrame],
        fits: bool = True,
        rms: bool = False,
        bkg: bool = False,
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
        png_offset_axes: bool = True,
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
            rms: Create and save fits cutouts of the rms images,
                defaults to `True`.
            bkg: Create and save fits cutouts of the bkg images,
                defaults to `True`.
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
            png_disable_autoscaling: Disable autoscaling.
            png_offset_axes: Use offset, rather than absolute, axis labels.
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

        self.logger.debug(f"Producing source products for {source.name}")

        if fits:
            source.save_all_fits_cutouts(cutout_data=cutout_data)
        if sum([rms, bkg]) > 1:
            source._save_all_noisemap_cutouts(cutout_data, rms=rms, bkg=bkg)

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
                plot_dpi=plot_dpi,
                offset_axes=png_offset_axes
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
        self,
        group: pd.DataFrame,
        imsize: Angle,
        img: bool = True,
        rms: bool = False,
        bkg: bool = False
    ) -> pd.DataFrame:
        """
        Function that handles fetching the cutout data per
        group object, where the requested sources have been
        grouped by image.

        Args:
            group: Catalogue of sources grouped by field.
            imsize: Size of the requested cutout.
            img: Fetch image data, defaults to `True`.
            rms: Fetch rms data, defaults to `False`.
            bkg: Fetch bkg data, defaults to `False`.

        Returns:
            Dataframe containing the cutout data for the group.
        """
        image_file = group.iloc[0]['image']
        self.logger.debug(f"Fetching cutouts from {image_file}")

        try:
            image = Image(
                group.iloc[0].field,
                group.iloc[0].epoch,
                self.settings['stokes'],
                self.base_folder,
                sbid=group.iloc[0].sbid,
                tiles=self.settings['tiles'],
                corrected_data=self.corrected_data
            )

            if img:
                image.get_img_data()
                img_cutout_data = group.apply(
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
            else:
                img_cutout_data = pd.DataFrame([[None] * 5] * len(group),
                                               columns=[
                    'data',
                    'wcs',
                    'header',
                    'selavy_overlay',
                    'beam',
                ]
                )
            if rms:
                image.get_rms_img()
                rms_cutout_data = group.apply(
                    self._get_cutout,
                    args=(image, imsize),
                    img=False,
                    rms=True,
                    axis=1,
                    result_type='expand'
                ).rename(columns={
                    0: "rms_data",
                    1: "rms_wcs",
                    2: "rms_header",
                }).drop(columns=[3, 4])
            else:
                rms_cutout_data = pd.DataFrame([[None] * 3] * len(group),
                                               columns=[
                    'rms_data',
                    'rms_wcs',
                    'rms_header',
                ]
                )

            if bkg:
                image.get_bkg_img()
                bkg_cutout_data = group.apply(
                    self._get_cutout,
                    args=(image, imsize),
                    img=False,
                    bkg=True,
                    axis=1,
                    result_type='expand'
                ).rename(columns={
                    0: "bkg_data",
                    1: "bkg_wcs",
                    2: "bkg_header",
                }).drop(columns=[3, 4])
            else:
                bkg_cutout_data = pd.DataFrame([[None] * 3] * len(group),
                                               columns=[
                    'bkg_data',
                    'bkg_wcs',
                    'bkg_header',
                ]
                )

            self.logger.debug("Generated all cutout data")

            to_concat = [img_cutout_data, rms_cutout_data, bkg_cutout_data]
            cutout_data = pd.concat(to_concat, axis=1).dropna(how='all')

            self.logger.debug("Concatenated into cutout_data")

            if bkg or rms:
                bkg_values = bkg_cutout_data['bkg_data'].values
                rms_values = rms_cutout_data['rms_data'].values
                if bkg_values == rms_values:
                    self.logger.warning("Bkg and RMS data are identical!")

            self.logger.debug(cutout_data.columns)
            self.logger.debug(len(cutout_data))
            self.logger.debug(group['name'].values)

            cutout_data['name'] = group['name'].values
            self.logger.debug(cutout_data['name'])
            cutout_data['dateobs'] = group['dateobs'].values
            self.logger.debug(cutout_data['dateobs'])

            del image
        except Exception as e:
            self.logger.warning(
                "Caught exception inside _grouped_fetch_cutouts")
            self.logger.warning(e)
            cutout_data = pd.DataFrame(columns=[
                'data',
                'wcs',
                'header',
                'selavy_overlay',
                'beam',
                'name',
                'dateobs',
                'rms_data',
                'rms_wcs',
                'rms_header',
                'bkg_data',
                'bkg_wcs',
                'bkg_header',
            ])

        return cutout_data

    def _get_cutout(
        self,
        row: pd.Series,
        image: Image,
        size: Angle = Angle(5. * u.arcmin),
        img: bool = True,
        rms: bool = False,
        bkg: bool = False
    ) -> Tuple[pd.DataFrame, WCS, fits.Header, pd.DataFrame, Beam]:
        """
        Create cutout centered on a source location

        Args:
            row: Row of query catalogue corresponding to the source of
                interest
            image: Image to create cutout from.
            size: Size of the cutout, defaults to Angle(5.*u.arcmin).
            img: Make a cutout from the image data, defaults to `True`.
            rms: Make a cutout from the rms data, defaults to `False`.
            bkg: Make a cutout from the bkg data, defaults to `False`.

        Returns:
            Tuple containing cutout data, WCS, image header, associated
            selavy components and beam information.

        Raises:
            ValueError: Exactly one of img, rms or bkg must be `True`
        """

        if sum([img, rms, bkg]) != 1:
            raise ValueError("Exactly one of img, rms or bkg must be True")

        if img:
            thedata = image.data
            thewcs = image.wcs
            theheader = image.header.copy()
            thepath = image.imgpath
        elif rms:
            thedata = image.rms_data
            thewcs = image.rms_wcs
            theheader = image.rms_header.copy()
            thepath = image.rmspath
        elif bkg:
            thedata = image.bkg_data
            thewcs = image.bkg_wcs
            theheader = image.bkg_header.copy()
            thepath = image.bkgpath

        self.logger.debug(f"Using data from {thepath}")

        try:
            cutout = Cutout2D(
                thedata,
                position=row.skycoord,
                size=size,
                wcs=thewcs
            )
        except NoOverlapError:
            self.logger.warning(f"Unable to create cutout for {row['name']}.")
            self.logger.warning(f"Image path: {thepath}")
            self.logger.warning(f"Coordinate: {row.skycoord.to_string()}")
            return (None, None, None, None, None)

        if img:
            selavy_components = read_selavy(row.selavy, cols=[
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

            del selavy_coords

            beam = image.beam
        else:
            beam = None
            selavy_components = None

        theheader.update(cutout.wcs.to_header())

        return (
            cutout.data, cutout.wcs, theheader, selavy_components, beam
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

        Raises:
            Exception: find_sources cannot be run with the incl_observed option
        """

        if self.settings['incl_observed']:
            raise Exception(
                'find_sources cannot be run with the incl_observed option'
            )

        self.logger.debug('Running find_sources...')

        if self.fields_found is False:
            self.find_fields()

        self.logger.info("Finding sources in VAST data...")

        self.sources_df = self.fields_df.sort_values(
            by=['name', 'dateobs']
        ).reset_index(drop=True)

        self.logger.debug("Adding files...")
        self.sources_df[
            ['selavy', 'image', 'rms']
        ] = self.sources_df[['epoch', 'field', 'sbid']].apply(
            self._add_files,
            axis=1,
            result_type='expand'
        )

        self._validate_files()

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
                ).compute(num_workers=self.ncpu,
                          scheduler=self.settings['scheduler']
                          )
            )

            if not f_results.empty:
                if isinstance(f_results.index, pd.MultiIndex):
                    f_results.index = f_results.index.droplevel()
            else:
                self.settings['forced_fits'] = False

            self.logger.info("Forced fitting finished.")

        self.logger.debug("Getting components...")
        results = (
            dd.from_pandas(self.sources_df, self.ncpu)
            .groupby('selavy')
            .apply(
                self._get_components,
                meta=self._get_selavy_meta(),
            ).compute(num_workers=self.ncpu,
                      scheduler=self.settings['scheduler']
                      )
        )

        self.logger.debug("Selavy components succesfully added.")
        self.logger.debug(results)

        if self.settings['islands']:
            results['rms_image'] = results['background_noise']
            results['flux_peak_err'] = results['background_noise']

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
        self.logger.debug("Crossmatch results:")
        self.logger.debug(self.crossmatch_results)

        meta = {'name': 'O'}

        self.num_sources_detected = (
            self.crossmatch_results.groupby('name').agg({
                'detection': any
            }).sum()
        )
        self.logger.debug(f"{self.num_sources_detected} sources detected:")

        if self.settings['search_around']:
            self.results = self.crossmatch_results.rename(
                columns={'#': 'distance'}
            )
        else:
            npart = min(self.ncpu, self.crossmatch_results.name.nunique())
            self.results = (
                dd.from_pandas(self.crossmatch_results, npart)
                .groupby('name')
                .apply(
                    self._init_sources,
                    meta=meta,
                ).compute(num_workers=npart,
                          scheduler=self.settings['scheduler']
                          )
            )
            self.results = self.results.dropna()

        self.logger.info("Source finding complete!")

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
            ).compute(num_workers=self.ncpu,
                      scheduler=self.settings['scheduler']
                      )
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

        if group.empty:
            return

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

        if '#' in group.columns:
            source_df = group.drop('#', axis=1)
        else:
            source_df = group

        source_df = source_df.sort_values('dateobs').reset_index(drop=True)

        self.logger.debug("Initialising Source with base folder:")
        self.logger.debug(source_base_folder)
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
            corrected_data=self.corrected_data
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
        bkg = rms.replace('noiseMap', 'meanMap')

        field = m['field']
        epoch = m['epoch']
        stokes = m['stokes']
        self.logger.debug("Getting Image for forced fits")
        try:
            img_beam = Image(
                field,
                epoch,
                stokes,
                self.base_folder,
                tiles=self.settings["tiles"],
                path=image,
                rmspath=rms,
                corrected_data=self.corrected_data
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

    def _get_selavy_meta(self) -> dict:
        """
        Obtains the correct metadata dictionary for use with
        Query._get_components

        Args:
            None
        Returns:
            The metadata dictionary
        """

        if self.settings["islands"]:
            meta = {
                'island_id': 'U',
                'island_name': 'U',
                'n_components': 'f',
                'ra_hms_cont': 'U',
                'dec_dms_cont': 'U',
                'ra_deg_cont': 'f',
                'dec_deg_cont': 'f',
                'freq': 'f',
                'maj_axis': 'f',
                'min_axis': 'f',
                'pos_ang': 'f',
                'flux_int': 'f',
                'flux_int_err': 'f',
                'flux_peak': 'f',
                'mean_background': 'f',
                'background_noise': 'f',
                'max_residual': 'f',
                'min_residual': 'f',
                'mean_residual': 'f',
                'rms_residual': 'f',
                'stdev_residual': 'f',
                'x_min': 'i',
                'x_max': 'i',
                'y_min': 'i',
                'y_max': 'i',
                'n_pix': 'i',
                'solid_angle': 'f',
                'beam_area': 'f',
                'x_ave': 'f',
                'y_ave': 'f',
                'x_cen': 'f',
                'y_cen': 'f',
                'x_peak': 'i',
                'y_peak': 'i',
                'flag_i1': 'i',
                'flag_i2': 'i',
                'flag_i3': 'i',
                'flag_i4': 'i',
                'comment': 'U',
                'detection': '?'
            }
        else:
            meta = {
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
            meta['#'] = 'f'
            meta['index'] = 'i'

        return meta

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
            self.logger.warning("Selavy file is None. Returning None.")
            return

        master = pd.DataFrame()

        selavy_df = read_selavy(selavy_file)
        self.logger.debug(f"Selavy df head: {selavy_df.head()}")

        if self.settings['stokes'] != "I":
            head, tail = os.path.split(selavy_file)
            nselavy_file = os.path.join(head, 'n{}'.format(tail))
            nselavy_df = read_selavy(nselavy_file)

            nselavy_df[["flux_peak", "flux_int"]] *= -1.0

            selavy_df = pd.concat(
                [selavy_df, nselavy_df],
                ignore_index=True,
                sort=False
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
            master = pd.concat([master, copy], sort=False)
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

            master = pd.concat([master, copy], sort=False)

            missing = group_coords[~mask]
            if missing.shape[0] > 0:
                if not self.settings['no_rms']:
                    try:
                        self.logger.debug(
                            "Initialising Image for components RMS estimates")
                        self.logger.debug(self.base_folder)
                        image = Image(
                            group.iloc[0].field,
                            group.iloc[0].epoch,
                            self.settings['stokes'],
                            self.base_folder,
                            sbid=group.iloc[0].sbid,
                            tiles=self.settings['tiles'],
                            corrected_data=self.corrected_data
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

                master = pd.concat([master, rms_df], sort=False)

        return master

    def _get_selavy_path(self, epoch_string: str, row: pd.Series) -> str:
        """
        Get the path to the selavy file for a specific row of the dataframe.
        Args:
            epoch_string: The name of the epoch in the form of 'EPOCHXX'.
            row: row: The input row of the dataframe.
        Returns:
            The path to the selavy file of interest
        """

        field = row.field.replace('RACS', 'VAST')

        if self.settings['islands']:
            cat_type = 'islands'
        else:
            cat_type = 'components'

        if self.settings['tiles']:
            dir_name = "TILES"

            data_folder = f"STOKES{self.settings['stokes']}_SELAVY"
            if self.corrected_data:
                data_folder += "_CORRECTED"

            selavy_folder = Path(
                self.base_folder,
                epoch_string,
                dir_name,
                data_folder
            )

            selavy_file_fmt = (
                "selavy-image.{}.{}.SB{}.cont."
                "taylor.0.restored.conv.{}.xml".format(
                    self.settings['stokes'].lower(), field, row.sbid, cat_type
                )
            )

            if self.corrected_data:
                selavy_file_fmt = selavy_file_fmt.replace(".xml",
                                                          ".corrected.xml"
                                                          )

            selavy_path = selavy_folder / selavy_file_fmt

            # Some epochs don't have .conv.
            if not selavy_path.is_file():
                self.logger.debug(f"{selavy_path} is not a file...")
                self.logger.debug(f"Removing '.conv' from filename")
                selavy_path = Path(str(selavy_path).replace('.conv', ''))

        else:
            dir_name = "COMBINED"
            selavy_folder = Path(
                self.base_folder,
                epoch_string,
                dir_name,
                f"STOKES{self.settings['stokes']}_SELAVY"
            )

            selavy_file_fmt = "selavy-{}.EPOCH{}.{}.conv.{}.xml".format(
                field,
                RELEASED_EPOCHS[row.epoch],
                self.settings['stokes'],
                cat_type
            )

            selavy_path = selavy_folder / selavy_file_fmt

        return str(selavy_path)

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

        img_dir = "STOKES{}_IMAGES".format(self.settings['stokes'])
        rms_dir = "STOKES{}_RMSMAPS".format(self.settings['stokes'])
        field = row.field.replace('RACS', 'VAST')

        if self.settings['tiles']:
            dir_name = "TILES"

            image_file_fmt = (
                "image.{}.{}.SB{}.cont"
                ".taylor.0.restored.fits".format(
                    self.settings['stokes'].lower(), field, row.sbid
                )
            )

            if self.corrected_data:
                img_dir += "_CORRECTED"
                rms_dir += "_CORRECTED"
                image_file_fmt = image_file_fmt.replace(".fits",
                                                        ".corrected.fits"
                                                        )
            rms_file_fmt = f"noiseMap.{image_file_fmt}"

        else:
            dir_name = "COMBINED"

            image_file_fmt = "{}.EPOCH{}.{}.conv.fits".format(
                field,
                RELEASED_EPOCHS[row.epoch],
                self.settings['stokes'],
            )

            rms_file_fmt = "noiseMap.{}.EPOCH{}.{}.conv.fits".format(
                field,
                RELEASED_EPOCHS[row.epoch],
                self.settings['stokes'],
            )

        selavy_file = self._get_selavy_path(epoch_string, row)

        image_file = Path(os.path.join(
            self.base_folder,
            epoch_string,
            dir_name,
            img_dir,
            image_file_fmt
        ))

        rms_file = Path(os.path.join(
            self.base_folder,
            epoch_string,
            dir_name,
            rms_dir,
            rms_file_fmt
        ))

        if not image_file.is_file():
            conv_image_file = Path(str(image_file).replace('.restored',
                                                           '.restored.conv')
                                   )
            if conv_image_file.is_file():
                image_file = conv_image_file
                rms_file = Path(str(rms_file).replace('.restored',
                                                      '.restored.conv')
                                )

        return selavy_file, str(image_file), str(rms_file)

    def _validate_files(self) -> None:
        """
        Check whether files in sources_df exist, and if not, remove them.

        Returns:
            None
        """

        missing_df = pd.DataFrame()
        missing_df['selavy'] = ~self.sources_df['selavy'].map(os.path.exists)
        missing_df['image'] = ~self.sources_df['image'].map(os.path.exists)
        missing_df['rms'] = ~self.sources_df['rms'].map(os.path.exists)

        missing_df['any'] = missing_df.any(axis=1)

        self.logger.debug(missing_df)

        for i, row in missing_df[missing_df['any']].iterrows():
            sources_row = self.sources_df.iloc[i]

            self.logger.warning(f"Removing {sources_row['name']}: Epoch "
                                f"{sources_row.epoch} due to missing files")
            if row.selavy:
                self.logger.debug(f"{sources_row.selavy} does not exist!")
            if row.image:
                self.logger.debug(f"{sources_row.image} does not exist!")
            if row.rms:
                self.logger.debug(f"{sources_row.rms} does not exist!")

        self.sources_df = self.sources_df[~missing_df['any']]

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

        if self.racs:
            base_fc = 'RACS'
        else:
            base_fc = 'VAST'

        self.logger.info(
            f"Matching queried sources to {base_fc} fields..."
        )

        base_epoch = BASE_EPOCHS[base_fc]

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
        if base_fc != 'VAST':
            field_centres['field'] = [
                f.replace("RACS", "VAST") for f in field_centres.field
            ]

        field_centre_names = field_centres.field

        if self.query_df is not None:
            self.fields_df = self.query_df.copy()

            # _field_matching returns 7 arguments. This dict specifies types,
            # O for object (in this case, lists) and U for unicode string.
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
                ).compute(num_workers=self.ncpu,
                          scheduler=self.settings['scheduler']
                          )
            )

            self.logger.debug("Finished field matching.")
            self.fields_df = self.fields_df.dropna()

            if self.fields_df.empty:
                raise Exception(
                    "No requested sources are within the requested footprint!")

            self.fields_df = self.fields_df.explode(
                'field_per_epoch'
            ).reset_index(drop=True)

            field_per_epoch = self.fields_df['field_per_epoch'].tolist()

            self.fields_df[
                ['epoch', 'field', 'sbid', 'dateobs', 'frequency']
            ] = pd.DataFrame(
                field_per_epoch,
                index=self.fields_df.index
            )

            to_drop = [
                'field_per_epoch',
                'epochs',
                'sbids',
                'dates',
                'freqs'
            ]
            self.logger.debug(self.fields_df['name'])
            self.logger.debug(self.fields_df['dateobs'])
            self.fields_df = self.fields_df.drop(
                labels=to_drop, axis=1
            ).sort_values(
                by=['name', 'dateobs']
            ).reset_index(drop=True)

            self.fields_df['planet'] = False
        else:
            self.fields_df = None

        # Handle Planets
        if len(self.planets) > 0:
            self.logger.debug(f"Searching for planets: {self.planets}")
            planet_fields = self._search_planets()

            if self.fields_df is None:
                self.fields_df = planet_fields
            else:
                self.fields_df = pd.concat(
                    [self.fields_df, planet_fields]
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
                f"{self.num_sources_searched}/{prev_num} "
                "sources in RACS & VAST footprint."
            )
        else:
            self.logger.info(
                f"{self.num_sources_searched}/{prev_num} "
                "sources in VAST footprint."
            )

        self.fields_df['dateobs'] = pd.to_datetime(
            self.fields_df['dateobs']
        )

        # All field names should start with VAST, not RACS
        self.fields_df['field'] = self.fields_df['field'].str.replace("RACS",
                                                                      "VAST"
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

        self.logger.debug("Running field matching with following row info:")
        self.logger.debug(row)
        self.logger.debug("Field names:")
        self.logger.debug(fields_names)

        seps = row.skycoord.separation(fields_coords)
        accept = seps.deg < self.settings['max_sep']
        fields = np.unique(fields_names[accept])

        if self.racs or self.vast_full:
            vast_fields = np.array(
                [f.replace("RACS", "VAST") for f in fields]
            )

        if fields.shape[0] == 0:
            self.logger.info(
                f"Source '{row['name']}' not in the requested epoch footprint."
            )
            return_vals = [np.nan] * 7  # Return nans in all 7 columns
            self.logger.debug(return_vals)
            return return_vals

        centre_seps = row.skycoord.separation(field_centres)
        primary_field = field_centre_names.iloc[np.argmin(centre_seps.deg)]
        self.logger.debug(f"Primary field: {primary_field}")
        epochs = []
        field_per_epochs = []
        sbids = []
        dateobs = []
        freqs = []

        for i in self.settings['epochs']:
            self.logger.debug(f"Epoch {i}")
            if i not in RACS_EPOCHS and self.racs:
                the_fields = vast_fields
            elif i not in RACS_EPOCHS and self.vast_full:
                the_fields = vast_fields
            else:
                the_fields = fields

            epoch_fields_names = self._epoch_fields.loc[i].index
            stripped = False
            if epoch_fields_names[0].endswith('A'):
                self.logger.debug("Using stripped field names")
                stripped = True
                epoch_fields_names = strip_fieldnames(epoch_fields_names)
            the_fields = list(set([f.rstrip('A') for f in the_fields]))

            self.logger.debug("Fields in epoch: ")
            self.logger.debug(epoch_fields_names)

            self.logger.debug("The fields: ")
            self.logger.debug(the_fields)

            available_fields = [
                f for f in the_fields if f in epoch_fields_names.to_list()
            ]
            self.logger.debug("Available fields:")
            self.logger.debug(available_fields)

            if i in RACS_EPOCHS:
                available_fields = [
                    j.replace("RACS", "VAST") for j in available_fields
                ]

            if len(available_fields) == 0:
                self.logger.debug("No fields available")
                continue

            if self.settings['search_all_fields']:
                selected_fields = available_fields

            elif primary_field in available_fields:
                selected_fields = [primary_field]
                self.logger.debug("Selecting primary field")

            elif len(available_fields) == 1:
                selected_fields = [available_fields[0]]
                self.logger.debug("Selecting only available field")

            else:
                field_indexes = [
                    field_centre_names[
                        field_centre_names == f.rstrip('A')
                    ].index[0] for f in available_fields
                ]
                min_field_index = np.argmin(
                    centre_seps[field_indexes].deg
                )

                selected_fields = [available_fields[min_field_index]]
                self.logger.debug("Selecting closest field")

            self.logger.debug(f"Selected fields: {selected_fields}")

            # Change VAST back to RACS
            if i in RACS_EPOCHS:
                selected_fields = [f.replace("VAST", "RACS")
                                   for f in selected_fields
                                   ]
            for field in selected_fields:
                if stripped:
                    field = f"{field}A"
                sbid_vals = self._epoch_fields.loc[i, field]["SBID"]
                date_vals = self._epoch_fields.loc[i, field]["DATEOBS"]
                freq_vals = self._epoch_fields.loc[i, field]["OBS_FREQ"]

                for sbid, date, freq in zip(sbid_vals, date_vals, freq_vals):
                    sbids.append(sbid)
                    dateobs.append(date)
                    freqs.append(freq)
                    epochs.append(i)
                    field_per_epochs.append([i, field, sbid, date, freq])

        return_vals = (fields,
                       primary_field,
                       epochs,
                       field_per_epochs,
                       sbids,
                       dateobs,
                       freqs
                       )
        # If len(available_fields) == 0 for all epochs need to return nan
        if len(epochs) == 0:
            return [np.nan] * 7  # Return nans in all 7 columns

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
        stripped_field_names = planet_epoch_fields.FIELD_NAME.str.rstrip('A')
        planet_epoch_fields['STRIPPED_FIELD_NAME'] = stripped_field_names

        planet_epoch_fields = planet_epoch_fields.merge(
            field_centres, left_on='STRIPPED_FIELD_NAME',
            right_on='field', how='left'
        ).drop(['field', 'OBS_FREQ', 'STRIPPED_FIELD_NAME'], axis=1).rename(
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
            ).compute(num_workers=self.ncpu,
                      scheduler=self.settings['scheduler']
                      )
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
        self.logger.debug("Building catalogue")

        cols = ['ra', 'dec', 'name', 'skycoord', 'stokes']

        if self.racs:
            self.logger.debug("Using RACS footprint for masking")
            mask = self.coords.dec.deg > 50

            if mask.any():
                self.logger.warning(
                    "Removing %i sources outside the RACS area", sum(mask)
                )
                self.coords = self.coords[~mask]
                self.source_names = self.source_names[~mask]
        else:
            mocs = VASTMOCS()

            pilot = self.vast_p1 or self.vast_p2

            if pilot:
                self.logger.debug("Using VAST pilot footprint for masking")
                footprint_moc = mocs.load_survey_footprint('pilot')

            if self.vast_full:
                self.logger.debug("Using full VAST footprint for masking")
                full_moc = mocs.load_survey_footprint('full')
                if pilot:
                    footprint_moc = footprint_moc.union(full_moc)
                else:
                    footprint_moc = full_moc

            self.logger.debug("Masking sources outside footprint")
            mask = footprint_moc.contains(
                self.coords.ra, self.coords.dec, keep_inside=False
            )
            if mask.any():
                self.logger.warning(
                    f"Removing {sum(mask)} sources outside the requested "
                    f"survey footprint."
                )
                self.coords = self.coords[~mask]
                self.source_names = self.source_names[~mask]

        self.logger.debug("Generating catalog dataframe")
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
            self.logger.debug("Handling SIMBAD naming")
            self.simbad_names = self.simbad_names[~mask]
            catalog['simbad_name'] = self.simbad_names

        return catalog

    def _get_epochs(self,
                    req_epochs: Union[str, List[str], List[int]]
                    ) -> List[str]:
        """
        Parse the list of epochs to query.

        Args:
            req_epochs: Requested epochs to query.

        Returns:
            Epochs to query, as a list of strings.

        Raises:
            QueryInitError: None of the requested epochs are available
        """

        epoch_dict = RELEASED_EPOCHS.copy()

        if self.settings['incl_observed']:
            epoch_dict.update(OBSERVED_EPOCHS)
        available_epochs = sorted(epoch_dict, key=epoch_dict.get)
        self.logger.debug("Available epochs: " + str(available_epochs))

        if req_epochs == 'all':
            epochs = available_epochs
        elif req_epochs == 'all-vast':
            epochs = available_epochs
            for racs_epoch in RACS_EPOCHS:
                if racs_epoch in epochs:
                    epochs.remove(racs_epoch)
        else:
            epochs = []
            if isinstance(req_epochs, list):
                epoch_iter = req_epochs
            elif isinstance(req_epochs, int):
                epoch_iter = [req_epochs]
            else:
                epoch_iter = req_epochs.split(',')

            for epoch in epoch_iter:
                if isinstance(epoch, int):
                    epoch = str(epoch)
                if epoch in available_epochs:
                    epochs.append(epoch)
                else:
                    epoch_x = f"{epoch}x"
                    self.logger.debug(
                        f"Epoch {epoch} is not available. Trying {epoch_x}"
                    )
                    if epoch_x in available_epochs:
                        epochs.append(epoch_x)
                        self.logger.debug(f"Epoch {epoch_x} available.")
                    else:
                        self.logger.info(
                            f"Epoch {epoch_x} is not available."
                        )

        # survey check
        self._check_survey(epochs)

        if self.racs:
            self.logger.warning('RACS data selected!')
            self.logger.warning(
                'Remember RACS data supplied by VAST is not final '
                'and results may vary.'
            )

        if len(epochs) == 0:
            raise QueryInitError(
                "None of the requested epochs are available"
            )

        return epochs

    def _check_survey(self, epochs: list) -> None:
        """
        Check which surveys are being queried (e.g. RACS, pilot/full VAST).

        Args:
            epochs: Requested epochs to query
        """

        self.racs = False
        self.vast_p1 = False
        self.vast_p2 = False
        self.vast_full = False

        non_full_epochs = RACS_EPOCHS + P1_EPOCHS + P2_EPOCHS
        all_epochs = RELEASED_EPOCHS.keys()
        full_epochs = set(all_epochs) - set(non_full_epochs)

        epochs_set = set(epochs)
        if len(epochs_set & set(RACS_EPOCHS)) > 0:
            self.racs = True
        if len(epochs_set & set(P1_EPOCHS)) > 0:
            self.vast_p1 = True
        if len(epochs_set & set(P2_EPOCHS)) > 0:
            self.vast_p2 = True
        if len(epochs_set & set(full_epochs)) > 0:
            self.vast_full = True

        self.logger.debug(f"self.racs: {self.racs}")
        self.logger.debug(f"self.vast_p1: {self.vast_p1}")
        self.logger.debug(f"self.vast_p2: {self.vast_p2}")
        self.logger.debug(f"self.vast_full: {self.vast_full}")

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

        We check against epochs 1 and 18 which are the first complete
        low- and mid-band epochs respectively.

        Returns:
            Bool representing if field is valid.
        """

        epoch_01 = load_fields_file("1")
        epoch_18 = load_fields_file("18")
        base_fields = pd.concat(epoch_01, epoch_18)
        self.logger.debug("Field name: {}".format(self.field))
        result = base_fields['FIELD_NAME'].str.contains(
            re.escape(self.field)
        ).any()
        self.logger.debug("Field found: {}".format(result))
        if result is False:
            self.logger.error(
                "Field {} is not a valid field name!".format(self.field)
            )
        del epoch_01, epoch_18, base_fields
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
            epochs_dict = RELEASED_EPOCHS.copy()
            epochs_dict.update(OBSERVED_EPOCHS)
            for i, val in enumerate(sorted(epochs_dict)):
                if i == 0:
                    self.pilot_info = load_fields_file(val)
                    self.pilot_info["EPOCH"] = epochs_dict[val]
                else:
                    to_append = load_fields_file(val)
                    to_append["EPOCH"] = epochs_dict[val]
                    self.pilot_info = pd.concat(
                        [self.pilot_info, to_append],
                        sort=False
                    )

        self.field_info = self.pilot_info[
            self.pilot_info.FIELD_NAME == self.field
        ]

        self.field_info.reset_index(drop=True, inplace=True)

        self.field_info = self.field_info.filter([
            "EPOCH",
            "FIELD_NAME",
            "SBID",
            "OBS_FREQ",
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
