"""Class to describe a VAST astrophysical source.
"""
# Source class

import gc
import logging.config
import logging.handlers
import logging
import matplotlib
import matplotlib.axes as maxes
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import signal
import warnings

from astropy.visualization import LinearStretch
from astropy.visualization import AsymmetricPercentileInterval
from astropy.visualization import PercentileInterval
from astropy.visualization import ZScaleInterval, ImageNormalize
from mpl_toolkits.axes_grid1.anchored_artists import (AnchoredEllipse,
                                                      AnchoredSizeBar)
from astropy.coordinates import Angle
from astropy.visualization.wcsaxes import SphericalCircle
from matplotlib.collections import PatchCollection
from astropy.wcs.utils import proj_plane_pixel_scales
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from astropy.utils.exceptions import AstropyWarning, AstropyDeprecationWarning
from astropy.wcs.utils import skycoord_to_pixel
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.nddata.utils import Cutout2D
from astropy import units as u
from astropy.time import Time
from astropy.timeseries import TimeSeries
from astropy.table import Table
from astroquery.simbad import Simbad
from astroquery.ned import Ned
from astroquery.casda import Casda
from astropy.stats import sigma_clipped_stats
from astroquery.skyview import SkyView
from astropy.wcs import WCS

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import make_axes_locatable

from typing import List, Tuple, Optional, Union

from radio_beam import Beam

from vasttools import RELEASED_EPOCHS
from vasttools.utils import crosshair
from vasttools.survey import Image
from vasttools.utils import filter_selavy_components
from vasttools.tools import offset_postagestamp_axes

# run crosshair to set up the marker.
crosshair()
# Switch matplotlib backend.
matplotlib.pyplot.switch_backend('Agg')


class SourcePlottingError(Exception):
    """
    A custom exception for plotting errors.
    """
    pass


class Source:
    """
    This is a class representation of a catalogued source position

    Attributes:
        pipeline (bool): Set to `True` if the source is generated from a VAST
            Pipeline run.
        coord (astropy.coordinates.SkyCoord):
            The coordinate of the source as a SkyCoord object.
            Planets can sometimes have a SkyCoord containing more than
            one coordinate.
        name (str): The name of the source.
        epochs (List[str]): The epochs the source contains.
        fields (List[str]): The fields the source contains.
        stokes (str): The Stokes parameter of the source.
        crossmatch_radius (astropy.coordinates.Angle):
            Angle of the crossmatch. This will not be valid for
            pipeline sources.
        measurements (pandas.core.frame.DataFrame):
            The individual measurements of the source.
        islands (bool): Set to `True` if islands have been used for the
            source creation.
        outdir (str): Path that will be appended to any files that are saved.
        base_folder (str): The directory where the data (fits files) is held.
        image_type (str): 'TILES' or 'COMBINED'.
        tiles (bool): `True` if `image_type` == `TILES`.
        detections (int): The number of selavy detections the source contains.
        limits (int):
            The number of upper limits the source contains. Will be set to
            `None` for pipeline sources.
        forced_fits (int): The number of forced fits the source contains.
        norms (astropy.visualization.ImageNormalize):
            Contains the normalization value to use for consistent
            normalization across the measurements for png representation.
        planet (bool): Set to `True` if the source has been defined
            as a planet.
    """

    def __init__(
        self,
        coord: SkyCoord,
        name: str,
        epochs: List[str],
        fields: List[str],
        stokes: str,
        primary_field: str,
        crossmatch_radius: Angle,
        measurements: pd.DataFrame,
        base_folder: str,
        image_type: str = "COMBINED",
        islands: bool = False,
        outdir: str = ".",
        planet: bool = False,
        pipeline: bool = False,
        tiles: bool = False,
        forced_fits: bool = False,
    ) -> None:
        """
        Constructor method

        Args:
            coord: Source coordinates.
            name: The name of the source.
            epochs: The epochs that the source contains.
            fields: The fields that the source contains.
            stokes: The stokes parameter of the source.
            primary_field: The primary VAST Pilot field of the source.
            crossmatch_radius: The crossmatch radius used to find the
                measurements.
            measurements: DataFrame containing the measurements.
            base_folder: Path to base folder in default directory structure
            image_type: The string representation of the image type,
                either 'COMBINED' or 'TILES', defaults to "COMBINED".
            islands: Is `True` if islands has been used instead of
                components, defaults to `False`.
            outdir: The directory where any media outputs will be written
                to, defaults to ".".
            planet: Set to `True` if the source is a planet, defaults
                to `False`.
            pipeline: Set to `True` if the source has been loaded from a
                VAST Pipeline run, defaults to `False`.
            tiles: Set to 'True` if the source is from a tile images,
                defaults to `False`.
            forced_fits: Set to `True` if forced fits are included in the
                source measurements, defaults to `False`.

        Returns:
            None
        """
        self.logger = logging.getLogger('vasttools.source.Source')
        self.logger.debug('Created Source instance')
        self.pipeline = pipeline
        self.coord = coord
        self.name = name
        self.epochs = epochs
        self.fields = fields
        self.stokes = stokes
        self.primary_field = primary_field
        self.crossmatch_radius = crossmatch_radius
        self.measurements = measurements.infer_objects()
        self.measurements.dateobs = pd.to_datetime(
            self.measurements.dateobs
        )
        self.islands = islands
        if self.islands:
            self.cat_type = 'islands'
        else:
            self.cat_type = 'components'

        self.outdir = outdir

        self.base_folder = base_folder
        self.image_type = image_type
        if image_type == 'TILES':
            self.tiles = True
        else:
            self.tiles = False

        if self.pipeline:
            self.detections = self.measurements[
                self.measurements.forced == False
            ].shape[0]

            self.forced = self.measurements[
                self.measurements.forced == False
            ].shape[0]

            self.limits = None
            self.forced_fits = False
        else:
            self.detections = self.measurements[
                self.measurements.detection
            ].shape[0]

            self.limits = self.measurements[
                self.measurements.detection == False
            ].shape[0]

            self.forced = None
            self.forced_fits = forced_fits

        self._cutouts_got = False

        self.norms = None
        self._checked_norms = False

        self.planet = planet

    def write_measurements(
        self, simple: bool = False, outfile: Optional[str] = None
    ) -> None:
        """Write the measurements to a CSV file.

        Args:
            simple: Only include flux density and uncertainty in returned
                table, defaults to `False`.
            outfile: File to write measurements to, defaults to None.

        Returns:
            None
        """
        if simple:
            if self.pipeline:
                cols = [
                    'source',
                    'ra',
                    'dec',
                    'component_id',
                    'flux_peak',
                    'flux_peak_err',
                    'flux_int',
                    'flux_int_err',
                    'rms',
                ]
            else:
                cols = [
                    'name',
                    'ra_deg_cont',
                    'dec_deg_cont',
                    'component_id',
                    'flux_peak',
                    'flux_peak_err',
                    'flux_int',
                    'flux_int_err',
                    'rms_image',
                ]

            measurements_to_write = self.measurements[cols]

        else:
            cols = [
                'fields',
                'skycoord',
                'selavy',
                'image',
                'rms',
            ]

            if self.pipeline:
                cols[0] = 'field'

            measurements_to_write = self.measurements.drop(
                labels=cols, axis=1
            )

        # drop any empty values
        if not self.pipeline and not self.forced_fits:
            measurements_to_write = measurements_to_write[
                measurements_to_write['rms_image'] != -99
            ]

        if measurements_to_write.empty:
            self.logger.warning(
                "%s has no measurements! No file will be written.",
                self.name
            )
            return

        if outfile is None:
            outfile = "{}_measurements.csv".format(self.name.replace(
                " ", "_"
            ).replace(
                "/", "_"
            ))

        elif not outfile.endswith(".csv"):
            outfile += ".csv"

        if self.outdir != ".":
            outfile = os.path.join(
                self.outdir,
                outfile
            )

        measurements_to_write.to_csv(outfile, index=False)

        self.logger.debug("Wrote {}.".format(outfile))

    def plot_lightcurve(
        self,
        sigma_thresh: int = 5,
        figsize: Tuple[int, int] = (8, 4),
        min_points: int = 2,
        min_detections: int = 0,
        mjd: bool = False,
        # TODO: Is this a pd.Timestamp or a datetime?
        start_date: Optional[pd.Timestamp] = None,
        grid: bool = False,
        yaxis_start: str = "0",
        peak_flux: bool = True,
        save: bool = False,
        outfile: Optional[str] = None,
        use_forced_for_limits: bool = False,
        use_forced_for_all: bool = False,
        hide_legend: bool = False,
        plot_dpi: int = 150
    ) -> Union[None, plt.Figure]:
        """
        Plot source lightcurves and save to file

        Args:
            sigma_thresh: Threshold to use for upper limits, defaults to 5.
            figsize: Figure size, defaults to (8, 4).
            min_points: Minimum number of points for plotting, defaults
                to 2.
            min_detections:  Minimum number of detections for plotting,
                defaults to 0.
            mjd: Plot x-axis in MJD rather than datetime, defaults to False.
            start_date: Plot in days from start date, defaults to None.
            grid: Turn on matplotlib grid, defaults to False.
            yaxis_start: Define where the y-axis begins from, either 'auto'
                or '0', defaults to "0".
            peak_flux: Uses peak flux instead of integrated flux,
                defaults to `True`.
            save: When `True` the plot is saved rather than displayed,
                defaults to `False`.
            outfile: The filename to save when using, defaults to None which
                will use '<souce_name>_lc.png'.
            use_forced_for_limits: Use the forced extractions instead of
                upper limits for non-detections., defaults to `False`.
            use_forced_for_all: Use the forced fits for all the datapoints,
                defaults to `False`.
            hide_legend: Hide the legend, defaults to `False`.
            plot_dpi: Specify the DPI of saved figures, defaults to 150.

        Returns:
            None if save is `True` or the matplotlib figure if save is
            `False`.

        Raises:
            SourcePlottingError: Source does not have any forced fits when the
                'use_forced_for_all' or 'use_forced_for_limits' options have
                been selected.
            SourcePlottingError: Number of detections lower than the
                minimum required.
            SourcePlottingError: Number of datapoints lower than the
                minimum required.
            SourcePlottingError: If measurements dataframe is empty.
        """
        if use_forced_for_all or use_forced_for_limits:
            if not self.forced_fits:
                raise SourcePlottingError(
                    "Source does not have any forced fits points to plot."
                )

        if self.detections < min_detections:
            msg = (
                f"Number of detections ({self.detections}) lower "
                f"than minimum required ({min_detections})"
            )
            self.logger.error(msg)
            raise SourcePlottingError(msg)

        if self.measurements.shape[0] < min_points:
            msg = (
                f"Number of datapoints ({self.measurements.shape[0]}) lower "
                f"than minimum required ({min_points})"
            )
            self.logger.error(msg)
            raise SourcePlottingError(msg)

        if mjd and start_date is not None:
            msg = (
                "The 'mjd' and 'start date' options "
                "cannot be used at the same time!"
            )
            self.logger.error(msg)
            raise SourcePlottingError(msg)

        # remove empty values
        measurements = self.measurements
        if not self.pipeline and not (
            use_forced_for_limits or use_forced_for_all
        ):
            measurements = self.measurements[
                self.measurements['rms_image'] != -99
            ]

        if measurements.empty:
            msg = f"{self.name} has no measurements!"
            self.logger.error(msg)
            raise SourcePlottingError(msg)

        plot_dates = measurements['dateobs']
        if mjd:
            plot_dates = Time(plot_dates.to_numpy()).mjd
        elif start_date:
            plot_dates = (plot_dates-start_date)/pd.Timedelta(1, unit='d')

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        plot_title = self.name
        if self.islands:
            plot_title += " (island)"
        ax.set_title(plot_title)

        if peak_flux:
            label = 'Peak Flux Density (mJy/beam)'
            flux_col = "flux_peak"
        else:
            label = 'Integrated Flux Density (mJy)'
            flux_col = "flux_int"

        if use_forced_for_all:
            label = "Forced " + label
            flux_col = "f_" + flux_col

        if self.stokes != "I":
            label = "Absolute " + label
            measurements[flux_col] = measurements[flux_col].abs()

        ax.set_ylabel(label)

        self.logger.debug("Plotting upper limit")
        if self.pipeline:
            upper_lim_mask = measurements.forced
        else:
            upper_lim_mask = measurements.detection == False
            if use_forced_for_all:
                upper_lim_mask = np.array([False for i in upper_lim_mask])
        upper_lims = measurements[
            upper_lim_mask
        ]
        if self.pipeline:
            if peak_flux:
                value_col = 'flux_peak'
                err_value_col = 'flux_peak_err'
            else:
                value_col = 'flux_int'
                err_value_col = 'flux_int_err'
            marker = "D"
            uplims = False
            sigma_thresh = 1.0
            label = 'Forced'
            markerfacecolor = 'w'
        else:
            if use_forced_for_limits:
                value_col = 'f_flux_peak'
                err_value_col = 'f_flux_peak_err'
                uplims = False
                marker = "D"
                sigma_thresh = 1.0
                markerfacecolor = 'w'
                label = "Forced"
            else:
                value_col = err_value_col = 'rms_image'
                marker = "_"
                uplims = True
                markerfacecolor = 'k'
                label = 'Upper limit'
        if upper_lim_mask.any():
            upperlim_points = ax.errorbar(
                plot_dates[upper_lim_mask],
                sigma_thresh *
                upper_lims[value_col],
                yerr=upper_lims[err_value_col],
                uplims=uplims,
                lolims=False,
                marker=marker,
                c='k',
                linestyle="none",
                markerfacecolor=markerfacecolor,
                label=label
            )

        self.logger.debug("Plotting detection")

        if use_forced_for_all:
            detections = measurements
        else:
            detections = measurements[
                ~upper_lim_mask
            ]

        if self.pipeline:
            if peak_flux:
                err_value_col = 'flux_peak_err'
            else:
                err_value_col = 'flux_int_err'
        else:
            if use_forced_for_all:
                err_value_col = flux_col + '_err'
            else:
                err_value_col = 'rms_image'

        if use_forced_for_all:
            marker = "D"
            markerfacecolor = 'w'
            label = 'Forced'
        else:
            marker = 'o'
            markerfacecolor = 'k'
            label = 'Selavy'
        if (~upper_lim_mask).any():
            detection_points = ax.errorbar(
                plot_dates[~upper_lim_mask],
                detections[flux_col],
                yerr=detections[err_value_col],
                marker=marker,
                c='k',
                linestyle="none",
                markerfacecolor=markerfacecolor,
                label=label)

        if yaxis_start == "0":
            max_det = detections.loc[:, [flux_col, err_value_col]].sum(axis=1)
            if use_forced_for_limits or self.pipeline:
                max_y = np.nanmax(
                    max_det.tolist() +
                    upper_lims[value_col].tolist()
                )
            elif use_forced_for_all:
                max_y = np.nanmax(max_det.tolist())
            else:
                max_y = np.nanmax(
                    max_det.tolist() +
                    (sigma_thresh * upper_lims[err_value_col]).tolist()
                )
            ax.set_ylim(
                bottom=0,
                top=max_y * 1.1
            )

        if mjd:
            ax.set_xlabel('Date (MJD)')
        elif start_date:
            ax.set_xlabel('Days since {}'.format(start_date))
        else:
            fig.autofmt_xdate()
            ax.set_xlabel('Date')

            date_form = mdates.DateFormatter("%Y-%m-%d")
            ax.xaxis.set_major_formatter(date_form)
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=15))

        ax.grid(grid)

        if not hide_legend:
            ax.legend()

        if save:
            if outfile is None:
                outfile = "{}_lc.png".format(self.name.replace(
                    " ", "_"
                ).replace(
                    "/", "_"
                ))

            elif not outfile.endswith(".png"):
                outfile += ".png"

            if self.outdir != ".":
                outfile = os.path.join(
                    self.outdir,
                    outfile
                )

            plt.savefig(outfile, bbox_inches='tight', dpi=plot_dpi)
            plt.close()

            return

        else:

            return fig

    def get_cutout_data(self, size: Optional[Angle] = None) -> None:
        """
        Function to fetch the cutout data for that source
        required for producing all the media output.

        If size is not provided then the default size of 5 arcmin will be
        used.

        Args:
            size: The angular size of the cutouts, defaults to None.

        Returns:
            None
        """
        if size is None:
            args = None
        else:
            args = (size,)

        self.cutout_df = self.measurements.apply(
            self._get_cutout,
            args=args,
            axis=1,
            result_type='expand'
        ).rename(columns={
            0: "data",
            1: "wcs",
            2: "header",
            3: "selavy_overlay",
            4: "beam"
        })
        self._cutouts_got = True

    def _analyse_norm_level(
        self,
        percentile: float = 99.9,
        zscale: bool = False,
        z_contrast: float = 0.2,
        cutout_data: Optional[pd.DataFrame] = None,
        return_norm: bool = False
    ) -> Union[None, ImageNormalize]:
        """
        Selects the appropriate image to use as the normalization
        value for each image.

        Either the first `detection` image is used, or the first image
        in time if there are no detections.

        Args:
            percentile: The value passed to the percentile
                normalization function, defaults to 99.9.
            zscale: Uses ZScale normalization instead of PercentileInterval,
                defaults to `False`
            z_contrast: Contast value passed to the ZScaleInterval
                function when zscale is selected, defaults to 0.2.
            cutout_data: Pass external cutout_data to be used
                instead of fetching the data, defaults to None.
            return_norm: If `True` the calculated norm is returned
                by the function, defaults to False.

        Returns:
            None if return_norm is `False` or the normalization if `True`.

        Raises:
            ValueError: If the cutout data is yet to be obtained.
        """
        if cutout_data is None:
            if not self._cutouts_got:
                self.logger.warning(
                    "Fetch cutout data before running this function!"
                )

                raise ValueError(
                    "Fetch cutout data before running this function!"
                )
            else:
                cutout_data = self.cutout_df

        if self.detections > 0:
            scale_index = self.measurements[
                self.measurements.detection
            ].index.values[0]
        else:
            scale_index = 0

        scale_data = cutout_data.loc[scale_index].data * 1.e3

        if zscale:
            norms = ImageNormalize(
                scale_data, interval=ZScaleInterval(
                    contrast=z_contrast))
        else:
            norms = ImageNormalize(
                scale_data,
                interval=PercentileInterval(percentile),
                stretch=LinearStretch())

        if return_norm:
            return norms
        else:
            self.norms = norms

        self._checked_norms = True

    def _get_cutout(
        self, row: pd.Series, size: Angle = Angle(5. * u.arcmin)
    ) -> Tuple[np.ndarray, WCS, fits.Header, pd.DataFrame, Beam]:
        """Does the actual fetching of the cutout data.

        Args:
            row: The row in the measurements df for which
                media will be fetched.
            size: The size of the cutout, defaults to Angle(5.*u.arcmin)

        Returns:
            Tuple containing the cutout data.
        """
        if self.pipeline:
            image = Image(
                row.field, row.epoch, self.stokes, self.base_folder,
                path=row.image, rmspath=row.rms
            )
            image.get_img_data()
        else:
            e = row.epoch
            if "-" in e:
                e = e.split("-")[0]
            image = Image(
                row.field, e, self.stokes,
                self.base_folder, tiles=self.tiles,
                sbid=row.sbid
            )
            image.get_img_data()

        cutout = Cutout2D(
            image.data,
            position=row.skycoord,
            size=size,
            wcs=image.wcs
        )

        if self.pipeline:
            selavy_components = pd.read_parquet(
                row.selavy,
                columns=[
                    'island_id',
                    'ra',
                    'dec',
                    'bmaj',
                    'bmin',
                    'pa'
                ]
            ).rename(
                columns={
                    'ra': 'ra_deg_cont',
                    'dec': 'dec_deg_cont',
                    'bmaj': 'maj_axis',
                    'bmin': 'min_axis',
                    'pa': 'pos_ang'
                }
            )
        else:
            selavy_components = pd.read_fwf(
                row.selavy, skiprows=[1, ], usecols=[
                    'island_id',
                    'ra_deg_cont',
                    'dec_deg_cont',
                    'maj_axis',
                    'min_axis',
                    'pos_ang'
                ]
            )

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

        self._size = size

        del image
        del selavy_coords

        return (
            cutout.data, cutout.wcs, header, selavy_components, beam
        )

    def show_png_cutout(
        self,
        epoch: str,
        selavy: bool = True,
        percentile: float = 99.9,
        zscale: bool = False,
        contrast: float = 0.2,
        no_islands: bool = True,
        label: str = "Source",
        no_colorbar: bool = False,
        title: Optional[str] = None,
        crossmatch_overlay: bool = False,
        hide_beam: bool = False,
        size: Optional[Angle] = None,
        force: bool = False,
        offset_axes: bool = True,
    ) -> plt.Figure:
        """
        Wrapper for make_png to make nicer interactive function.
        No access to save.

        Args:
            epoch: The epoch to show.
            selavy: If `True` then selavy overlay are shown,
                 defaults to `True`.
            percentile: The value passed to the percentile
                normalization function, defaults to 99.9.
            zscale: Uses ZScale normalization instead of
                PercentileInterval, defaults to `False`.
            contrast: Contrast value passed to the ZScaleInterval
                function when zscale is selected, defaults to 0.2.
            no_islands: Hide island name labels, defaults to `True`.
            label: legend label for source, defaults to "Source".
            no_colorbar: Hides the colorbar, defaults to `False`.
            title: Sets the plot title, defaults to None.
            crossmatch_overlay: Plots a circle that represents the
                crossmatch radius, defaults to `False`.
            hide_beam: Hide the beam on the plot, defaults to `False`.
            size: Size of the cutout, defaults to None.
            force: Whether to force the re-fetching of the cutout data,
                defaults to `False`.
            offset_axes: Use offset, rather than absolute, axis labels.

        Returns:
            Figure object.
        """

        fig = self.make_png(
            epoch,
            selavy=selavy,
            percentile=percentile,
            zscale=zscale,
            contrast=contrast,
            no_islands=no_islands,
            label=label,
            no_colorbar=no_colorbar,
            title=title,
            crossmatch_overlay=crossmatch_overlay,
            hide_beam=hide_beam,
            size=size,
            force=force
        )

        return fig

    def save_png_cutout(
        self,
        epoch: str,
        selavy: bool = True,
        percentile: float = 99.9,
        zscale: bool = False,
        contrast: float = 0.2,
        no_islands: bool = True,
        label: str = "Source",
        no_colorbar: bool = False,
        title: Optional[str] = None,
        crossmatch_overlay: bool = False,
        hide_beam: bool = False,
        size: Optional[Angle] = None,
        force: bool = False,
        outfile: Optional[str] = None,
        plot_dpi: int = 150,
        offset_axes: bool = True
    ) -> None:
        """
        Wrapper for make_png to make nicer interactive function.
        Always save.

        Args:
            epoch: The epoch to show.
            selavy: If `True` then selavy overlay are shown,
                 defaults to `True`.
            percentile: The value passed to the percentile
                normalization function, defaults to 99.9.
            zscale: Uses ZScale normalization instead of
                PercentileInterval, defaults to `False`.
            contrast: Contrast value passed to the ZScaleInterval
                function when zscale is selected, defaults to 0.2.
            no_islands: Hide island name labels, defaults to `True`.
            label: legend label for source, defaults to "Source".
            no_colorbar: Hides the colorbar, defaults to `False`.
            title: Sets the plot title, defaults to None.
            crossmatch_overlay: Plots a circle that represents the
                crossmatch radius, defaults to `False`.
            hide_beam: Hide the beam on the plot, defaults to `False`.
            size: Size of the cutout, defaults to None.
            force: Whether to force the re-fetching
                of the cutout data, defaults to `False`.
            outfile: Name to give the file, if None then the name is
                automatically generated, defaults to None.
            plot_dpi: Specify the DPI of saved figures, defaults to 150.
            offset_axes: Use offset, rather than absolute, axis labels.

        Returns:
            None
        """
        fig = self.make_png(
            epoch,
            selavy=selavy,
            percentile=percentile,
            zscale=zscale,
            contrast=contrast,
            no_islands=no_islands,
            label=label,
            no_colorbar=no_colorbar,
            title=title,
            crossmatch_overlay=crossmatch_overlay,
            hide_beam=hide_beam,
            size=size,
            force=force,
            outfile=outfile,
            save=True,
            plot_dpi=plot_dpi
        )

        return

    def _get_save_name(self, epoch: str, ext: str) -> str:
        """
        Generate name of file to save to.

        Args:
            epoch: Epoch corresponding to requested data
            ext: File extension

        Returns:
            Name of file to save.
        """

        if self.pipeline:
            name_epoch = epoch
        else:
            if "-" in epoch:
                e_split = epoch.split("-")
                e = e_split[0]
                name_epoch = RELEASED_EPOCHS[e] + "-" + e_split[1]
            else:
                name_epoch = RELEASED_EPOCHS[epoch]
        outfile = "{}_EPOCH{}{}".format(
            self.name.replace(" ", "_").replace(
                "/", "_"
            ),
            name_epoch,
            ext
        )
        return outfile

    def save_fits_cutout(
        self,
        epoch: str,
        outfile: Optional[str] = None,
        size: Optional[Angle] = None,
        force: bool = False,
        cutout_data: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Saves the FITS file cutout of the requested epoch.

        Args:
            epoch: Requested epoch.
            outfile: File to save to, defaults to None.
            size: Size of the cutout, defaults to None.
            force: Whether to force the re-fetching
                of the cutout data, defaults to `False`.
            cutout_data: Pass external cutout_data to be used
                instead of fetching the data, defaults to None.

        Returns:
            None

        Raises:
            ValueError: If the source does not contain the requested epoch.
        """

        if (self._cutouts_got is False) or (force):
            if cutout_data is None:
                self.get_cutout_data(size)

        if epoch not in self.epochs:
            raise ValueError(
                "This source does not contain Epoch {}!".format(epoch)
            )

            return

        if outfile is None:
            outfile = self._get_save_name(epoch, ".fits")
        if self.outdir != ".":
            outfile = os.path.join(
                self.outdir,
                outfile
            )

        index = self.epochs.index(epoch)

        if cutout_data is None:
            cutout_row = self.cutout_df.iloc[index]
        else:
            cutout_row = cutout_data.iloc[index]

        hdu_stamp = fits.PrimaryHDU(
            data=cutout_row.data,
            header=cutout_row.header
        )

        # Write the cutout to a new FITS file
        hdu_stamp.writeto(outfile, overwrite=True)

        del hdu_stamp

    def save_all_ann(
        self,
        crossmatch_overlay: bool = False,
        cutout_data: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Save kvis annotation file corresponding to the source

        Args:
            crossmatch_overlay: Include the crossmatch radius,
                defaults to `False`
            cutout_data: Pass external cutout_data to be used
                instead of fetching the data, defaults to None.

        Returns:
            None
        """
        self.measurements['epoch'].apply(
            self.write_ann,
            args=(
                None,
                crossmatch_overlay,
                None,
                False,
                cutout_data
            )
        )

    def save_all_reg(
        self,
        crossmatch_overlay: bool = False,
        cutout_data: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Save DS9 region file corresponding to the source

        Args:
            crossmatch_overlay: Include the crossmatch radius,
                defaults to `False`
            cutout_data: Pass external cutout_data to be used
                instead of fetching the data, defaults to None.

        Returns:
            None
        """

        self.measurements['epoch'].apply(
            self.write_reg,
            args=(
                None,
                crossmatch_overlay,
                None,
                False,
                cutout_data
            )
        )

    def save_all_fits_cutouts(
        self,
        size: Optional[Angle] = None,
        force: bool = False,
        cutout_data: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Save all cutouts of the source to fits file

        Args:
            size: Size of the cutouts, defaults to None.
            force: Whether to force the re-fetching
                of the cutout data, defaults to `False`.
            cutout_data: Pass external cutout_data to be used
                instead of fetching the data, defaults to None.

        Returns:
            None
        """
        if (self._cutouts_got is False) or (force):
            if cutout_data is None:
                self.get_cutout_data(size)

        for e in self.measurements['epoch']:
            self.save_fits_cutout(e, cutout_data=cutout_data)

    def save_all_png_cutouts(
        self,
        selavy: bool = True,
        percentile: float = 99.9,
        zscale: bool = False,
        contrast: float = 0.2,
        no_islands: bool = True,
        no_colorbar: bool = False,
        crossmatch_overlay: bool = False,
        hide_beam: bool = False,
        size: Optional[Angle] = None,
        disable_autoscaling: bool = False,
        cutout_data: Optional[pd.DataFrame] = None,
        calc_script_norms: bool = False,
        plot_dpi: int = 150,
        offset_axes: bool = True
    ) -> None:
        """
        Wrapper function to save all the png cutouts
        for all epochs.

        Args:
            selavy: If `True` then selavy overlay are shown,
                 defaults to `True`.
            percentile: The value passed to the percentile
                normalization function, defaults to 99.9.
            zscale: Uses ZScale normalization instead of
                PercentileInterval, defaults to `False`.
            contrast: Contrast value passed to the ZScaleInterval
                function when zscale is selected, defaults to 0.2.
            no_islands: Hide island name labels, defaults to `True`.
            no_colorbar: Hides the colorbar, defaults to `False`.
            crossmatch_overlay: Plots a circle that represents the
                crossmatch radius, defaults to `False`.
            hide_beam: Hide the beam on the plot, defaults to `False`.
            size: Size of the cutout, defaults to None.
            disable_autoscaling: Do not use the consistent normalization
                values but calculate norms separately for each epoch,
                defaults to `False`.
            cutout_data: Pass external cutout_data to be used
                instead of fetching the data, defaults to None.
            calc_script_norms: When passing cutout data this parameter
                can be set to True to pass this cutout data to the analyse
                norms function, defaults to False.
            plot_dpi: Specify the DPI of saved figures, defaults to 150.
            offset_axes: Use offset, rather than absolute, axis labels.

        Returns:
            None
        """
        if self._cutouts_got is False:
            if cutout_data is None:
                self.get_cutout_data(size)

        if not calc_script_norms:
            if not self._checked_norms:
                self._analyse_norm_level(
                    percentile=percentile,
                    zscale=zscale,
                    z_contrast=contrast
                )
            norms = None
        else:
            norms = self._analyse_norm_level(
                percentile=percentile,
                zscale=zscale,
                z_contrast=contrast,
                cutout_data=cutout_data
            )

        self.measurements['epoch'].apply(
            self.make_png,
            args=(
                selavy,
                percentile,
                zscale,
                contrast,
                None,
                no_islands,
                "Source",
                no_colorbar,
                None,
                crossmatch_overlay,
                hide_beam,
                True,
                None,
                False,
                disable_autoscaling,
                cutout_data,
                norms,
                plot_dpi
            )
        )

    def show_all_png_cutouts(
        self,
        columns: int = 4,
        percentile: float = 99.9,
        zscale: bool = False,
        contrast: float = 0.1,
        outfile: Optional[str] = None,
        save: bool = False,
        size: Optional[Angle] = None,
        figsize: Tuple[int, int] = (10, 5),
        force: bool = False,
        no_selavy: bool = False,
        disable_autoscaling: bool = False,
        hide_epoch_labels: bool = False,
        plot_dpi: int = 150,
        offset_axes: bool = True
    ) -> Union[None, plt.figure]:
        """
        Creates a grid plot showing the source in each epoch.

        Args:
            columns: Number of columns to use for the grid plot,
                defaults to 4.
            percentile: The value passed to the percentile
                normalization function, defaults to 99.9.
            zscale: Uses ZScale normalization instead of
                PercentileInterval, defaults to `False`.
            contrast: Contast value passed to the ZScaleInterval
                function when zscale is selected, defaults to 0.2.
            outfile: Name of the output file, if None then the name
                 is automatically generated, defaults to None.
            save: Save the plot instead of displaying,
                defaults to `False`.
            size: Size of the cutout, defaults to None.
            figsize: Size of the matplotlib.pyplot figure,
                defaults to (10, 5).
            force: Whether to force the re-fetching
                of the cutout data, defaults to `False`.
            no_selavy: When `True` the selavy overlay
                is hidden, defaults to `False`.
            disable_autoscaling: Turn off the consistent normalization and
                 calculate the normalizations separately for each epoch,
                defaults to `False`.
            hide_epoch_labels: Turn off the epoch number label (found in
                top left corner of image).
            plot_dpi: Specify the DPI of saved figures, defaults to 150.
            offset_axes: Use offset, rather than absolute, axis labels.

        Returns:
            None is save is `True` or the Figure if `False`.
        """

        if (self._cutouts_got is False) or (force):
            self.get_cutout_data(size)

        num_plots = self.measurements.shape[0]
        nrows = int(np.ceil(num_plots / columns))

        fig = plt.figure(figsize=figsize)

        fig.tight_layout()

        plots = {}

        if not self._checked_norms or force:
            self._analyse_norm_level(
                percentile=percentile,
                zscale=zscale,
                z_contrast=contrast
            )
        img_norms = self.norms

        for i in range(num_plots):
            cutout_row = self.cutout_df.iloc[i]
            measurement_row = self.measurements.iloc[i]
            target_coords = np.array(
                ([[
                    measurement_row.ra,
                    measurement_row.dec
                ]])
            )
            i += 1
            plots[i] = fig.add_subplot(
                nrows,
                columns,
                i,
                projection=cutout_row.wcs
            )

            if disable_autoscaling:
                if zscale:
                    img_norms = ImageNormalize(
                        cutout_row.data * 1.e3,
                        interval=ZScaleInterval(
                            contrast=contrast
                        )
                    )
                else:
                    img_norms = ImageNormalize(
                        cutout_row.data * 1.e3,
                        interval=PercentileInterval(percentile),
                        stretch=LinearStretch())

            im = plots[i].imshow(
                cutout_row.data * 1.e3, norm=img_norms, cmap="gray_r"
            )

            epoch_time = measurement_row.dateobs
            epoch = measurement_row.epoch

            plots[i].set_title('{}'.format(
                epoch_time.strftime("%Y-%m-%d %H:%M:%S")
            ))

            if not hide_epoch_labels:
                plots[i].text(
                    0.05, 0.9, f"{epoch}", transform=plots[i].transAxes
                )

            cross_target_coords = cutout_row.wcs.wcs_world2pix(
                target_coords, 0
            )
            crosshair_lines = self._create_crosshair_lines(
                cross_target_coords,
                0.15,
                0.15,
                cutout_row.data.shape
            )

            if (not cutout_row['selavy_overlay'].empty) and (not no_selavy):
                plots[i].set_autoscale_on(False)
                (
                    collection,
                    patches,
                    island_names
                ) = self._gen_overlay_collection(
                    cutout_row
                )
                plots[i].add_collection(collection, autolim=False)

            if self.forced_fits:
                (
                    collection,
                    patches,
                    island_names
                ) = self._gen_overlay_collection(
                    cutout_row, f_source=measurement_row
                )
                plots[i].add_collection(collection, autolim=False)
                del collection

            [plots[i].plot(
                l[0], l[1], color="C3", zorder=10, lw=1.5, alpha=0.6
            ) for l in crosshair_lines]

            lon = plots[i].coords[0]
            lat = plots[i].coords[1]

            lon.set_ticks_visible(False)
            lon.set_ticklabel_visible(False)
            lat.set_ticks_visible(False)
            lat.set_ticklabel_visible(False)

        if save:
            if outfile is None:
                outfile = self._get_save_name(epoch, ".png")

            if self.outdir != ".":
                outfile = os.path.join(
                    self.outdir,
                    outfile
                )

            plt.savefig(outfile, bbox_inches=True, dpi=plot_dpi)

            plt.close()

            return

        else:

            return fig

    def _gen_overlay_collection(
        self,
        cutout_row: pd.Series,
        f_source: Optional[pd.DataFrame] = None
    ) -> Tuple[PatchCollection, Patch, List[str]]:
        """
        Generates the ellipse collection for selavy sources to add
        to the matplotlib axis.

        Args:
            cutout_row: The row containing the selavy data
                to make the ellipses from.
            f_source: Forced fit extraction to create the
                 forced fit ellipse, defaults to None.

        Returns:
            Tuple of the ellipse collection, patches, and the island names.
        """

        wcs = cutout_row.wcs
        selavy_sources = cutout_row.selavy_overlay
        pix_scale = proj_plane_pixel_scales(wcs)
        sx = pix_scale[0]
        sy = pix_scale[1]
        degrees_per_pixel = np.sqrt(sx * sy)

        # define ellipse properties for clarity, selavy cut will have
        # already been created.
        if f_source is None:
            ww = selavy_sources["maj_axis"]
            hh = selavy_sources["min_axis"]
            aa = selavy_sources["pos_ang"]
            x = selavy_sources["ra_deg_cont"]
            y = selavy_sources["dec_deg_cont"]
        else:
            ww = np.array([f_source["f_maj_axis"]])
            hh = np.array([f_source["f_min_axis"]])
            aa = np.array([f_source["f_pos_ang"]])
            x = np.array([f_source["ra"]])
            y = np.array([f_source["dec"]])

        ww = ww.astype(float) / 3600.
        hh = hh.astype(float) / 3600.
        ww /= degrees_per_pixel
        hh /= degrees_per_pixel
        aa = aa.astype(float)
        x = x.astype(float)
        y = y.astype(float)

        coordinates = np.column_stack((x, y))

        coordinates = wcs.wcs_world2pix(coordinates, 0)

        # Create ellipses, collect them, add to axis.
        # Also where correction is applied to PA to account for how selavy
        # defines it vs matplotlib
        if f_source is None:
            island_names = selavy_sources["island_id"].apply(
                self._remove_sbid
            )
            colors = ["C2" if c.startswith(
                "n") else "C1" for c in island_names]
        else:
            island_names = [f_source["f_island_id"], ]
            colors = ["C3" for c in island_names]

        patches = [Ellipse(
            coordinates[i], hh[i], ww[i],
            aa[i]) for i in range(len(coordinates))]
        collection = PatchCollection(
            patches,
            facecolors="None",
            edgecolors=colors,
            lw=1.5)

        return collection, patches, island_names

    def skyview_contour_plot(
        self,
        epoch: str,
        survey: str,
        contour_levels: List[float] = [3., 5., 10., 15.],
        percentile: float = 99.9,
        zscale: bool = False,
        contrast: float = 0.2,
        outfile: Optional[str] = None,
        no_colorbar: bool = False,
        title: Optional[str] = None,
        save: bool = False,
        size: Optional[Angle] = None,
        force: bool = False,
        plot_dpi: int = 150,
    ) -> Union[None, plt.figure]:
        """
        Fetches a FITS file from SkyView of the requested survey at
        the source location and overlays ASKAP contours.

        Args:
            epoch: Epoch requested for the ASKAP contours.
            survey: Survey requested to be fetched using SkyView.
            contour_levels: Contour levels to plot which are multiples
                 of the local rms, defaults to [3., 5., 10., 15.].
            percentile: The value passed to the percentile
                normalization function, defaults to 99.9.
            zscale: Uses ZScale normalization instead of
                PercentileInterval, defaults to `False`.
            contrast: Contrast value passed to the ZScaleInterval
                function when zscale is selected, defaults to 0.2.
            outfile: Name to give the file, if None then the name is
                automatically generated, defaults to None.
            no_colorbar: Hides the colorbar, defaults to `False`.
            title: Plot title, defaults to None.
            save: Saves the file instead of returing the figure,
                defaults to `False`.
            size: Size of the cutout, defaults to None.
            force: Whether to force the re-fetching
                of the cutout data, defaults to `False`.
            plot_dpi: Specify the DPI of saved figures, defaults to 150.

        Returns:
            None if save is `True` or the figure object if `False`

        Raises:
            ValueError: If the source does not contain the requested epoch.
        """

        if (self._cutouts_got is False) or (force):
            self.get_cutout_data(size)

        size = self._size

        if epoch not in self.epochs:
            raise ValueError(
                "This source does not contain Epoch {}!".format(epoch)
            )

            return

        if outfile is None:
            outfile = self._get_save_name(epoch, ".png")

        if self.outdir != ".":
            outfile = os.path.join(
                self.outdir,
                outfile
            )

        index = self.epochs.index(epoch)

        try:
            paths = SkyView.get_images(
                position=self.measurements.iloc[index]['skycoord'],
                survey=[survey], radius=size
            )
            path_fits = paths[0][0]

            path_wcs = WCS(path_fits.header)

        except Exception as e:
            warnings.warn("SkyView fetch failed!")
            warnings.warn(e)
            return

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection=path_wcs)

        mean_vast, median_vast, rms_vast = sigma_clipped_stats(
            self.cutout_df.iloc[index].data
        )

        levels = [
            i * rms_vast for i in contour_levels
        ]

        if zscale:
            norm = ImageNormalize(
                path_fits.data,
                interval=ZScaleInterval(
                    contrast=contrast
                )
            )
        else:
            norm = ImageNormalize(
                path_fits.data,
                interval=PercentileInterval(percentile),
                stretch=LinearStretch()
            )

        im = ax.imshow(path_fits.data, norm=norm, cmap='gray_r')

        ax.contour(
            self.cutout_df.iloc[index].data,
            levels=levels,
            transform=ax.get_transform(self.cutout_df.iloc[index].wcs),
            colors='C0',
            zorder=10,
        )

        if title is None:
            if self.pipeline:
                title = "'{}' Epoch {} {}".format(
                    self.name, epoch, survey
                )
            else:
                title = "VAST Epoch {} '{}' {}".format(
                    epoch, self.name, survey
                )

        ax.set_title(title)

        lon = ax.coords[0]
        lat = ax.coords[1]
        lon.set_axislabel("Right Ascension (J2000)")
        lat.set_axislabel("Declination (J2000)")

        if not no_colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes(
                "right", size="3%", pad=0.1, axes_class=maxes.Axes)
            cb = fig.colorbar(im, cax=cax)

        if save:
            plt.savefig(outfile, bbox_inches="tight", dpi=plot_dpi)
            self.logger.debug("Saved {}".format(outfile))

            plt.close(fig)

            return
        else:
            return fig

    def make_png(
        self,
        epoch: str,
        selavy: bool = True,
        percentile: float = 99.9,
        zscale: bool = False,
        contrast: float = 0.2,
        outfile: Optional[str] = None,
        no_islands: bool = True,
        label: str = "Source",
        no_colorbar: bool = False,
        title: Optional[str] = None,
        crossmatch_overlay: bool = False,
        hide_beam: bool = False,
        save: bool = False,
        size: Optional[Angle] = None,
        force: bool = False,
        disable_autoscaling: bool = False,
        cutout_data: Optional[pd.DataFrame] = None,
        norms: Optional[ImageNormalize] = None,
        plot_dpi: int = 150,
        offset_axes: bool = True
    ) -> Union[None, plt.figure]:
        """
        Save a PNG of the image postagestamp.

        Args:
            epoch: The requested epoch.
            selavy: `True` to overlay selavy components, `False` otherwise.
            percentile: The value passed to the percentile
                normalization function, defaults to 99.9.
            zscale: Uses ZScale normalization instead of
                PercentileInterval, defaults to `False`.
            contrast: Contrast value passed to the ZScaleInterval
                function when zscale is selected, defaults to 0.2.
            outfile: Name to give the file, if None then the name is
                automatically generated, defaults to None.
            no_islands: Disable island lables on the png, defaults to
                `False`.
            label: Figure title (usually the name of the source of
                interest), defaults to "Source".
            no_colorbar: If `True`, do not show the colorbar on the png,
                defaults to `False`.
            title: String to set as title,
                defaults to None where a default title will be used.
            crossmatch_overlay: If 'True' then a circle is added to the png
                plot representing the crossmatch radius, defaults to `False`.
            hide_beam: If 'True' then the beam is not plotted onto the png
                plot, defaults to `False`.
            save: If `True` the plot is saved rather than the figure being
                returned, defaults to `False`.
            size: Size of the cutout, defaults to None.
            force: Whether to force the re-fetching of the cutout data,
                defaults to `False`.
            disable_autoscaling: Turn off the consistent normalization and
                calculate the normalizations separately for each epoch,
                defaults to `False`.
            cutout_data: Pass external cutout_data to be used
                instead of fetching the data, defaults to None.
            norms: Pass external normalization to be used
                instead of internal calculations.
            plot_dpi: Specify the DPI of saved figures, defaults to 150.
            offset_axes: Use offset, rather than absolute, axis labels.

        Returns:
            None if save is `True` or the figure object if `False`

        Raises:
            ValueError: If the source does not contain the requested epoch.
        """

        if (self._cutouts_got is False) or (force):
            if cutout_data is None:
                self.get_cutout_data(size)

        if epoch not in self.epochs:
            raise ValueError(
                "This source does not contain Epoch {}!".format(epoch)
            )

            return

        if outfile is None:
            outfile = self._get_save_name(epoch, ".png")

        if self.outdir != ".":
            outfile = os.path.join(
                self.outdir,
                outfile
            )

        index = self.epochs.index(epoch)

        if cutout_data is None:
            cutout_row = self.cutout_df.iloc[index]
        else:
            cutout_row = cutout_data.iloc[index]

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection=cutout_row.wcs)
        # Get the Image Normalisation from zscale, user contrast.
        if not disable_autoscaling:
            if norms is not None:
                img_norms = norms
            else:
                if not self._checked_norms:
                    self._analyse_norm_level(
                        percentile=percentile,
                        zscale=zscale,
                        z_contrast=contrast
                    )
                img_norms = self.norms
        else:
            if zscale:
                img_norms = ImageNormalize(
                    cutout_row.data * 1.e3,
                    interval=ZScaleInterval(
                        contrast=contrast
                    ))
            else:
                img_norms = ImageNormalize(
                    cutout_row.data * 1.e3,
                    interval=PercentileInterval(percentile),
                    stretch=LinearStretch())

        im = ax.imshow(
            cutout_row.data * 1.e3,
            norm=img_norms,
            cmap="gray_r"
        )

        # insert crosshair of target
        target_coords = np.array(
            ([[
                self.measurements.iloc[index].ra,
                self.measurements.iloc[index].dec
            ]])
        )

        target_coords = cutout_row.wcs.wcs_world2pix(
            target_coords, 0
        )

        crosshair_lines = self._create_crosshair_lines(
            target_coords,
            0.03,
            0.03,
            cutout_row.data.shape
        )

        [ax.plot(
            l[0], l[1], color="C3", zorder=10, lw=1.5, alpha=0.6
        ) for l in crosshair_lines]
        # the commented lines below are to use the crosshair
        # marker directly.
        # ax.scatter(
        #     [self.src_coord.ra.deg], [self.src_coord.dec.deg],
        #     transform=ax.get_transform('world'), marker="c",
        #     color="C3", zorder=10, label=label, s=1000, lw=1.5,
        #     alpha=0.5
        # )
        if crossmatch_overlay:
            try:
                crossmatch_patch = SphericalCircle(
                    (
                        self.measurements.iloc[index].skycoord.ra,
                        self.measurements.iloc[index].skycoord.dec
                    ),
                    self.crossmatch_radius,
                    transform=ax.get_transform('world'),
                    label="Crossmatch radius ({:.1f} arcsec)".format(
                        self.crossmatch_radius.arcsec
                    ), edgecolor='C4', facecolor='none', alpha=0.8)
                ax.add_patch(crossmatch_patch)
            except Exception as e:
                self.logger.warning(
                    "Crossmatch circle png overlay failed!"
                    " Has the source been crossmatched?")
                crossmatch_overlay = False

        if (not cutout_row['selavy_overlay'].empty) and selavy:
            ax.set_autoscale_on(False)
            collection, patches, island_names = self._gen_overlay_collection(
                cutout_row
            )
            ax.add_collection(collection, autolim=False)
            del collection

            # Add island labels, haven't found a better way other than looping
            # at the moment.
            if not no_islands and not self.islands:
                for i, val in enumerate(patches):
                    ax.annotate(
                        island_names[i],
                        val.center,
                        annotation_clip=True,
                        color="C0",
                        weight="bold")
        else:
            self.logger.debug(
                "PNG: No selavy selected or selavy catalogue failed. (%s)",
                self.name
            )

        if self.forced_fits:
            collection, patches, island_names = self._gen_overlay_collection(
                cutout_row,
                f_source=self.measurements.iloc[index]
            )
            ax.add_collection(collection, autolim=False)
            del collection

        legend_elements = [
            Line2D(
                [0], [0], marker='c', color='C3', label=label,
                markerfacecolor='g', ls="none", markersize=8
            )
        ]

        if selavy:
            legend_elements.append(
                Line2D(
                    [0], [0], marker='o', color='C1',
                    label="Selavy {}".format(self.cat_type),
                    markerfacecolor='none', ls="none", markersize=10
                )
            )

        if crossmatch_overlay:
            legend_elements.append(
                Line2D(
                    [0], [0], marker='o', color='C4',
                    label="Crossmatch radius ({:.1f} arcsec)".format(
                        self.crossmatch_radius.arcsec
                    ),
                    markerfacecolor='none', ls="none",
                    markersize=10
                )
            )

        if self.forced_fits:
            legend_elements.append(
                Line2D(
                    [0], [0], marker='o', color='C3',
                    label="Forced Fit",
                    markerfacecolor='none', ls="none",
                    markersize=10
                )
            )

        ax.legend(handles=legend_elements)
        lon = ax.coords[0]
        lat = ax.coords[1]
        lon.set_axislabel("Right Ascension (J2000)")
        lat.set_axislabel("Declination (J2000)")

        if not no_colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes(
                "right", size="3%", pad=0.1, axes_class=maxes.Axes)
            cb = fig.colorbar(im, cax=cax)
            cb.set_label("mJy/beam")

        if title is None:
            epoch_time = self.measurements[
                self.measurements['epoch'] == epoch
            ].iloc[0].dateobs
            title = "{} Epoch {} {}".format(
                self.name,
                epoch,
                epoch_time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )

        ax.set_title(title)

        if cutout_row.beam is not None and hide_beam is False:
            img_beam = cutout_row.beam
            if cutout_row.wcs.is_celestial:
                major = img_beam.major.value
                minor = img_beam.minor.value
                pa = img_beam.pa.value
                pix_scale = proj_plane_pixel_scales(
                    cutout_row.wcs
                )
                sx = pix_scale[0]
                sy = pix_scale[1]
                degrees_per_pixel = np.sqrt(sx * sy)
                minor /= degrees_per_pixel
                major /= degrees_per_pixel

                png_beam = AnchoredEllipse(
                    ax.transData, width=minor,
                    height=major, angle=pa, loc="lower right",
                    pad=0.5, borderpad=0.4,
                    frameon=False)
                png_beam.ellipse.set_edgecolor("k")
                png_beam.ellipse.set_facecolor("w")
                png_beam.ellipse.set_linewidth(1.5)

                ax.add_artist(png_beam)
        else:
            self.logger.debug("Hiding beam.")

        if offset_axes:
            axis_units = u.arcmin

            if size is None and cutout_row.wcs.is_celestial:
                pix_scale = proj_plane_pixel_scales(
                    cutout_row.wcs
                )
                sx = pix_scale[0]
                sy = pix_scale[1]
                xlims = ax.get_xlim()
                ylims = ax.get_ylim()

                xsize = sx * (xlims[1] - xlims[0])
                ysize = sy * (ylims[1] - ylims[0])
                size = max([xsize, ysize]) * u.deg

            if size is not None:
                if size < 2 * u.arcmin:
                    axis_units = u.arcsec
                elif size > 2 * u.deg:
                    axis_units = u.deg

            offset_postagestamp_axes(ax,
                                     self.coord,
                                     ra_units=axis_units,
                                     dec_units=axis_units
                                     )

        if save:
            plt.savefig(outfile, bbox_inches="tight", dpi=plot_dpi)
            self.logger.debug("Saved {}".format(outfile))

            plt.close(fig)
            return

        else:
            return fig

    def write_ann(
        self,
        epoch: str,
        outfile: str = None,
        crossmatch_overlay: bool = False,
        size: Optional[Angle] = None,
        force: bool = False,
        cutout_data: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Write a kvis annotation file containing all selavy sources
        within the image.

        Args:
            epoch: The requested epoch.
            outfile: Name of the file to write, defaults to None.
            crossmatch_overlay: If True, a circle is added to the
                annotation file output denoting the crossmatch radius,
                defaults to False.
            size: Size of the cutout, defaults to None.
            force: Whether to force the re-fetching
                of the cutout data, defaults to `False`
            cutout_data: Pass external cutout_data to be used
                instead of fetching the data, defaults to None.

        Returns:
            None
        """
        if (self._cutouts_got is False) or (force):
            if cutout_data is None:
                self.get_cutout_data(size)

        if outfile is None:
            outfile = self._get_save_name(epoch, ".ann")
        if self.outdir != ".":
            outfile = os.path.join(
                self.outdir,
                outfile
            )

        index = self.epochs.index(epoch)

        neg = False
        with open(outfile, 'w') as f:
            f.write("COORD W\n")
            f.write("PA SKY\n")
            f.write("FONT hershey14\n")
            f.write("COLOR BLUE\n")
            f.write("CROSS {0} {1} {2} {2}\n".format(
                self.measurements.iloc[index].ra,
                self.measurements.iloc[index].dec,
                3. / 3600.
            ))
            if crossmatch_overlay:
                try:
                    f.write("CIRCLE {} {} {}\n".format(
                        self.measurements.iloc[index].ra,
                        self.measurements.iloc[index].dec,
                        self.crossmatch_radius.deg
                    ))
                except Exception as e:
                    self.logger.warning(
                        "Crossmatch circle overlay failed!"
                        " Has the source been crossmatched?")
            f.write("COLOR GREEN\n")

            if cutout_data is None:
                selavy_cat_cut = self.cutout_df.iloc[index].selavy_overlay
            else:
                selavy_cat_cut = cutout_data.iloc[index].selavy_overlay

            for i, row in selavy_cat_cut.iterrows():
                if row["island_id"].startswith("n"):
                    neg = True
                    f.write("COLOR RED\n")
                ra = row["ra_deg_cont"]
                dec = row["dec_deg_cont"]
                f.write(
                    "ELLIPSE {} {} {} {} {}\n".format(
                        ra,
                        dec,
                        float(
                            row["maj_axis"]) /
                        3600. /
                        2.,
                        float(
                            row["min_axis"]) /
                        3600. /
                        2.,
                        float(
                            row["pos_ang"])))
                f.write(
                    "TEXT {} {} {}\n".format(
                        ra, dec, self._remove_sbid(
                            row["island_id"])))
                if neg:
                    f.write("COLOR GREEN\n")
                    neg = False

        self.logger.debug("Wrote annotation file {}.".format(outfile))

    def write_reg(
        self,
        epoch: str,
        outfile: Optional[str] = None,
        crossmatch_overlay: bool = False,
        size: Optional[Angle] = None,
        force: bool = False,
        cutout_data: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Write a DS9 region file containing all selavy sources within the image

        Args:
            epoch: The requested epoch.
            outfile: Name of the file to write, defaults to None.
            crossmatch_overlay: If True, a circle is added to the
                annotation file output denoting the crossmatch radius,
                defaults to False.
            size: Size of the cutout, defaults to None.
            force: Whether to force the re-fetching
                of the cutout data, defaults to `False`.
            cutout_data: Pass external cutout_data to be used
                instead of fetching the data, defaults to None.

        Returns:
            None
        """
        if (self._cutouts_got is False) or (force):
            if cutout_data is None:
                self.get_cutout_data(size)

        if outfile is None:
            outfile = self._get_save_name(epoch, ".reg")
        if self.outdir != ".":
            outfile = os.path.join(
                self.outdir,
                outfile
            )

        index = self.epochs.index(epoch)
        with open(outfile, 'w') as f:
            f.write("# Region file format: DS9 version 4.0\n")
            f.write("global color=green font=\"helvetica 10 normal\" "
                    "select=1 highlite=1 edit=1 "
                    "move=1 delete=1 include=1 "
                    "fixed=0 source=1\n")
            f.write("fk5\n")
            f.write(
                "point({} {}) # point=x color=blue\n".format(
                    self.measurements.iloc[index].ra,
                    self.measurements.iloc[index].dec,
                ))
            if crossmatch_overlay:
                try:
                    f.write("circle({} {} {}) # color=blue\n".format(
                        self.measurements.iloc[index].ra,
                        self.measurements.iloc[index].dec,
                        self.crossmatch_radius.deg
                    ))
                except Exception as e:
                    self.logger.warning(
                        "Crossmatch circle overlay failed!"
                        " Has the source been crossmatched?")

            if cutout_data is None:
                selavy_cat_cut = self.cutout_df.iloc[index].selavy_overlay
            else:
                selavy_cat_cut = cutout_data.iloc[index].selavy_overlay

            for i, row in selavy_cat_cut.iterrows():
                if row["island_id"].startswith("n"):
                    color = "red"
                else:
                    color = "green"
                ra = row["ra_deg_cont"]
                dec = row["dec_deg_cont"]
                f.write(
                    "ellipse({} {} {} {} {}) # color={}\n".format(
                        ra,
                        dec,
                        float(
                            row["maj_axis"]) /
                        3600. /
                        2.,
                        float(
                            row["min_axis"]) /
                        3600. /
                        2.,
                        float(
                            row["pos_ang"]) +
                        90.,
                        color))
                f.write(
                    "text({} {} \"{}\") # color={}\n".format(
                        ra - (10. / 3600.), dec, self._remove_sbid(
                            row["island_id"]), color))

        self.logger.debug("Wrote region file {}.".format(outfile))

    def _remove_sbid(self, island: str) -> str:
        """Removes the SBID component of the island name.

        Takes into account negative 'n' label for negative sources.

        Args:
            island: island name.

        Returns:
            Truncated island name.
        """

        temp = island.split("_")
        new_val = "_".join(temp[-2:])
        if temp[0].startswith("n"):
            new_val = "n" + new_val
        return new_val

    def _create_crosshair_lines(
        self,
        target: np.ndarray,
        pixel_buff: float,
        length: float,
        img_size: Tuple[int, int]
    ) -> List[List[float]]:
        """
        Takes the target pixel coordinates and creates the plots
        that are required to plot a 'crosshair' marker.

        To keep the crosshair consistent between scales, the values are
        provided as percentages of the image size.

        Args:
            target: The target in pixel coordinates.
            pixel_buff: Percentage of image size that is the buffer
                of the crosshair, i.e. the distance between the target and
                beginning of the line.
            length: Size of the line of the crosshair, again as a
                percentage of the image size.
            img_size: Tuple size of the image array.

        Returns:
            List of pairs of pixel coordinates to plot using scatter.
        """
        x = target[0][0]
        y = target[0][1]

        min_size = np.min(img_size)

        pixel_buff = int(min_size * pixel_buff)
        length = int(min_size * length)

        plots = []

        plots.append([[x, x], [y + pixel_buff, y + pixel_buff + length]])
        plots.append([[x, x], [y - pixel_buff, y - pixel_buff - length]])
        plots.append([[x + pixel_buff, x + pixel_buff + length], [y, y]])
        plots.append([[x - pixel_buff, x - pixel_buff - length], [y, y]])

        return plots

    def simbad_search(
        self, radius: Angle = Angle(20. * u.arcsec)
    ) -> Union[None, Table]:
        """
        Searches SIMBAD for object coordinates and returns matches.

        Args:
            radius: Radius to search, defaults to Angle(20. * u.arcsec)

        Returns:
            Table of matches or None if no matches

        Raises:
            ValueError: Error in performing the SIMBAD region search.
        """
        Simbad.add_votable_fields('ra(d)', 'dec(d)')

        try:
            result_table = Simbad.query_region(self.coord, radius=radius)
            if result_table is None:
                return None

            return result_table

        except Exception as e:
            raise ValueError(
                "Error in performing the SIMBAD region search! Error: %s", e
            )
            return None

    def ned_search(
        self, radius: Angle = Angle(20. * u.arcsec)
    ) -> Union[None, Table]:
        """
        Searches NED for object coordinates and returns matches.

        Args:
            radius: Radius to search, defaults to Angle(20. * u.arcsec).

        Returns:
            Table of matches or None if no matches

        Raises:
            ValueError: Error in performing the NED region search.
        """
        try:
            result_table = Ned.query_region(self.coord, radius=radius)

            return result_table

        except Exception as e:
            raise ValueError(
                "Error in performing the NED region search! Error: %s", e
            )
            return None

    def casda_search(
        self,
        radius: Angle = Angle(20. * u.arcsec),
        filter_out_unreleased: bool = False,
        show_all: bool = False
    ) -> Union[None, Table]:
        """
        Searches CASDA for object coordinates and returns matches.

        Args:
            radius: Radius to search, defaults to Angle(20. * u.arcsec).
            filter_out_unreleased: Remove unreleased data,
                defaults to `False`.
            show_all: Show all available data, defaults to `False`.

        Returns:
            Table of matches or None if no matches

        Raises:
            ValueError: Error in performing the CASDA region search.
        """
        try:
            result_table = Casda.query_region(self.coord, radius=radius)

            if filter_out_unreleased:
                result_table = Casda.filter_out_unreleased(result_table)
            if not show_all:
                mask = result_table[
                    'dataproduct_subtype'
                ] == 'cont.restored.t0'
                result_table = result_table[mask]
                mask = [(
                    ("image.i" in i) & ("taylor.0.res" in i)
                ) for i in result_table[
                    'filename'
                ]]
                result_table = result_table[mask]

            return result_table

        except Exception as e:
            raise ValueError(
                "Error in performing the CASDA region search! Error: %s", e
            )
            return None

    def _get_fluxes_and_errors(
        self,
        suffix: str,
        forced_fits: bool
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Selects the correct fluxes, upper limits or forced fits
        to calculate the metrics

        Args:
            suffix: 'peak' or 'int'.
            forced_fits: Set to `True` if forced fits should be used.

        Returns:
            The fluxes and errors to use.
        """
        if self.pipeline:
            non_detect_label = 'flux_{}'.format(suffix)
            non_detect_label_err = 'flux_{}_err'.format(suffix)
            scale = 1.
            detection_label = 'forced'
            detection_value = False
        else:
            detection_label = 'detection'
            detection_value = True
            if forced_fits:
                non_detect_label = 'f_flux_{}'.format(suffix)
                non_detect_label_err = 'f_flux_{}_err'.format(suffix)
                scale = 1.
            else:
                scale = 5.
                non_detect_label = 'rms_image'
                non_detect_label_err = 'rms_image'

        detect_mask = self.measurements[detection_label] == detection_value

        detect_fluxes = (
            self.measurements[detect_mask]['flux_{}'.format(suffix)]
        )
        detect_errors = (
            self.measurements[detect_mask]['flux_{}_err'.format(
                suffix
            )]
        )

        non_detect_fluxes = (
            self.measurements[~detect_mask][non_detect_label] * scale
        )
        non_detect_errors = (
            self.measurements[~detect_mask][non_detect_label_err]
        )

        fluxes = detect_fluxes.append(non_detect_fluxes)
        errors = detect_errors.append(non_detect_errors)

        return fluxes, errors

    def calc_eta_metric(
        self,
        use_int: bool = False,
        forced_fits: bool = False
    ) -> float:
        """
        Calculate the eta variability metric

        Args:
            use_int: Calculate using integrated (rather than peak) flux,
                defaults to `False`
            forced_fits: Use forced fits, defaults to `False`

        Returns:
            Eta variability metric.

        Raises:
            Exception: No forced fits are present when forced fits have been
                selected.
        """
        if self.measurements.shape[0] == 1:
            return 0.

        suffix = 'int' if use_int else 'peak'

        if forced_fits and not self.forced_fits:
            raise Exception(
                "Forced fits selected but no forced fits are present!"
            )

        fluxes, errors = self._get_fluxes_and_errors(suffix, forced_fits)
        n_src = fluxes.shape[0]

        weights = 1. / errors**2
        eta = (n_src / (n_src - 1)) * (
            (weights * fluxes**2).mean() - (
                (weights * fluxes).mean()**2 / weights.mean()
            )
        )

        return eta

    def calc_v_metric(
        self,
        use_int: bool = False,
        forced_fits: bool = False
    ) -> float:
        """
        Calculate the V variability metric.

        Args:
            use_int: Calculate using integrated (rather than peak) flux,
                defaults to `False`.
            forced_fits: Use forced fits, defaults to `False`.

        Returns:
            V variability metric.

        Raises:
            Exception: No forced fits are present when forced fits have been
                selected.
        """
        if self.measurements.shape[0] == 1:
            return 0.

        suffix = 'int' if use_int else 'peak'

        if forced_fits and not self.forced_fits:
            raise Exception(
                "Forced fits selected but no forced fits are present!"
            )

        fluxes, _ = self._get_fluxes_and_errors(suffix, forced_fits)
        v = fluxes.std() / fluxes.mean()

        return v

    def calc_eta_and_v_metrics(
        self,
        use_int: bool = False,
        forced_fits: bool = False
    ) -> Tuple[float, float]:
        """
        Calculate both variability metrics

        Args:
            use_int: Calculate using integrated (rather than peak) flux,
                defaults to `False`.
            forced_fits: Use forced fits, defaults to `False`.

        Returns:
            Variability metrics eta and v.
        """

        eta = self.calc_eta_metric(use_int=use_int, forced_fits=forced_fits)
        v = self.calc_v_metric(use_int=use_int, forced_fits=forced_fits)

        return eta, v
