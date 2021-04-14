# Source class

import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.axes as maxes
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
from astroquery.simbad import Simbad
from astroquery.ned import Ned
from astroquery.casda import Casda
from astropy.stats import sigma_clipped_stats
from astroquery.skyview import SkyView
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import logging.config
import logging.handlers
import logging
import warnings
import pandas as pd
import os
import numpy as np
import gc
import signal

from vasttools.utils import crosshair
from vasttools.survey import Image
from vasttools.survey import RELEASED_EPOCHS
from vasttools.utils import filter_selavy_components
# run crosshair to set up the marker
crosshair()

matplotlib.pyplot.switch_backend('Agg')


class Source:
    '''
    This is a class representation of a catalogued source position

    Attributes
    ----------

    pipeline : bool
        Set to `True` if the source is generated from a VAST
        Pipeline run.
    coord : astropy.coordinates.SkyCoord
        The coordinate of the source as a SkyCoord object.
        Planets can sometimes have a SkyCoord containing more than
        one coordinate.
    name : str
        The name of the source.
    epochs : list
        The epochs the source contains.
    fields : list
        The fields the source contains.
    stokes : str
        The Stokes parameter of the source.
    crossmatch_radius : astropy.coordinates.Angle
        Angle of the crossmatch. This will not be valid for
        pipeline sources.
    measurements : pandas.core.frame.DataFrame
        The individual measurements of the source.
    islands : bool
       Set to `True` if islands have been used for the source creation.
    outdir : str
        Path that will be appended to any files that are saved.
    base_folder : str
        The directory where the data (fits files) is held.
    image_type : str
        'TILES' or 'COMBINED'
    tiles : bool
        `True` if `image_type` == `TILES`.
    detections : int
        The number of selavy detections the source contains.
    limits : int
        The number of upper limits the source contains. Will be set to
        `None` for pipeline sources.
    forced_fits : int
        The number of forced fits the source contains.
    norms : astropy.visualization.ImageNormalize
        Contains the normalization value to use for
        consistent normalization across the measurements
        for png representation.
    planet : bool
        Set to `True` if the source has been defined as a planet.

    Methods
    ----------

    write_measurements(simple=False, outfile=None)
        Saves the measurements to a csv file.

    plot_lightcurve(
        sigma_thresh=5, figsize=(8, 4), min_points=2,
        min_detections=0, mjd=False, start_date=None, grid=False,
        yaxis_start="auto", peak_flux=True, save=False,
        outfile=None, use_forced_for_limits=False,
        use_forced_for_all=False, hide_legend=False, plot_dpi=150
    )
        Displays the lightcurve plot. Use the save parameter to write
        the plot to a png file.

    show_png_cutout(
        epoch, selavy=True, percentile=99.9, zscale=False,
        contrast=0.2, no_islands=True, label="Source", no_colorbar=False,
        title=None, crossmatch_overlay=False, hide_beam=False, size=None,
        force=False
    )
        Displays the png cutout of the epoch selected.

    save_png_cutout(
        epoch, selavy=True, percentile=99.9, zscale=False,
        contrast=0.2, no_islands=True, label="Source", no_colorbar=False,
        title=None, crossmatch_overlay=False, hide_beam=False, size=None,
        force=False, outfile=None, plot_dpi=150
    )
        Saves the png cutout of the epoch selected.

    save_fits_cutout(
        epoch, outfile=None, size=None, force=False, cutout_data=None
    )
        Saves the FITS cutout of the epoch selected.

    show_all_png_cutouts(
        columns=4, percentile=99.9, zscale=False, contrast=0.1,
        outfile=None, save=False, size=None, figsize=(10, 5),
        force=False, no_selavy=False, disable_autoscaling=False, plot_dpi=150
    )
        Displays a grid view plot of all the cutouts.

    skyview_contour_plot(
        epoch, survey, contour_levels=[3., 5., 10., 15.],
        percentile=99.9, zscale=False, contrast=0.2, outfile=None,
        no_colorbar=False, title=None, save=False, size=None,
        force=False, plot_dpi=150
    )
        Fetches the SkyView image FITS from the selected survey
        and overlays the ASKAP contours from the selected epoch.
        By default the contour levels are at 3, 5, 10 and 15 sigma.

    write_ann(
        epoch, outfile=None, crossmatch_overlay=False,
        size=None, force=False, cutout_data=None
    )
        Save the selected epoch annotation file.

    write_reg(
        epoch, outfile=None, crossmatch_overlay=False,
        size=None, force=False, cutout_data=None
    )
        Save the selected epoch annotation file.

    save_all_fits_cutouts(size=None, force=False, cutout_data=None)
        Saves all the FITS cutout files.

    save_all_png_cutouts(
        selavy=True, percentile=99.9, zscale=False, contrast=0.2,
        islands=True, no_colorbar=False, crossmatch_overlay=False,
        hide_beam=False, size=None, disable_autoscaling=False,
        cutout_data=None, calc_script_norms=False, plot_dpi=150
    )
        Saves all the png cutouts.

    save_all_ann(crossmatch_overlay=False, cutout_data=None)
        Saves all the annotation files.

    save_all_reg(crossmatch_overlay=False, cutout_data=None)
        Saves all the region files

    simbad_search(radius=Angle(20. * u.arcsec))
        Perform a SIMBAD search around the source location.
        Returns an astropy Table with results.

    ned_search(radius=Angle(20. * u.arcsec))
        Perform a NED search around the source location.
        Returns an astropy Table with results.

    casda_search(
        radius=Angle(20. * u.arcsec),
        filter_out_unreleased=False,
        show_all=False
    )
        Perform a CASDA search around the source location.
        Returns an astropy Table with results.

    calc_eta_metric(use_int=False, forced_fits=False)
        Calculates the eta metric of the source. Forced fits can be
        used instead of upper limits by setting forced_fits to `True`.

    calc_v_metric(use_int=False, forced_fits=False)
        Calculates the v metric of the source. Forced fits can be
        used instead of upper limits by setting forced_fits to `True`.

    calc_eta_and_v_metrics(self, use_int=False, forced_fits=False)
        Returns both the eta and v metrics using the previously defined
        functions.

    get_cutout_data(self, size=None)
        Fetches the cutout data required to produce plots for the source.
        This shouldn't need to be called directly.
    '''

    def __init__(
        self,
        coord,
        name,
        epochs,
        fields,
        stokes,
        primary_field,
        crossmatch_radius,
        measurements,
        base_folder,
        image_type="COMBINED",
        islands=False,
        outdir=".",
        planet=False,
        pipeline=False,
        tiles=False,
        forced_fits=False,
    ):
        '''
        Constructor method

        :param coord: Source coordinates
        :type coord: `astropy.coordinates.sky_coordinate.SkyCoord`
        :param name: The name of the source.
        :type name: str
        :param epochs: The epochs that the source contains.
        :type epochs: list
        :param fields: The fields that the source contains.
        :type fields: list
        :param stokes: The stokes parameter of the source.
        :type stokes: str
        :param primary_field: The primary VAST Pilot field of the source.
        :type primary_field: str
        :param crossmatch_radius: The crossmatch radius used to find the
            measurements.
        :type crossmatch_radius: astropy.coordinates.Angle
        :param measurements: DataFrame containing the measurements.
        :type measurements: pandas.core.frame.DataFrame
        :param base_folder: Path to base folder in default directory structure
        :type base_folder: str
        :param image_type: The string representation of the image type,
            either 'COMBINED' or 'TILES', defaults to "COMBINED"
        :type image_type: str, optional
        :param islands: Is `True` if islands has been useed instead of
            components, defaults to `False`
        :type islands: bool, optional
        :param outdir: The directory where any media outputs will be written
            to, defaults to "."
        :type outdir: str, optional
        :param planet: Set to `True` if the source is a planet, defaults
            to `False`.
        :type planet: bool, optional
        :param pipeline: Set to `True` if the source has been loaded from a
            VAST Pipeline run, defaults to `False`
        :type pipeline: bool, optional
        :param tiles: Set to 'True` if the source is from a tile images,
            defaults to `False`.
        :type tiles: bool, optional
        :param forced_fits: Set to `True` if forced fits are included in the
            source measurments, defaults to `False`.
        :type forced_fits: bool, optional

        '''
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

    def write_measurements(self, simple=False, outfile=None):
        '''
        Write the measurements to a CSV file.

        :param simple: Only include flux density and uncertainty in returned \
        table, defaults to `False`
        :type simple: bool, optional
        :param outfile: File to write measurements to, defaults to None
        :type outfile: str, optional
        '''

        if simple:
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

    def plot_lightcurve(self, sigma_thresh=5, figsize=(8, 4),
                        min_points=2, min_detections=0, mjd=False,
                        start_date=None, grid=False, yaxis_start="0",
                        peak_flux=True, save=False, outfile=None,
                        use_forced_for_limits=False, use_forced_for_all=False,
                        hide_legend=False, plot_dpi=150):
        '''
        Plot source lightcurves and save to file

        :param sigma_thresh: Threshold to use for upper limits, defaults to 5
        :type sigma_thresh: int or float, optional
        :param figsize: Figure size, defaults to (8, 4)
        :type figsize: tuple of floats, optional
        :param min_points: Minimum number of points for plotting, defaults
            to 2
        :type min_points: float, optional
        :param min_detections:  Minimum number of detections for plotting,
            defaults to 0
        :type min_detections: float, optional
        :param mjd: Plot x-axis in MJD rather than datetime, defaults to False
        :type mjd: bool, optional
        :param start_date: Plot in days from start date, defaults to None
        :type start_date: pandas datetime, optional
        :param grid: Turn on matplotlib grid, defaults to False
        :type grid: bool, optional
        :param yaxis_start: Define where the y-axis begins from, either 'auto'
            or '0', defaults to "0".
        :type yaxis_start: str, optional
        :param peak_flux: Uses peak flux instead of integrated flux,
            defaults to `True`
        :type peak_flux: bool, optional
        :param save: When `True` the plot is saved rather than displayed,
            defaults to `False`
        :type save: bool, optional
        :param outfile: , defaults to None
        :type outfile: , optional
        :param use_forced_for_limits: Use the forced extractions instead of
            upper limits for non-detections., defaults to `False`
        :type use_forced_for_limits: bool, optional
        :param use_forced_for_all: Use the forced fits for all the datapoints,
            defaults to `False`
        :type use_forced_for_all: bool, optional
        :param hide_legend: Hide the legend, defaults to `False`
        :type hide_legend: bool, optional
        :param plot_dpi: Specify the DPI of saved figures, defaults to 150
        :type plot_dpi: int, optional

        :returns: None if save is `True` or the matplotlib figure
            if save is `False`.
        :rtype: None or matplotlib.pyplot.Figure
        '''
        if use_forced_for_all or use_forced_for_limits:
            if not self.forced_fits:
                raise Exception(
                    "Source does not have any forced fits points to plot."
                )

        if self.detections < min_detections:
            self.logger.error(
                "Number of detections (%i) lower than minimum required (%i)",
                self.detections, min_detections
            )
            return

        if self.measurements.shape[0] < min_points:
            self.logger.error(
                "Number of datapoints (%i) lower than minimum required (%i)",
                self.detections, min_detections
            )
            return

        # remove empty values
        measurements = self.measurements
        if not self.pipeline and not (
            use_forced_for_limits or use_forced_for_all
        ):
            measurements = self.measurements[
                self.measurements['rms_image'] != -99
            ]

        if measurements.empty:
            self.logger.debug(
                "%s has no measurements! No lightcurve will be produced.",
                self.name
            )
            return

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
            if use_forced_for_limits or self.pipeline:
                max_y = np.nanmax(
                    detections[flux_col].tolist() +
                    upper_lims[value_col].tolist()
                )
            elif use_forced_for_all:
                max_y = np.nanmax(detections[flux_col].tolist())
            else:
                max_y = np.nanmax(
                    detections[flux_col].tolist() +
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

    def get_cutout_data(self, size=None):
        '''
        Function to fetch the cutout data for that source
        required for producing all the media output. If size
        is not provided then the default size in _get_cutout
        will be used (5 arcmin).

        :param size: The angular size of the cutouts,
            defaults to None
        :type size: astropy.coordinates.Angle, optional
        '''
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
        self, percentile=99.9,
        zscale=False, z_contrast=0.2,
        cutout_data=None,
        return_norm=False
    ):
        '''
        Selects the appropirate image to use as the normalization
        value for each image. Either the first `detection` image
        is used, or the first image in time if there are no detections.

        :param percentile: The valye passed to the percentile
            normalization function, defaults to 99.9.
        :type percentile: float, optional
        :param zscale: Uses ZScale normalization instead of
            PercentileInterval, defaults to `False`
        :type zscale: bool, optional
        :param z_contrast: Contast value passed to the ZScaleInterval
            function when zscale is selected, defaults to 0.2.
        :type z_contrast: float, optional
        :param cutout_data: Pass external cutout_data to be used
            instead of fetching the data, defaults to None.
        :type cutout_data: None or pandas.core.frame.DataFrame
        :param return_norm: If `True` the calculated norm is returned
            by the function, defaults to False.
        :type return_norm: bool, optional.

        :returns: None if return_norm is `False` or the normalization
            `True`.
        :rtype: None or astropy.visualization.ImageNormalize.
        '''

        if cutout_data is None:
            if not self._cutouts_got:
                self.logger.warning(
                    "Fetch cutout data before running this function!"
                )

        if cutout_data is None:
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

    def _get_cutout(self, row, size=Angle(5. * u.arcmin)):
        '''
        Does the actual fetching of the cutout data.

        :param row: The row in the measurements df for which
            media will be fetched.
        :type row: pandas.core.series.Series
        :param size: The size of the cutout,
            defaults to Angle(5.*u.arcmin)
        :type size: `astropy.coordinates.angles.Angle`, optional

        :returns: Tuple containing the cutout data.
        :rtype: Tuple (numpy.ndarray, astropy.wcs.WCS,
            astropy.io.fits.header, pandas.core.frame.DataFrame,
            radio_beam.Beam)
        '''

        if self.pipeline:
            image = Image(
                row.field, row.epoch, self.stokes, self.base_folder,
                path=row.image, rmspath=row.rms
            )
        else:
            e = row.epoch
            if "-" in e:
                e = e.split("-")[0]
            image = Image(
                row.field, e, self.stokes,
                self.base_folder, tiles=self.tiles,
                sbid=row.sbid
            )

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
            epoch,
            selavy=True,
            percentile=99.9,
            zscale=False,
            contrast=0.2,
            no_islands=True,
            label="Source",
            no_colorbar=False,
            title=None,
            crossmatch_overlay=False,
            hide_beam=False,
            size=None,
            force=False,
    ):
        '''
        Wrapper for _make_png to make nicer interactive function.
        No access to save.

        :param epoch: The epoch to show.
        :type epoch: str
        :param selavy: If `True` then selavy overlay are shown,
             defaults to `True`
        :type selavy: bool, optional
        :param percentile: The valye passed to the percentile
            normalization function, defaults to 99.9.
        :type percentile: float, optional
        :param zscale: Uses ZScale normalization instead of
            PercentileInterval, defaults to `False`
        :type zscale: bool, optional
        :param contrast: Contast value passed to the ZScaleInterval
            function when zscale is selected, defaults to 0.2.
        :type contrast: float, optional
        :param no_islands: Hide island name labels, defaults to `True`
        :type no_islands: bool, optional
        :param label: legend label for source, defaults to "Source"
        :type label: str, optional
        :param no_colorbar: Hides the colorbar, defaults to `False`
        :type no_colorbar: bool, optional
        :param title: Sets the plot title, defaults to None
        :type title: str, optional
        :param crossmatch_overlay: Plots a circle that represents the
            crossmatch radius, defaults to `False`.
        :type crossmatch_overlay: bool, optional
        :param hide_beam: Hide the beam on the plot, defaults to `False`.
        :type hide_beam: bool, optional
        :param size: Size of the cutout, defaults to None
        :type size: None or astropy.coordinates.Angle, optional
        :param force: Whether to force the re-fetching
            of the cutout data, defaults to `False`
        :type force: bool, optional

        :returns: Figure object.
        :rtype: matplotlib.pyplot.Figure
        '''

        fig = self._make_png(
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
            epoch,
            selavy=True,
            percentile=99.9,
            zscale=False,
            contrast=0.2,
            no_islands=True,
            label="Source",
            no_colorbar=False,
            title=None,
            crossmatch_overlay=False,
            hide_beam=False,
            size=None,
            force=False,
            outfile=None,
            plot_dpi=150
    ):
        '''
        Wrapper for _make_png to make nicer interactive function.
        Always save.

        :param epoch: The epoch to show.
        :type epoch: str
        :param selavy: If `True` then selavy overlay are shown,
             defaults to `True`
        :type selavy: bool, optional
        :param percentile: The valye passed to the percentile
            normalization function, defaults to 99.9.
        :type percentile: float, optional
        :param zscale: Uses ZScale normalization instead of
            PercentileInterval, defaults to `False`
        :type zscale: bool, optional
        :param contrast: Contast value passed to the ZScaleInterval
            function when zscale is selected, defaults to 0.2.
        :type contrast: float, optional
        :param no_islands: Hide island name labels, defaults to `True`
        :type no_islands: bool, optional
        :param label: legend label for source, defaults to "Source"
        :type label: str, optional
        :param no_colorbar: Hides the colorbar, defaults to `False`
        :type no_colorbar: bool, optional
        :param title: Sets the plot title, defaults to None
        :type title: str, optional
        :param crossmatch_overlay: Plots a circle that represents the
            crossmatch radius, defaults to `False`.
        :type crossmatch_overlay: bool, optional
        :param hide_beam: Hide the beam on the plot, defaults to `False`.
        :type hide_beam: bool, optional
        :param size: Size of the cutout, defaults to None
        :type size: None or astropy.coordinates.Angle, optional
        :param force: Whether to force the re-fetching
            of the cutout data, defaults to `False`
        :type force: bool, optional
        :param outfile: Name to give the file, if None then the name is
            automatically generated, defaults to None.
        :type outfile: None or str, optional
        :param plot_dpi: Specify the DPI of saved figures, defaults to 150
        :type plot_dpi: int, optional
        '''

        fig = self._make_png(
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

    def _get_save_name(self, epoch, ext):
        '''
        Generate name of file to save to

        :param epoch: Epoch corresponding to requested data
        :type epoch: str
        :param ext: File extension
        :type ext: str

        :returns: Name of file to save.
        :rtype: str
        '''

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
        self, epoch, outfile=None, size=None, force=False, cutout_data=None
    ):
        '''
        Saves the FITS file cutout of the requested epoch.

        :param epoch: Requested epoch
        :type epoch: str
        :param outfile: File to save to, defaults to None
        :type outfile: str, optional
        :param size: Size of the cutout, defaults to None
        :type size: None or astropy.coordinates.Angle, optional
        :param force: Whether to force the re-fetching
            of the cutout data, defaults to `False`
        :type force: bool, optional
        :param cutout_data: Pass external cutout_data to be used
            instead of fetching the data, defaults to None.
        :type cutout_data: None or pandas.core.frame.DataFrame
        '''

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

    def save_all_ann(self, crossmatch_overlay=False, cutout_data=None):
        '''
        Save kvis annotation file corresponding to the source

        :param crossmatch_overlay: Include the crossmatch radius, \
        defaults to `False`
        :type crossmatch_overlay: bool, optional
        :param cutout_data: Pass external cutout_data to be used
            instead of fetching the data, defaults to None.
        :type cutout_data: None or pandas.core.frame.DataFrame
        '''

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

    def save_all_reg(self, crossmatch_overlay=False, cutout_data=None):
        '''
        Save DS9 region file corresponding to the source

        :param crossmatch_overlay: Include the crossmatch radius, \
        defaults to `False`
        :type crossmatch_overlay: bool, optional
        :param cutout_data: Pass external cutout_data to be used
            instead of fetching the data, defaults to None.
        :type cutout_data: None or pandas.core.frame.DataFrame
        '''

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
        self, size=None, force=False, cutout_data=None
    ):
        '''
        Save all cutouts of the source to fits file

        :param size: Size of the cutouts, defaults to None
        :type size: None or astropy.coordinates.Angle, optional
        :param force: Whether to force the re-fetching
            of the cutout data, defaults to `False`
        :type force: bool, optional
        :param cutout_data: Pass external cutout_data to be used
            instead of fetching the data, defaults to None.
        :type cutout_data: None or pandas.core.frame.DataFrame
        '''

        if (self._cutouts_got is False) or (force):
            if cutout_data is None:
                self.get_cutout_data(size)

        for e in self.measurements['epoch']:
            self.save_fits_cutout(e, cutout_data=cutout_data)

    def save_all_png_cutouts(
        self,
        selavy=True,
        percentile=99.9,
        zscale=False,
        contrast=0.2,
        no_islands=True,
        no_colorbar=False,
        crossmatch_overlay=False,
        hide_beam=False,
        size=None,
        disable_autoscaling=False,
        cutout_data=None,
        calc_script_norms=False,
        plot_dpi=150
    ):
        '''
        Wrapper function to save all the png cutouts
        for all epochs.

        :param selavy: If `True` then selavy overlay are shown,
             defaults to `True`
        :type selavy: bool, optional
        :param percentile: The valye passed to the percentile
            normalization function, defaults to 99.9.
        :type percentile: float, optional
        :param zscale: Uses ZScale normalization instead of
            PercentileInterval, defaults to `False`
        :type zscale: bool, optional
        :param contrast: Contast value passed to the ZScaleInterval
            function when zscale is selected, defaults to 0.2.
        :type contrast: float, optional
        :param no_islands: Hide island name labels, defaults to `True`
        :type no_islands: bool, optional
        :param no_colorbar: Hides the colorbar, defaults to `False`
        :type no_colorbar: bool, optional
        :param crossmatch_overlay: Plots a circle that represents the
            crossmatch radius, defaults to `False`.
        :type crossmatch_overlay: bool, optional
        :param hide_beam: Hide the beam on the plot, defaults to `False`.
        :type hide_beam: bool, optional
        :param size: Size of the cutout, defaults to None
        :type size: None or astropy.coordinates.Angle, optional
        :param disable_autoscaling: Do not use the consistent normalization
            values but calculate norms separately for each epoch,
            defaults to `False`
        :type disable_autoscaling: bool, optional
        :param cutout_data: Pass external cutout_data to be used
            instead of fetching the data, defaults to None.
        :type cutout_data: None or pandas.core.frame.DataFrame
        :param calc_script_norms: When passing cutout data this parameter
            can be set to True to pass this cutout data to the analyse norms
            function, defaults to False.
        :type calc_script_norms: bool, optional
        :param plot_dpi: Specify the DPI of saved figures, defaults to 150
        :type plot_dpi: int, optional
        '''

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
            self._make_png,
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
        self, columns=4, percentile=99.9, zscale=False,
        contrast=0.1, outfile=None, save=False, size=None, figsize=(10, 5),
        force=False, no_selavy=False, disable_autoscaling=False,
        hide_epoch_labels=False, plot_dpi=150
    ):
        '''
        Creates a grid plot showing the source in each epoch.

        :param columns: Number of columns to use for the grid plot,
            defaults to 4
        :type columns: float, optional
        :param percentile: The valye passed to the percentile
            normalization function, defaults to 99.9.
        :type percentile: float, optional
        :param zscale: Uses ZScale normalization instead of
            PercentileInterval, defaults to `False`
        :type zscale: bool, optional
        :param contrast: Contast value passed to the ZScaleInterval
            function when zscale is selected, defaults to 0.2.
        :type contrast: float, optional
        :param outfile: Name of the output file, if None then the name
             is automatically generated, defaults to None.
        :type outfile: None or str, optional
        :param save: Save the plot instead of displaying,
            defaults to `False`
        :type save: bool, optional
        :param size: Size of the cutout, defaults to None
        :type size: None or astropy.coordinates.Angle, optional
        :param figsize: Size of the matplotlib.pyplot figure,
            defaults to (10, 5).
        :type figsize: tuple, optional
        :param force: Whether to force the re-fetching
            of the cutout data, defaults to `False`
        :type force: bool, optional
        :param no_selavy: When `True` the selavy overlay
            is hidden, defaults to `False`
        :type no_selavy: bool, optional
        :param disable_autoscaling: Turn off the consistent normalization and
             calculate the normalizations separately for each epoch,
            defaults to `False`
        :type disable_autoscaling: bool, optional
        :param hide_epoch_labels: Turn off the epoch number label (found in
            top left corner of image).
        :type hide_epoch_labels: bool, optional
        :param plot_dpi: Specify the DPI of saved figures, defaults to 150
        :type plot_dpi: int, optional

        :returns: None is save is `True` or the Figure if `False`.
        :rtype: None or matplotlib.pyplot.Figure.
        '''

        if (self._cutouts_got is False) or (force):
            self.get_cutout_data(size)

        num_plots = self.measurements.shape[0]
        nrows = np.ceil(num_plots / columns)

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

    def _gen_overlay_collection(self, cutout_row, f_source=None):
        '''
        Generates the ellipse collection for selavy sources to add
        to the matplotlib axis.

        :param cutout_row: The row containing the selavy data
            to make the ellipses from.
        :type cutout_row: pandas.core.series.Series
        :param f_source: Forced fit extraction to create the
             forced fit ellipse, defaults to None
        :type f_source: pandas.core.frame.DataFrame, optional

        :returns: Tuple of the ellipse collection, patches, and
            the island names
        :rtype: Tuple (matplotlib.collections.PatchCollection.
            matplotlib.patches.Patch, list)
        '''

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
        epoch,
        survey,
        contour_levels=[3., 5., 10., 15.],
        percentile=99.9,
        zscale=False,
        contrast=0.2,
        outfile=None,
        no_colorbar=False,
        title=None,
        save=False,
        size=None,
        force=False,
        plot_dpi=150,
    ):
        '''
        Fetches a FITS file from SkyView of the requested survey at
        the source location and overlays ASKAP contours.

        :param epoch: Epoch requested for the ASKAP contours.
        :type epoch: str
        :param survey: Survey requested to be fetched using SkyView.
        :type survey: str
        :param contour_levels: Contour levels to plot which are multiples
             of the local rms, defaults to [3., 5., 10., 15.].
        :type contour_levels: list , optional
        :param percentile: The valye passed to the percentile
            normalization function, defaults to 99.9.
        :type percentile: float, optional
        :param zscale: Uses ZScale normalization instead of
            PercentileInterval, defaults to `False`
        :type zscale: bool, optional
        :param contrast: Contast value passed to the ZScaleInterval
            function when zscale is selected, defaults to 0.2.
        :type contrast: float, optional
        :param outfile: Name to give the file, if None then the name is
            automatically generated, defaults to None.
        :type outfile: None or str, optional
        :param no_colorbar: Hides the colorbar, defaults to `False`
        :type no_colorbar: bool, optional
        :param title: Plot title, defaults to None
        :type title: None or st, optional
        :param save: Saves the file instead of returing the figure,
            defaults to `False`
        :type save: bool, optional
        :param size: Size of the cutout, defaults to None
        :type size: None or astropy.coordinates.Angle, optional
        :param force: Whether to force the re-fetching
            of the cutout data, defaults to `False`
        :type force: bool, optional
        :param plot_dpi: Specify the DPI of saved figures, defaults to 150
        :type plot_dpi: int, optional

        :returns: None if save is `True` or the figure object if `False`
        :rtype: None or matplotlib.pyplot.Figure
        '''

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

    def _make_png(
            self,
            epoch,
            selavy=True,
            percentile=99.9,
            zscale=False,
            contrast=0.2,
            outfile=None,
            no_islands=True,
            label="Source",
            no_colorbar=False,
            title=None,
            crossmatch_overlay=False,
            hide_beam=False,
            save=False,
            size=None,
            force=False,
            disable_autoscaling=False,
            cutout_data=None,
            norms=None,
            plot_dpi=150
    ):
        '''
        Save a PNG of the image postagestamp

        :param epoch: The requested epoch.
        :type epoch: str
        :param selavy: `True` to overlay selavy components, `False` otherwise
        :type selavy: bool
        :param percentile: The valye passed to the percentile
            normalization function, defaults to 99.9.
        :type percentile: float, optional
        :param zscale: Uses ZScale normalization instead of
            PercentileInterval, defaults to `False`
        :type zscale: bool, optional
        :param contrast: Contast value passed to the ZScaleInterval
            function when zscale is selected, defaults to 0.2.
        :type contrast: float, optional
        :param outfile: Name to give the file, if None then the name is
            automatically generated, defaults to None.
        :type outfile: None or str, optional
         :param no_islands: Disable island lables on the png, defaults to
            `False`
        :type no_islands: bool, optional
        :param label: Figure title (usually the name of the source of
            interest), defaults to "Source"
        :type label: str, optional
        :param no_colorbar: If `True`, do not show the colorbar on the png,
            defaults to `False`
        :type no_colorbar: bool, optional
        :param title: String to set as title,
            defaults to `` where no title will be used.
        :type title: str, optional
        :param crossmatch_overlay: If 'True' then a circle is added to the png
            plot representing the crossmatch radius, defaults to `False`.
        :type crossmatch_overlay: bool, optional
        :param hide_beam: If 'True' then the beam is not plotted onto the png
            plot, defaults to `False`.
        :type hide_beam: bool, optional
        :param save: If `True` the plot is saved rather than the figure being
            returned, defaults to `False`.
        :type save: bool, optional
        :param size: Size of the cutout, defaults to None
        :type size: None or astropy.coordinates.Angle, optional
        :param force: Whether to force the re-fetching
            of the cutout data, defaults to `False`
        :type force: bool, optional
        :param disable_autoscaling: Turn off the consistent normalization and
            calculate the normalizations separately for each epoch,
            defaults to `False`
        :type disable_autoscaling: bool, optional
        :param cutout_data: Pass external cutout_data to be used
            instead of fetching the data, defaults to None.
        :type cutout_data: None or pandas.core.frame.DataFrame
        :param norms: Pass external normalization to be used
            instead of internal calculations.
        :type cutout_data: astropy.visualization.ImageNormalize
        :param plot_dpi: Specify the DPI of saved figures, defaults to 150
        :type plot_dpi: int, optional

        :returns:
        :rtype:
        '''

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
                self.measurements.epoch == epoch
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

        if save:
            plt.savefig(outfile, bbox_inches="tight", dpi=plot_dpi)
            self.logger.debug("Saved {}".format(outfile))

            plt.close(fig)
            return

        else:
            return fig

    def write_ann(
        self, epoch, outfile=None, crossmatch_overlay=False,
        size=None, force=False, cutout_data=None
    ):
        '''
        Write a kvis annotation file containing all selavy sources
        within the image.


        :param epoch: The requested epoch.
        :type epoch: str
        :param outfile: Name of the file to write, defaults to None
        :type outfile: str, optional
        :param crossmatch_overlay: If True, a circle is added to the
            annotation file output denoting the crossmatch radius,
            defaults to False.
        :type crossmatch_overlay: bool, optional.
        :param size: Size of the cutout, defaults to None
        :type size: None or astropy.coordinates.Angle, optional
        :param force: Whether to force the re-fetching
            of the cutout data, defaults to `False`
        :type force: bool, optional
        :param cutout_data: Pass external cutout_data to be used
            instead of fetching the data, defaults to None.
        :type cutout_data: None or pandas.core.frame.DataFrame

        '''
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
            self, epoch, outfile=None, crossmatch_overlay=False,
            size=None, force=False, cutout_data=None
    ):
        '''
        Write a DS9 region file containing all selavy sources within the image

        :param epoch:
        :type epoch:
        :param outfile: Name of the file to write, defaults to None
        :type outfile: str, optional
        :param crossmatch_overlay: If True, a circle is added to the
            annotation file output denoting the crossmatch radius,
            defaults to False.
        :type crossmatch_overlay: bool, optional.
        :param size: Size of the cutout, defaults to None
        :type size: None or astropy.coordinates.Angle, optional
        :param force: Whether to force the re-fetching
            of the cutout data, defaults to `False`
        :type force: bool, optional
        :param cutout_data: Pass external cutout_data to be used
            instead of fetching the data, defaults to None.
        :type cutout_data: None or pandas.core.frame.DataFrame
        '''
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

    def _remove_sbid(self, island):
        '''
        Removes the SBID component of the island name. Takes into account
        negative 'n' label for negative sources.

        :param island: island name.
        :type island: str

        :returns: truncated island name.
        :rtype: str
        '''

        temp = island.split("_")
        new_val = "_".join(temp[-2:])
        if temp[0].startswith("n"):
            new_val = "n" + new_val
        return new_val

    def _create_crosshair_lines(self, target, pixel_buff, length, img_size):
        '''
        Takes the target pixel coordinates and creates the plots
        that are required to plot a 'crosshair' marker. To keep
        the crosshair consistent between scales, the values are
        provided as percentages of the image size.

        :param target: The target in pixel coordinates.
        :type target: `np.array`
        :param pixel_buff: Percentage of image size that is the buffer
            of the crosshair, i.e. the distance between the target and
            beginning of the line.
        :type pixel_buff: float.
        :param length: Size of the line of the crosshair, again as a
            percentage of the image size.
        :type length: float.
        :param img_size: Tuple size of the image array.
        :type img_size: tuple.

        :returns: list of pairs of pixel coordinates to plot using
            scatter.
        :rtype: list.
        '''
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

    def simbad_search(self, radius=Angle(20. * u.arcsec)):
        '''
        Searches SIMBAD for object coordinates and returns matches.

        :param radius: Radius to search, defaults to Angle(20. * u.arcsec)
        :type radius: `astropy.coordinates.Angle`, optional

        :returns: Table of matches or None if no matches
        :rtype: astropy.table.Table or None
        '''

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

    def ned_search(self, radius=Angle(20. * u.arcsec)):
        '''
        Searches NED for object coordinates and returns matches.

        :param radius: Radius to search, defaults to Angle(20. * u.arcsec)
        :type radius: `astropy.coordinates.Angle`, optional

        :returns: Table of matches or None if no matches
        :rtype: astropy.table.Table or None
        '''

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
        radius=Angle(20. * u.arcsec),
        filter_out_unreleased=False,
        show_all=False
    ):
        '''
        Searches NED for object coordinates and returns matches.

        :param radius: Radius to search, defaults to Angle(20. * u.arcsec)
        :type radius: `astropy.coordinates.Angle`, optional
        :param filter_out_unreleased: Remove unreleased data, \
        defaults to `False`
        :type filter_out_unreleased: bool, optional
        :param show_all: Show all available data, defaults to `False`
        :type show_all: bool, optional

        :returns: Table of matches or None if no matches
        :rtype: astropy.table.Table or None
        '''
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
                "Error in performing the NED region search! Error: %s", e
            )
            return None

    def _get_fluxes_and_errors(self, suffix, forced_fits):
        '''
        Selects the correct fluxes, upper limits or forced fits
        to calculate the metrics

        :param suffix: 'peak' or 'int'.
        :type suffix:
        :param forced_fits: Set to `True` if forced fits should be used.
        :type forced_fits: bool

        :returns: The fluxs and errors to use.
        :rtype: Tuple (list, list)
        '''

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

    def calc_eta_metric(self, use_int=False, forced_fits=False):
        '''
        Calculate the eta variability metric

        :param use_int: Calculate using integrated (rather than peak) flux, \
        defaults to `False`
        :type use_int: bool, optional
        :param forced_fits: Use forced fits, defaults to `False`
        :type forced_fits: bool, optional

        :returns: Eta variability metric
        :rtype: float
        '''

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

    def calc_v_metric(self, use_int=False, forced_fits=False):
        '''
        Calculate the V variability metric

        :param use_int: Calculate using integrated (rather than peak) flux, \
        defaults to `False`
        :type use_int: bool, optional
        :param forced_fits: Use forced fits, defaults to `False`
        :type forced_fits: bool, optional

        :returns: V variability metric
        :rtype: float
        '''

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

    def calc_eta_and_v_metrics(self, use_int=False, forced_fits=False):
        '''
        Calculate both variability metrics

        :param use_int: Calculate using integrated (rather than peak) flux, \
        defaults to `False`
        :type use_int: bool, optional
        :param forced_fits: Use forced fits, defaults to `False`
        :type forced_fits: bool, optional

        :returns: Variability metrics
        :rtype: tuple of floats
        '''

        eta = self.calc_eta_metric(use_int=use_int, forced_fits=forced_fits)
        v = self.calc_v_metric(use_int=use_int, forced_fits=forced_fits)

        return eta, v
