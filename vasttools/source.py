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

    :param field: Name of the field containing the source
    :type field: str
    :param src_coord: Source coordinates
    :type src_coord: `astropy.coordinates.sky_coordinate.SkyCoord`
    :param sbid: SBID of the field containing the source
    :type sbid: str
    :param SELAVY_FOLDER: Path to selavy directory
    :type SELAVY_FOLDER: str
    :param vast_pilot: Survey epoch
    :type vast_pilot: str
    :param tiles: `True` if image tiles should be used,
        `False` for mosaiced images, defaults to `False`
    :type tiles: bool, optional
    :param stokes: Stokes parameter to query, defaults to "I"
    :type stokes: str, optional
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
        image_type = "COMBINED",
        islands=False,
        outdir=".",
        planet=False
        ):
        '''Constructor method
        '''
        self.logger = logging.getLogger('vasttools.source.Source')
        self.logger.debug('Created Source instance')

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

        self.detections = self.measurements[
            self.measurements.detection == True
        ].shape[0]

        self.limits = self.measurements[
            self.measurements.detection == False
        ].shape[0]

        self._cutouts_got = False

        self.norms = None
        self.checked_norms = False

        self.planet = planet


    def write_measurements(self, simple=False, outfile=None):
        """
        Write the measurements to a CSV file.
        """

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

            measurements_to_write = self.measurements[[cols]]

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
                        grid=False, yaxis_start="auto", peak_flux=True,
                        save=False, outfile=None):
        '''
        Plot source lightcurves and save to file

        :param sigma_thresh: Threshold to use for upper limits, defaults to 5
        :type sigma_thresh: int or float
        :param savefile: Filename to save plot, defaults to None
        :type savefile: str
        :param min_points: Minimum number of points for plotting,
            defaults to 2
        :type min_points: int, optional
        :param min_detections: Minimum number of detections for plotting, \
        defaults to 1
        :type min_detections: int, optional
        :param mjd: Plot x-axis in MJD rather than datetime, defaults to False
        :type mjd: bool, optional
        :param grid: Turn on matplotlib grid, defaults to False
        :type grid: bool, optional
        '''
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
        measurements = self.measurements[
            self.measurements['rms_image'] != -99
        ]

        if measurements.empty:
            self.logger.warning(
                "%s has no measurements! No lightcurve will be produced.",
                self.name
            )
            return

        plot_dates = measurements['dateobs']
        if mjd:
            plot_dates = Time(plot_dates.to_numpy()).mjd

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
        ax.set_ylabel(label)

        self.logger.debug("Plotting upper limit")
        upper_lim_mask = measurements.detection==False
        upper_lims = measurements[
            upper_lim_mask
        ]

        upperlim_points = ax.errorbar(
            plot_dates[upper_lim_mask],
            sigma_thresh *
            upper_lims['rms_image'],
            yerr=upper_lims['rms_image'],
            uplims=True,
            lolims=False,
            marker='_',
            c='k',
            linestyle="none")

        self.logger.debug("Plotting detection")
        detections = measurements[
            ~upper_lim_mask
        ]

        detection_points = ax.errorbar(
            plot_dates[~upper_lim_mask],
            detections[flux_col],
            yerr=detections['rms_image'],
            marker='o',
            c='k',
            linestyle="none")

        if yaxis_start == "0":
            max_y = np.nanmax(
                detections[flux_col].tolist() +
                (sigma_thresh * upper_lims['rms_image']).tolist()
            )
            ax.set_ylim(
                bottom=0,
                top=max_y * 1.1
            )

        if mjd:
            ax.set_xlabel('Date (MJD)')
        else:
            fig.autofmt_xdate()
            ax.set_xlabel('Date')

            date_form = mdates.DateFormatter("%Y-%m-%d")
            ax.xaxis.set_major_formatter(date_form)
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))

        ax.grid(grid)

        if save:
            if outfile is None:
                outfile = "{}_lc.png".format(self.name.replace(
                    " ", "_"
                ))

            elif not outfile.endswith(".png"):
                outname+=".png"

            if self.outdir != ".":
                outfile = os.path.join(
                    self.outdir,
                    outfile
                )

            plt.savefig(outfile, bbox_inches='tight')
            plt.close()

            return

        else:

            return fig


    def get_cutout_data(self, size=None):
        '''
        Make a FITS postagestamp of the source region and write to file

        :param img_data: Numpy array containing the image data
        :type img_data: `numpy.ndarray`
        :param header: FITS header data units of the image
        :type header: `astropy.io.fits.header.Header`
        :param wcs: World Coordinate System of the image
        :type wcs: `astropy.wcs.wcs.WCS`
        :param size: Size of the cutout array along each axis
        :type size: `astropy.coordinates.angles.Angle`
            or tuple of two `Angle` objects
        :param outfile: Name of output FITS file
        :type outfile: str
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


    def analyse_norm_level(self, percentile=99.9,
        zscale=False, z_contrast=0.2):
        if not self._cutouts_got:
            self.logger.warning(
                "Fetch cutout data before running this function!"
            )

        if self.detections > 0:
            scale_index = self.measurements[
                self.measurements.detection == True
            ].index.values[0]
        else:
            scale_index = 0

        scale_data = self.cutout_df.loc[scale_index].data * 1.e3

        if zscale:
            self.norms = ImageNormalize(
                scale_data, interval=ZScaleInterval(
                    contrast=contrast))
        else:
            self.norms = ImageNormalize(
                scale_data,
                interval=PercentileInterval(percentile),
                stretch=LinearStretch())

        self.checked_norms = True


    def _get_cutout(self, row, size=Angle(5. * u.arcmin)):

        image = Image(row.field, row.epoch, self.stokes, self.base_folder)

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
            force=False
        ):
        """
        Wrapper for make_png to make nicer interactive function.
        No access to save.
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
            outfile=None
        ):
        """
        Wrapper for make_png to make nicer interactive function.
        Always save.
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
            save=True
        )

        return


    def save_fits_cutout(self, epoch, outfile=None, size=None, force=False):
        if (self._cutouts_got is False) or (force == True):
            self.get_cutout_data(size)

        if epoch not in self.epochs:
            raise ValueError(
                "This source does not contain Epoch {}!".format(epoch)
            )

            return

        if outfile is None:
            outfile = "{}_EPOCH{}.fits".format(
                self.name.replace(" ", "_"),
                RELEASED_EPOCHS[epoch]
            )
        if self.outdir != ".":
            outfile = os.path.join(
                self.outdir,
                outfile
            )

        index = self.epochs.index(epoch)

        cutout_data = self.cutout_df.iloc[index]

        hdu_stamp = fits.PrimaryHDU(
            data=cutout_data.data,
            header=cutout_data.header
        )

        # Write the cutout to a new FITS file
        hdu_stamp.writeto(outfile, overwrite=True)


    def save_png_cutout(self, epoch):
        fig = self.make_png(epoch)

        return fig


    def save_all_ann(self, crossmatch_overlay=False):
        self.measurements['epoch'].apply(
            self.write_ann,
            args=(
                None,
                crossmatch_overlay
            )
        )


    def save_all_reg(self, crossmatch_overlay=False):
        self.measurements['epoch'].apply(
            self.write_reg,
            args=(
                None,
                crossmatch_overlay
            )
        )


    def save_all_fits_cutouts(self, size=None, force=False):
        if (self._cutouts_got is False) or (force == True):
            self.get_cutout_data(size)

        for e in self.measurements['epoch']:
            self.save_fits_cutout(e)


    def save_all_png_cutouts(
        self,
        selavy=True,
        percentile=99.9,
        zscale=False,
        contrast=0.2,
        islands=True,
        no_colorbar=False,
        crossmatch_overlay=False,
        hide_beam=False,
        size=None
    ):
        if self._cutouts_got is False:
            self.get_cutout_data(size)

        if not self.checked_norms:
            self.analyse_norm_level()

        self.measurements['epoch'].apply(
            self.make_png,
            args = (
                selavy,
                percentile,
                zscale,
                contrast,
                None,
                islands,
                "Source",
                no_colorbar,
                None,
                crossmatch_overlay,
                hide_beam,
                True
            )
        )
        # plt.close(fig)


    def plot_all_cutouts(self, columns=4, percentile=99.9, zscale=False,
        contrast=0.1,outfile=None, save=False, size=None, figsize=(10, 5),
        force=False, no_selavy=False
        ):
        if (self._cutouts_got is False) or (force == True):
            self.get_cutout_data(size)

        num_plots = self.measurements.shape[0]
        nrows = np.ceil(num_plots/columns)

        fig = plt.figure(figsize=figsize)

        fig.tight_layout()

        plots = {}

        if not self.checked_norms:
            if self.detections > 0:
                scale_index = self.measurements[
                    self.measurements.detection == True
                ].index.values[0]
            else:
                scale_index = 0

            scale_data = self.cutout_df.loc[scale_index].data

            if zscale:
                img_norms = ImageNormalize(
                    scale_data, interval=ZScaleInterval(
                        contrast=contrast))
            else:
                img_norms = ImageNormalize(
                    scale_data,
                    interval=PercentileInterval(percentile),
                    stretch=LinearStretch())
        else:
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
            i+=1
            plots[i] = fig.add_subplot(
                nrows,
                columns,
                i,
                projection=cutout_row.wcs
            )

            im = plots[i].imshow(
                cutout_row.data, norm=img_norms, cmap="gray_r"
            )

            plots[i].set_title('Epoch {}'.format(
                measurement_row.epoch
            ))

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
                outfile = "{}_EPOCH{}.png".format(
                    self.name.replace(" ", "_"),
                    RELEASED_EPOCHS[epoch]
                )

            if self.outdir != ".":
                outfile = os.path.join(
                    self.outdir,
                    outfile
                )

            plt.savefig(outfile, bbox_inches=True)

            plt.close()

            return

        else:

            return fig


    def _gen_overlay_collection(self, cutout_row):
        wcs = cutout_row.wcs
        selavy_sources = cutout_row.selavy_overlay
        pix_scale = proj_plane_pixel_scales(wcs)
        sx = pix_scale[0]
        sy = pix_scale[1]
        degrees_per_pixel = np.sqrt(sx * sy)

        # define ellipse properties for clarity, selavy cut will have
        # already been created.
        ww = selavy_sources["maj_axis"].astype(float) / 3600.
        hh = selavy_sources["min_axis"].astype(float) / 3600.
        ww /= degrees_per_pixel
        hh /= degrees_per_pixel
        aa = selavy_sources["pos_ang"].astype(float)
        x = selavy_sources["ra_deg_cont"].astype(float)
        y = selavy_sources["dec_deg_cont"].astype(float)

        coordinates = np.column_stack((x, y))

        coordinates = wcs.wcs_world2pix(coordinates, 0)

        island_names = selavy_sources["island_id"].apply(
            self._remove_sbid
        )
        # Create ellipses, collect them, add to axis.
        # Also where correction is applied to PA to account for how selavy
        # defines it vs matplotlib
        colors = ["C2" if c.startswith(
            "n") else "C1" for c in island_names]
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
    ):
        """docstring for skyview_contour_plot"""
        if (self._cutouts_got is False) or (force == True):
            self.get_cutout_data(size)

        size = self._size

        if epoch not in self.epochs:
            raise ValueError(
                "This source does not contain Epoch {}!".format(epoch)
            )

            return

        if outfile is None:
            outfile = "{}_EPOCH{}.png".format(
                self.name.replace(" ", "_"),
                RELEASED_EPOCHS[epoch]
            )

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

        except:
            warnings.warn("SkyView fetch failed!")
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
            plt.savefig(outfile, bbox_inches="tight")
            self.logger.info("Saved {}".format(outfile))

            plt.close(fig)

            return
        else:
            return fig


    def make_png(
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
    ):
        '''
        Save a PNG of the image postagestamp

        :param selavy: `True` to overlay selavy components, `False` otherwise
        :type selavy: bool
        :param percentile: Percentile level for normalisation.
        :type percentile: float
        :param zscale: Use ZScale normalisation instead of linear
        :type zscale: bool
        :param contrast: ZScale contrast to use
        :type contrast: float
        :param outfile: Name of the file to write to, or the name of the FITS
            file
        :type outfile: str
        :param img_beam: Object containing the beam information of the image,
            from which the source is being plotted from.
        :type img_beam: radio_beam.Beam
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
        '''
        if (self._cutouts_got is False) or (force == True):
            self.get_cutout_data(size)

        if epoch not in self.epochs:
            raise ValueError(
                "This source does not contain Epoch {}!".format(epoch)
            )

            return

        if outfile is None:
            outfile = "{}_EPOCH{}.png".format(
                self.name.replace(" ", "_"),
                RELEASED_EPOCHS[epoch]
            )

        if self.outdir != ".":
            outfile = os.path.join(
                self.outdir,
                outfile
            )

        index = self.epochs.index(epoch)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection=self.cutout_df.iloc[index].wcs)
        # Get the Image Normalisation from zscale, user contrast.
        if not self.checked_norms:
            if zscale:
                img_norms = ImageNormalize(
                    self.cutout_df.iloc[index].data * 1.e3,
                    interval=ZScaleInterval(
                        contrast=contrast
                    ))
            else:
                img_norms = ImageNormalize(
                    self.cutout_df.iloc[index].data * 1.e3,
                    interval=PercentileInterval(percentile),
                    stretch=LinearStretch())
        else:
            img_norms = self.norms

        im = ax.imshow(
            self.cutout_df.iloc[index].data * 1.e3,
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

        target_coords = self.cutout_df.iloc[index].wcs.wcs_world2pix(
            target_coords, 0
        )

        crosshair_lines = self._create_crosshair_lines(
            target_coords,
            0.03,
            0.03,
            self.cutout_df.iloc[index].data.shape
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
                        self.measurements.iloc[index].ra,
                        self.measurements.iloc[index].dec
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

        if (not self.cutout_df.iloc[index]['selavy_overlay'].empty) and selavy:
            ax.set_autoscale_on(False)
            collection, patches, island_names = self._gen_overlay_collection(
                self.cutout_df.iloc[index]
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
            self.logger.warning(
                "PNG: No selavy selected or selavy catalogue failed."
            )

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

        if self.cutout_df.iloc[index].beam is not None and hide_beam is False:
            img_beam = self.cutout_df.iloc[index].beam
            if self.cutout_df.iloc[index].wcs.is_celestial:
                major = img_beam.major.value
                minor = img_beam.minor.value
                pa = img_beam.pa.value
                pix_scale = proj_plane_pixel_scales(
                    self.cutout_df.iloc[index].wcs
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
            plt.savefig(outfile, bbox_inches="tight")
            self.logger.info("Saved {}".format(outfile))

            plt.close(fig)

            return
        else:
            return fig


    def write_ann(self, epoch, outfile=None, crossmatch_overlay=False,
        size=None, force=False):
        '''
        Write a kvis annotation file containing all selavy sources
        within the image.

        :param outfile: Name of the file to write
        :type outfile: str
        :param crossmatch_overlay: If True, a circle is added to the
            annotation file output denoting the crossmatch radius,
            defaults to False.
        :type crossmatch_overlay: bool, optional.
        '''
        if (self._cutouts_got is False) or (force == True):
            self.get_cutout_data(size)

        if outfile is None:
            outfile = "{}_EPOCH{}.ann".format(
                self.name.replace(" ", "_"),
                RELEASED_EPOCHS[epoch]
            )
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
                self.measurements.iloc[index].ra.deg,
                self.measurements.iloc[index].ra.deg,
                3./3600.
            ))
            if crossmatch_overlay:
                try:
                    f.write("CIRCLE {} {} {}\n".format(
                        self.measurements.iloc[index].ra.deg,
                        self.measurements.iloc[index].dec.deg,
                        self.crossmatch_radius.deg
                    ))
                except Exception as e:
                    self.logger.warning(
                        "Crossmatch circle overlay failed!"
                        " Has the source been crossmatched?")
            f.write("COLOR GREEN\n")


            selavy_cat_cut = self.cutout_df.iloc[index].selavy_overlay

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

        self.logger.info("Wrote annotation file {}.".format(outfile))


    def write_reg(self, epoch, outfile=None, crossmatch_overlay=False,
        size=None, force=False):
        '''
        Write a DS9 region file containing all selavy sources within the image

        :param outfile: Name of the file to write
        :type outfile: str
        :param crossmatch_overlay: If True, a circle is added to the region
            file output denoting the crossmatch radius, defaults to False.
        :type crossmatch_overlay: bool, optional.
        '''
        if (self._cutouts_got is False) or (force == True):
            self.get_cutout_data(size)

        if outfile is None:
            outfile = "{}_EPOCH{}.reg".format(
                self.name.replace(" ", "_"),
                RELEASED_EPOCHS[epoch]
            )
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
                    self.measurements.iloc[index].ra.deg,
                    self.measurements.iloc[index].dec.deg,
                ))
            if crossmatch_overlay:
                try:
                    f.write("circle({} {} {}) # color=blue\n".format(
                        self.measurements.iloc[index].ra.deg,
                        self.measurements.iloc[index].dec.deg,
                        self.crossmatch_radius.deg
                    ))
                except Exception as e:
                    self.logger.warning(
                        "Crossmatch circle overlay failed!"
                        " Has the source been crossmatched?")

            selavy_cat_cut = self.cutout_df.iloc[index].selavy_overlay

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
                        ra-(10./3600.), dec, self._remove_sbid(
                            row["island_id"]), color))

        self.logger.info("Wrote region file {}.".format(outfile))


    def save_measurements(
        self,
        out_dir=None,
        simple=False,
        detections_only=False
    ):

        if simple:
            to_write = self.measurements[[
                'name',
                'field',
                'epoch',
                'dateobs',
                'ra_deg_cont',
                'dec_deg_cont',
                'flux_peak',
                'flux_peak_err',
                'flux_int',
                'flux_int_err',
                'rms_image',
                'detection'
            ]].sort_values(
                by='dateobs'
            )
        else:
            to_write = self.measurements.drop(['skycoord']).sort_values(
                by='dateobs'
            )

        if detections_only:
            to_write = to_write[to_write.detection == True]

        outname = "{}_results.csv".format(
            self.name.replace(" ", "_")
        )

        if out_dir is not None:
            outname = os.join.path(
                out_dir,
                outname
            )

        to_write.to_csv(outname, index=False)


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
        :type np.array: array of the pixel values of the target
            returned by wcs.wcs_world2pix.
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

        plots.append([[x, x], [y+pixel_buff, y+pixel_buff+length]])
        plots.append([[x, x], [y-pixel_buff, y-pixel_buff-length]])
        plots.append([[x+pixel_buff, x+pixel_buff+length], [y, y]])
        plots.append([[x-pixel_buff, x-pixel_buff-length], [y, y]])

        return plots
