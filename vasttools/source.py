# Source class

import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.axes as maxes
from astropy.visualization import LinearStretch
from astropy.visualization import AsymmetricPercentileInterval
from astropy.visualization import PercentileInterval
from astropy.visualization import ZScaleInterval, ImageNormalize
from mpl_toolkits.axes_grid1.anchored_artists import (AnchoredEllipse,
                                                      AnchoredSizeBar)
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
import matplotlib.pyplot as plt
import logging.config
import logging.handlers
import logging
import warnings
import pandas as pd
import os
import numpy as np

import vasttools.crosshair

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
    :param stokesv: `True` if Stokes V information is requested,
        `False` for Stokes I, defaults to `False`
    :type stokesv: bool, optional
    '''

    def __init__(
            self,
            field,
            src_coord,
            sbid,
            SELAVY_FOLDER,
            vast_pilot,
            tiles=False,
            stokesv=False,
            islands=False):
        '''Constructor method
        '''
        self.logger = logging.getLogger('vasttools.source.Source')
        self.logger.debug('Created Source instance')

        self.src_coord = src_coord
        self.field = field
        self.sbid = sbid

        if islands:
            self.cat_type = "islands"
            self.islands = True
        else:
            self.cat_type = "components"
            self.islands = False

        if tiles:
            selavyname_template = 'selavy-image.i.SB{}.cont.{}.' \
                'linmos.taylor.0.restored.{}.txt'
            self.selavyname = selavyname_template.format(
                self.sbid, self.field, self.cat_type
            )
        else:
            if vast_pilot == "0":
                self.selavyname = '{}.taylor.0.{}.txt'.format(
                    self.field, self.cat_type)
            else:
                self.selavyname = '{}.selavy.{}.txt'.format(
                    self.field, self.cat_type
                )

            self.nselavyname = 'n{}'.format(self.selavyname)

        self.selavypath = os.path.join(SELAVY_FOLDER, self.selavyname)
        if stokesv:
            self.nselavypath = os.path.join(SELAVY_FOLDER, self.nselavyname)

    def make_postagestamp(self, img_data, header, wcs, size, outfile):
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

        self.cutout = Cutout2D(
            img_data,
            position=self.src_coord,
            size=size,
            wcs=wcs)
        self.logger.debug(type(header))
        # Put the cutout image in the FITS HDU

        hdu_stamp = fits.PrimaryHDU(data=self.cutout.data)

        hdu_stamp.header = header
        # Update the FITS header with the cutout WCS
        hdu_stamp.header.update(self.cutout.wcs.to_header())

        # Write the cutout to a new FITS file
        hdu_stamp.writeto(outfile, overwrite=True)

    def _empty_selavy(self, islands=False):
        '''
        Create an empty `DataFrame` for sources that have no selavy match

        :returns: `DataFrame` with column names corresponding to selavy columns
        :rtype: `pandas.core.frame.DataFrame`
        '''

        if islands:
            columns = [
                "#",
                "island_id",
                "island_name",
                "n_components",
                "ra_hms_cont",
                "dec_dms_cont",
                "ra_deg_cont",
                "dec_deg_cont",
                "freq",
                "maj_axis",
                "min_axis",
                "pos_ang",
                "flux_int",
                "flux_int_err",
                "flux_peak",
                "mean_background",
                "background_noise",
                "max_residual",
                "min_residual",
                "mean_residual",
                "rms_residual",
                "stdev_residual",
                "x_min",
                "x_max",
                "y_min",
                "y_max",
                "n_pix",
                "solid_angle",
                "beam_area",
                "x_ave",
                "y_ave",
                "x_cen",
                "y_cen",
                "x_peak",
                "y_peak",
                "flag_i1",
                "flag_i2",
                "flag_i3",
                "flag_i4"
            ]
        else:
            columns = [
                '#',
                'island_id',
                'component_id',
                'component_name',
                'ra_hms_cont',
                'dec_dms_cont',
                'ra_deg_cont',
                'dec_deg_cont',
                'ra_err',
                'dec_err',
                'freq',
                'flux_peak',
                'flux_peak_err',
                'flux_int',
                'flux_int_err',
                'maj_axis',
                'min_axis',
                'pos_ang',
                'maj_axis_err',
                'min_axis_err',
                'pos_ang_err',
                'maj_axis_deconv',
                'min_axis_deconv',
                'pos_ang_deconv',
                'maj_axis_deconv_err',
                'min_axis_deconv_err',
                'pos_ang_deconv_err',
                'chi_squared_fit',
                'rms_fit_gauss',
                'spectral_index',
                'spectral_curvature',
                'spectral_index_err',
                'spectral_curvature_err',
                'rms_image',
                'has_siblings',
                'fit_is_estimate',
                'spectral_index_from_TT',
                'flag_c4',
                'comment'
            ]
        return pd.DataFrame(
            np.array(
                [[np.nan for i in range(len(columns))]]), columns=columns
            )

    def extract_source(self, crossmatch_radius, stokesv):
        '''
        Search for catalogued selavy sources within `crossmatch_radius` of
        `self.src_coord` and store information of best match

        :param crossmatch_radius: Crossmatch radius to use
        :type crossmatch_radius: `astropy.coordinates.angles.Angle`
        :param stokesv: `True` to crossmatch with Stokes V image,
            `False` to match with Stokes I image, defaults to `False`
        :type stokesv: bool, optional
        '''

        self.crossmatch_radius = crossmatch_radius

        try:
            self.selavy_cat = pd.read_fwf(self.selavypath, skiprows=[1, ])

            if stokesv:
                nselavy_cat = pd.read_fwf(self.nselavypath, skiprows=[1, ])

                nselavy_cat["island_id"] = [
                    "n{}".format(i) for i in nselavy_cat["island_id"]]
                if not self.islands:
                    nselavy_cat["component_id"] = [
                        "n{}".format(i) for i in nselavy_cat["component_id"]]

                self.selavy_cat = self.selavy_cat.append(
                    nselavy_cat, ignore_index=True, sort=False)

        except Exception as e:
            self.logger.warning('{} does not exist'.format(self.selavypath))
            self.selavy_fail = True
            self.selavy_info = self._empty_selavy(islands=self.islands)
            self.selavy_info["has_match"] = False
            self.has_match = False
            return

        self.selavy_sc = SkyCoord(
            self.selavy_cat['ra_deg_cont'],
            self.selavy_cat['dec_deg_cont'],
            unit=(
                u.deg,
                u.deg))

        match_id, match_sep, _dist = self.src_coord.match_to_catalog_sky(
            self.selavy_sc)

        if match_sep < crossmatch_radius:
            self.has_match = True
            selavy_index = self.selavy_cat.index.isin([match_id])
            self.selavy_info = self.selavy_cat[selavy_index].copy()

            selavy_ra = self.selavy_info['ra_hms_cont'].iloc[0]
            selavy_dec = self.selavy_info['dec_dms_cont'].iloc[0]

            selavy_iflux = self.selavy_info['flux_int'].iloc[0]
            selavy_iflux_err = self.selavy_info['flux_int_err'].iloc[0]
            self.logger.info(
                "Source in selavy catalogue {} {}, {:.3f}+/-{:.3f} mJy "
                "({:.3f} arcsec offset)".format(
                    selavy_ra,
                    selavy_dec,
                    selavy_iflux,
                    selavy_iflux_err,
                    match_sep[0].arcsec))
        else:
            self.logger.info(("No selavy catalogue match. "
                              "Nearest source {:.0f} arcsec away."
                              ).format(match_sep[0].arcsec))
            self.has_match = False
            self.selavy_info = self._empty_selavy(islands=self.islands)

        self.selavy_fail = False
        self.selavy_info["has_match"] = self.has_match

    def write_ann(self, outfile, crossmatch_overlay=False):
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

        outfile = outfile.replace(".fits", ".ann")
        neg = False
        with open(outfile, 'w') as f:
            f.write("COORD W\n")
            f.write("PA SKY\n")
            f.write("FONT hershey14\n")
            f.write("COLOR BLUE\n")
            f.write("CROSS {0} {1} {2} {2}\n".format(
                self.src_coord.ra.deg,
                self.src_coord.dec.deg,
                3./3600.
            ))
            if crossmatch_overlay:
                try:
                    f.write("CIRCLE {} {} {}\n".format(
                        self.src_coord.ra.deg,
                        self.src_coord.dec.deg,
                        self.crossmatch_radius.deg
                    ))
                except Exception as e:
                    self.logger.warning(
                        "Crossmatch circle overlay failed!"
                        " Has the source been crossmatched?")
            f.write("COLOR GREEN\n")
            for i, row in self.selavy_cat_cut.iterrows():
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

    def write_reg(self, outfile, crossmatch_overlay=False):
        '''
        Write a DS9 region file containing all selavy sources within the image

        :param outfile: Name of the file to write
        :type outfile: str
        :param crossmatch_overlay: If True, a circle is added to the region
            file output denoting the crossmatch radius, defaults to False.
        :type crossmatch_overlay: bool, optional.
        '''

        outfile = outfile.replace(".fits", ".reg")
        with open(outfile, 'w') as f:
            f.write("# Region file format: DS9 version 4.0\n")
            f.write("global color=green font=\"helvetica 10 normal\" "
                    "select=1 highlite=1 edit=1 "
                    "move=1 delete=1 include=1 "
                    "fixed=0 source=1\n")
            f.write("fk5\n")
            f.write(
                "point({} {}) # point=x color=blue\n".format(
                    self.src_coord.ra.deg,
                    self.src_coord.dec.deg,
                ))
            if crossmatch_overlay:
                try:
                    f.write("circle({} {} {}) # color=blue\n".format(
                        self.src_coord.ra.deg,
                        self.src_coord.dec.deg,
                        self.crossmatch_radius.deg
                    ))
                except Exception as e:
                    self.logger.warning(
                        "Crossmatch circle overlay failed!"
                        " Has the source been crossmatched?")
            for i, row in self.selavy_cat_cut.iterrows():
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

    def filter_selavy_components(self, imsize):
        '''
        Create a shortened catalogue by filtering out selavy components
        outside of the image

        :param imsize: Size of the image along each axis
        :type imsize: `astropy.coordinates.angles.Angle` or tuple of two
            `Angle` objects
        '''

        seps = self.src_coord.separation(self.selavy_sc)
        mask = seps <= imsize / 1.4
        self.selavy_cat_cut = self.selavy_cat[mask].reset_index(drop=True)


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


    def make_png(
            self,
            selavy,
            percentile,
            zscale,
            contrast,
            outfile,
            img_beam,
            no_islands=False,
            label="Source",
            no_colorbar=False,
            title="",
            crossmatch_overlay=False,
            hide_beam=False):
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

        # image has already been loaded to get the fits
        outfile = outfile.replace(".fits", ".png")
        # convert data to mJy in case colorbar is used.
        cutout_data = self.cutout.data * 1000.
        cutout_wcs = self.cutout.wcs
        # create figure
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1, projection=cutout_wcs)
        # Get the Image Normalisation from zscale, user contrast.
        if zscale:
            self.img_norms = ImageNormalize(
                cutout_data, interval=ZScaleInterval(
                    contrast=contrast))
        else:
            self.img_norms = ImageNormalize(
                cutout_data,
                interval=PercentileInterval(percentile),
                stretch=LinearStretch())
        im = ax.imshow(cutout_data, norm=self.img_norms, cmap="gray_r")
        # insert crosshair of target
        target_coords = np.array(
            ([[self.src_coord.ra.deg, self.src_coord.dec.deg]])
        )
        target_coords = cutout_wcs.wcs_world2pix(target_coords, 0)
        crosshair_lines  = self._create_crosshair_lines(
            target_coords,
            0.03,
            0.03,
            cutout_data.shape
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
                    (self.src_coord.ra, self.src_coord.dec),
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
        if selavy and self.selavy_fail is False:
            pix_scale = proj_plane_pixel_scales(cutout_wcs)
            sx = pix_scale[0]
            sy = pix_scale[1]
            degrees_per_pixel = np.sqrt(sx * sy)
            ax.set_autoscale_on(False)
            # define ellipse properties for clarity, selavy cut will have
            # already been created.
            ww = self.selavy_cat_cut["maj_axis"].astype(float) / 3600.
            hh = self.selavy_cat_cut["min_axis"].astype(float) / 3600.
            ww /= degrees_per_pixel
            hh /= degrees_per_pixel
            aa = self.selavy_cat_cut["pos_ang"].astype(float)
            x = self.selavy_cat_cut["ra_deg_cont"].astype(float)
            y = self.selavy_cat_cut["dec_deg_cont"].astype(float)

            coordinates = np.column_stack((x, y))

            coordinates = cutout_wcs.wcs_world2pix(coordinates, 0)

            island_names = self.selavy_cat_cut["island_id"].apply(
                self._remove_sbid)
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
            ax.add_collection(collection, autolim=False)
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
                "PNG: No selavy selected or selavy catalogue failed.")
        legend_elements = [
            Line2D(
                [0], [0], marker='c', color='C3', label=label,
                markerfacecolor='g', ls="none", markersize=8)]
        if selavy and self.selavy_fail is False:
            legend_elements.append(
                Line2D(
                    [0], [0], marker='o', color='C1',
                    label="Selavy {}".format(self.cat_type),
                    markerfacecolor='none', ls="none", markersize=10)
            )
        if crossmatch_overlay:
            legend_elements.append(
                Line2D(
                    [0], [0], marker='o', color='C4',
                    label="Crossmatch radius ({:.1f} arcsec)".format(
                        self.crossmatch_radius.arcsec),
                    markerfacecolor='none', ls="none",
                    markersize=10)
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
        if title != "":
            ax.set_title(title)
        if img_beam is not None and hide_beam is False:
            if cutout_wcs.is_celestial:
                major = img_beam.major.value
                minor = img_beam.minor.value
                pa = img_beam.pa.value
                pix_scale = proj_plane_pixel_scales(cutout_wcs)
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

        plt.savefig(outfile, bbox_inches="tight")
        self.logger.info("Saved {}".format(outfile))
        plt.close()

    def get_background_rms(self, rms_img_data, rms_wcs):
        '''
        Get the background noise from the RMS image

        :param rms_img_data: Numpy array containing the RMS image data
        :type rms_img_data: `numpy.ndarray`
        :param rms_wcs: World Coordinate System of the image
        :type rms_wcs: `astropy.wcs.wcs.WCS`
        '''

        pix_coord = np.rint(skycoord_to_pixel(
            self.src_coord, rms_wcs)).astype(int)
        rms_val = rms_img_data[pix_coord[0], pix_coord[1]] * 1e3
        try:
            self.selavy_info['SELAVY_rms'] = rms_val
        except Exception as e:
            self.selavy_info = self._empty_selavy(islands=self.islands)
            self.selavy_info['SELAVY_rms'] = rms_val
