# Source class

import numpy as np
import os
import pandas as pd
import warnings

import logging
import logging.handlers
import logging.config

import matplotlib.pyplot as plt

from astropy import units as u
from astropy.nddata.utils import Cutout2D
from astropy.coordinates import SkyCoord
from astropy.io import fits
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

class Source:
    '''
    This is a class representation of a catalogued source position

    :param field: Name of the field containing the source
    :type field: str
    :param sbid: SBID of the field containing the source
    :type sbid: str
    :param SELAVY_FOLDER: Path to selavy directory
    :type SELAVY_FOLDER: str
    :param tiles: `True` if image tiles should be used, \
    `False` for mosaiced images, defaults to `False`
    :type tiles: bool, optional
    :param stokesv: `True` if Stokes V information is requested, \
    `False` for Stokes I, defaults to `False`
    :type stokesv: bool, optional
    '''

    def __init__(
            self,
            field,
            sbid,
            SELAVY_FOLDER,
            tiles=False,
            stokesv=False):
        '''Constructor method
        '''
        self.logger = logging.getLogger('vasttools.survey.Dropbox')
        self.logger.info('Created Source instance')
        
        self.field = field
        self.sbid = sbid

        if tiles:
            selavyname_template = 'selavy-image.i.SB{}.cont.{}.' \
                'linmos.taylor.0.restored.components.txt'
            self.selavyname = selavyname_template.format(self.sbid, self.field)
        else:
            if args.vast_pilot:
                self.selavyname = '{}.selavy.components.txt'.format(self.field)
            else:
                self.selavyname = '{}-selavy.components.txt'.format(self.field)
            if args.stokesv:
                self.nselavyname = 'n{}-selavy.components.txt'.format(
                    self.field)
        self.selavypath = os.path.join(SELAVY_FOLDER, self.selavyname)
        if args.stokesv:
            self.nselavypath = os.path.join(SELAVY_FOLDER, self.nselavyname)

    def make_postagestamp(self, img_data, hdu, wcs, src_coord, size, outfile):
        '''
        Make a FITS postagestamp of the source region and write to file

        :param img_data: Numpy array containing the image data
        :type img_data: `numpy.ndarray`
        :param hdu: FITS header data units of the image
        :type hdu: `astropy.io.fits.hdu.image.PrimaryHDU`
        :param wcs: World Coordinate System of the image
        :type wcs: `astropy.wcs.wcs.WCS`
        :param src_coord: Centre coordinates of the postagestamp
        :type src_coord: `astropy.coordinates.sky_coordinate.SkyCoord`
        :param size: Size of the cutout array along each axis
        :type size: `astropy.coordinates.angles.Angle` \
        or tuple of two `Angle` objects
        :param outfile: Name of output FITS file
        :type outfile: str
        '''

        self.cutout = Cutout2D(
            img_data,
            position=src_coord,
            size=size,
            wcs=wcs)

        # Put the cutout image in the FITS HDU

        hdu_stamp = fits.PrimaryHDU(data=self.cutout.data)

        hdu_stamp.header = hdu.header
        # Update the FITS header with the cutout WCS
        hdu_stamp.header.update(self.cutout.wcs.to_header())

        # Write the cutout to a new FITS file
        hdu_stamp.writeto(outfile, overwrite=True)

    def _empty_selavy(self):
        '''
        Create an empty `DataFrame` for sources that have no selavy match

        :returns: `DataFrame` with column names corresponding to selavy columns
        :rtype: `pandas.core.frame.DataFrame`
        '''

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
            'comment']
        return pd.DataFrame(
            np.array([[np.nan for i in range(len(columns))]]), columns=columns)

    def extract_source(self, src_coord, crossmatch_radius, stokesv):
        '''
        Search for catalogued selavy sources within `crossmatch_radius` of
        `src_coord` and store information of best match

        :param src_coord: Coordinate of the source of interest
        :type src_coord: `astropy.coordinates.sky_coordinate.SkyCoord`
        :param crossmatch_radius: Crossmatch radius to use
        :type crossmatch_radius: `astropy.coordinates.angles.Angle`
        :param stokesv: `True` to crossmatch with Stokes V image, \
        `False` to match with Stokes I image, defaults to `False`
        :type stokesv: bool, optional
        '''

        try:
            self.selavy_cat = pd.read_fwf(self.selavypath, skiprows=[1, ])

            if stokesv:
                nselavy_cat = pd.read_fwf(self.nselavypath, skiprows=[1, ])

                nselavy_cat["island_id"] = [
                    "n{}".format(i) for i in nselavy_cat["island_id"]]
                nselavy_cat["component_id"] = [
                    "n{}".format(i) for i in nselavy_cat["component_id"]]

                self.selavy_cat = self.selavy_cat.append(
                    nselavy_cat, ignore_index=True, sort=False)

        except Exception as e:
            self.logger.warning('{} does not exist'.format(self.selavypath))
            self.selavy_fail = True
            self.selavy_info = self._empty_selavy()
            self.selavy_info["has_match"] = False
            self.has_match = False
            return

        self.selavy_sc = SkyCoord(
            self.selavy_cat['ra_deg_cont'],
            self.selavy_cat['dec_deg_cont'],
            unit=(
                u.deg,
                u.deg))

        match_id, match_sep, _dist = src_coord.match_to_catalog_sky(
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
                "Source in selavy catalogue {} {}, {:.3f}+/-{:.3f} mJy \
                ({:.3f} arcsec offset)".format(
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
            self.selavy_info = self._empty_selavy()

        self.selavy_fail = False
        self.selavy_info["has_match"] = self.has_match

    def write_ann(self, outfile):
        '''
        Write a kvis annotation file containing all selavy sources
        within the image.

        :param outfile: Name of the file to write
        :type outfile: str
        '''

        outfile = outfile.replace(".fits", ".ann")
        neg = False
        with open(outfile, 'w') as f:
            f.write("COORD W\n")
            f.write("PA SKY\n")
            f.write("COLOR GREEN\n")
            f.write("FONT hershey14\n")
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

    def write_reg(self, outfile):
        '''
        Write a DS9 region file containing all selavy sources within the image

        :param outfile: Name of the file to write
        :type outfile: str
        '''

        outfile = outfile.replace(".fits", ".reg")
        with open(outfile, 'w') as f:
            f.write("# Region file format: DS9 version 4.0\n")
            f.write("global color = green font = \"helvetica 10 normal\" "
                    "select = 1 highlite = 1 edit = 1 "
                    "move = 1 delete = 1 include = 1 "
                    "fixed = 0 source = 1\n")
            f.write("fk5\n")
            for i, row in self.selavy_cat_cut.iterrows():
                if row["island_id"].startswith("n"):
                    color = "red"
                else:
                    color = "green"
                ra = row["ra_deg_cont"]
                dec = row["dec_deg_cont"]
                f.write(
                    "ellipse({} {} {} {} {}) # color = {}\n".format(
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
                    "text({} {} \"{}\") # color = {}\n".format(
                        ra, dec, self._remove_sbid(
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

    def filter_selavy_components(self, src_coord, imsize):
        '''
        Create a shortened catalogue by filtering out selavy components
        outside of the image

        :param src_coord: Coordinates of the source of interest
        :type src_coord: `astropy.coordinates.sky_coordinate.SkyCoord`
        :param imsize: Size of the image along each axis
        :type imsize: `astropy.coordinates.angles.Angle` or tuple of two \
        `Angle` objects
        '''

        seps = src_coord.separation(self.selavy_sc)
        mask = seps <= imsize / 1.4
        self.selavy_cat_cut = self.selavy_cat[mask].reset_index(drop=True)

    def make_png(
            self,
            src_coord,
            selavy,
            percentile,
            zscale,
            contrast,
            outfile,
            pa_corr,
            no_islands=False,
            label="Source",
            no_colorbar=False,
            title=""):
        '''
        Save a PNG of the image postagestamp

        :param src_coord: Centre coordinates of the postagestamp
        :type src_coord: `astropy.coordinates.sky_coordinate.SkyCoord`
        :param selavy: `True` to overlay selavy components, `False` otherwise
        :type selavy: bool
        :param percentile: Percentile level for normalisation.
        :type percentile: float
        :param zscale: Use ZScale normalisation instead of linear
        :type zscale: bool
        :param contrast: ZScale contrast to use
        :type contrast: float
        :param outfile: Name of the file to write to, or the name of the FITS \
        file
        :type outfile: str
        :param pa_corr: Correction to apply to ellipse position angle if \
        needed (in deg). Angle is from x-axis from left to right.
        :type pa_corr: float
        :param no_islands: Disable island lables on the png, defaults to \
        `False`
        :type no_islands: bool, optional
        :param label: Figure title (usually the name of the source of \
        interest), defaults to "Source"
        :type label: str, optional
        :param no_colorbar: If `True`, do not show the colorbar on the png, \
        defaults to `False`
        :type no_colorbar: bool, optional
        :param title: String to set as title, \
        defaults to `` where no title will be used.
        :type title: str, optional
        '''

        # image has already been loaded to get the fits
        outfile = outfile.replace(".fits", ".png")
        # convert data to mJy in case colorbar is used.
        cutout_data = self.cutout.data * 1000.
        # create figure
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1, projection=self.cutout.wcs)
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
        ax.scatter([src_coord.ra.deg], [src_coord.dec.deg],
                   transform=ax.get_transform('world'), marker="x",
                   color="r", zorder=10, label=label)
        if selavy and self.selavy_fail is False:
            ax.set_autoscale_on(False)
            # define ellipse properties for clarity, selavy cut will have
            # already been created.
            ww = self.selavy_cat_cut["maj_axis"].astype(float) / 3600.
            hh = self.selavy_cat_cut["min_axis"].astype(float) / 3600.
            aa = self.selavy_cat_cut["pos_ang"].astype(float)
            x = self.selavy_cat_cut["ra_deg_cont"].astype(float)
            y = self.selavy_cat_cut["dec_deg_cont"].astype(float)
            island_names = self.selavy_cat_cut["island_id"].apply(
                self._remove_sbid)
            # Create ellipses, collect them, add to axis.
            # Also where correction is applied to PA to account for how selavy
            # defines it vs matplotlib
            colors = ["C2" if c.startswith(
                "n") else "C1" for c in island_names]
            patches = [Ellipse((x[i], y[i]), ww[i] *
                               1.1, hh[i] *
                               1.1, 90. +
                               (180. -
                                aa[i]) +
                               pa_corr) for i in range(len(x))]
            collection = PatchCollection(
                patches,
                facecolors="None",
                edgecolors=colors,
                linestyle="--",
                lw=2,
                transform=ax.get_transform('world'))
            ax.add_collection(collection, autolim=False)
            # Add island labels, haven't found a better way other than looping
            # at the moment.
            if not no_islands:
                for i, val in enumerate(patches):
                    ax.annotate(
                        island_names[i],
                        val.center,
                        xycoords=ax.get_transform('world'),
                        annotation_clip=True,
                        color="C0",
                        weight="bold")
        else:
            self.logger.warning(
                "PNG: No selavy selected or selavy catalogue failed.")
        ax.legend()
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
        plt.savefig(outfile, bbox_inches="tight")
        self.logger.info("Saved {}".format(outfile))
        plt.close()

    def get_background_rms(self, rms_img_data, rms_wcs, src_coord):
        pix_coord = np.rint(skycoord_to_pixel(src_coord, rms_wcs)).astype(int)
        rms_val = rms_img_data[pix_coord[0], pix_coord[1]]
        try:
            self.selavy_info['BANE_rms'] = rms_val
        except Exception as e:
            self.selavy_info = self._empty_selavy()
            self.selavy_info['BANE_rms'] = rms_val
