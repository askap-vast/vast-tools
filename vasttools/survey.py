# Functions and classes related to loading and searching survey data

import sys
import os
import pandas as pd
import warnings
import pkg_resources
import itertools
import numpy as np
import re

import logging
import logging.handlers
import logging.config

from astropy.coordinates import Angle
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from astropy.utils.exceptions import AstropyWarning, AstropyDeprecationWarning
from radio_beam import Beam

warnings.filterwarnings('ignore', category=AstropyWarning, append=True)
warnings.filterwarnings('ignore',
                        category=AstropyDeprecationWarning, append=True)


RELEASED_EPOCHS = {
    "0": "00",  # RACS, needs check that it exists, not part of VAST release
    "1": "01",
    "2": "02",
    "3x": "03x",
    "4x": "04x",
    "5x": "05x",
    "6x": "06x",
    "7x": "07x",
    "8": "08",
    "9": "09",
    "10x": "10x",
    "11x": "11x"
}

FIELD_FILES = {
    "0": pkg_resources.resource_filename(
        __name__, "./data/csvs/racs_info.csv"),
    "1": pkg_resources.resource_filename(
        __name__, "./data/csvs/vast_epoch01_info.csv"),
    "2": pkg_resources.resource_filename(
        __name__, "./data/csvs/vast_epoch02_info.csv"),
    "3x": pkg_resources.resource_filename(
        __name__, "./data/csvs/vast_epoch03x_info.csv"),
    "4x": pkg_resources.resource_filename(
        __name__, "./data/csvs/vast_epoch04x_info.csv"),
    "5x": pkg_resources.resource_filename(
        __name__, "./data/csvs/vast_epoch05x_info.csv"),
    "6x": pkg_resources.resource_filename(
        __name__, "./data/csvs/vast_epoch06x_info.csv"),
    "7x": pkg_resources.resource_filename(
        __name__, "./data/csvs/vast_epoch07x_info.csv"),
    "8": pkg_resources.resource_filename(
        __name__, "./data/csvs/vast_epoch08_info.csv"),
    "9": pkg_resources.resource_filename(
        __name__, "./data/csvs/vast_epoch09_info.csv"),
    "10x": pkg_resources.resource_filename(
        __name__, "./data/csvs/vast_epoch10x_info.csv"),
    "11x": pkg_resources.resource_filename(
        __name__, "./data/csvs/vast_epoch11x_info.csv")
}

FIELD_CENTRES = pd.read_csv(
    pkg_resources.resource_filename(
        __name__, "./data/csvs/vast_field_centres.csv"
    )
)

NIMBUS_BASE_DIR = "/Users/adam/testing/vast-tools-testing/PSR_J2129-04_data_2"
ADA_BASE_DIR = "/import/ada1/askap/PILOT/release/"

ALLOWED_PLANETS = [
    'mercury',
    'venus',
    'mars',
    'jupiter',
    'saturn',
    'uranus',
    'neptune',
    'sun',
    'moon'
]


def get_fields_per_epoch_info():
    """
    Function to create a dataframe suitable for fast
    field querying per epoch.

    :returns: Dataframe of epoch information
    :rtype: `pandas.core.frame.DataFrame`
    """

    for i, e in enumerate(FIELD_FILES):
        temp = pd.read_csv(FIELD_FILES[e], comment='#')
        temp['EPOCH'] = e
        if i == 0:
            epoch_fields = temp
        else:
            epoch_fields = epoch_fields.append(temp)

    epoch_fields = epoch_fields.drop_duplicates(
        ['FIELD_NAME', 'EPOCH']
    ).set_index(
        ['EPOCH', 'FIELD_NAME']
    ).drop(columns=[
        'BEAM', 'RA_HMS', 'DEC_DMS', 'DATEEND',
        'NINT', 'BMAJ', 'BMIN', 'BPA'
    ])

    return epoch_fields


def get_askap_observing_location():
    """
    Function to return ASKAP observing location.

    :returns: Location of ASKAP
    :rtype: `astropy.coordinates.earth.EarthLocation`
    """
    from astropy.coordinates import EarthLocation
    ASKAP_latitude = Angle("-26:41:46.0", unit=u.deg)
    ASKAP_longitude = Angle("116:38:13.0", unit=u.deg)

    observing_location = EarthLocation(
        lat=ASKAP_latitude, lon=ASKAP_longitude
    )

    return observing_location


class Fields:
    """
    Class to represent the VAST Pilot survey fields of a given
    epoch.

    Attributes
    ----------

    fields : pandas.core.frame.DataFrame
        DataFrame containing the fields information for the selected
        epoch.
    direction : astropy.coordinates.sky_coordinate.SkyCoord
        SkyCoord object representing the centres of each beam that
        make up each field in the epoch.
    field_cat : pandas.core.frame.DataFrame
        A dataframe containing the nearest beam for the sources
        queried (created through the 'find' method)

    Methods
    -------

    find(src_coord, max_sep, catalog)
        Finds the nearest beam centre to the queried coordinates.

    write_fields_cat(outfile)
        Write the results of find to a csv file.
    """

    def __init__(self, epoch):
        '''
        Constructor method.

        :param epoch: The epoch number of fields to collect
        :type epoch: str
        '''

        self.logger = logging.getLogger('vasttools.survey.Fields')
        self.logger.debug('Created Fields instance')
        self.logger.debug(FIELD_FILES[epoch])

        self.fields = pd.read_csv(FIELD_FILES[epoch], comment='#')
        # Epoch 99 has some empty beam directions (processing failures)
        # Drop them and any issue rows in the future.
        self.fields.dropna(inplace=True)
        self.fields.reset_index(drop=True, inplace=True)

        self.direction = SkyCoord(
            Angle(self.fields["RA_HMS"], unit=u.hourangle),
            Angle(self.fields["DEC_DMS"], unit=u.deg)
        )

    def find(self, src_coord, max_sep, catalog):
        '''
        Find which field each source in the catalogue is in.

        :param src_coord: Coordinates of sources to find fields for
        :type src_coord: `astropy.coordinates.sky_coordinate.SkyCoord`
        :param max_sep: Maximum allowable separation between source
            and beam centre in degrees
        :type max_sep: float
        :param catalog: Catalogue of sources to find fields for
        :type catalog: `pandas.core.frame.DataFrame`

        :returns: An updated catalogue with nearest field data for each
            source, and a boolean array corresponding to whether the source
            is within max_sep
        :rtype: `pandas.core.frame.DataFrame`, `numpy.ndarray`
        '''
        self.logger.debug(src_coord)
        self.logger.debug(catalog[np.isnan(src_coord.ra)])
        nearest_beams, seps, _d3d = src_coord.match_to_catalog_sky(
            self.direction)
        self.logger.debug(seps.deg)
        self.logger.debug(
            "Nearest beams: {}".format(self.fields["BEAM"][nearest_beams]))
        within_beam = seps.deg < max_sep
        catalog["sbid"] = self.fields["SBID"].iloc[nearest_beams].values
        nearest_fields = self.fields["FIELD_NAME"].iloc[nearest_beams]
        self.logger.debug(nearest_fields)
        catalog["field_name"] = nearest_fields.values
        catalog["original_index"] = catalog.index.values
        obs_dates = self.fields["DATEOBS"].iloc[nearest_beams]
        date_end = self.fields["DATEEND"].iloc[nearest_beams]
        catalog["obs_date"] = obs_dates.values
        catalog["date_end"] = date_end.values
        beams = self.fields["BEAM"][nearest_beams]
        catalog["beam"] = beams.values
        new_catalog = catalog[within_beam].reset_index(drop=True)
        self.logger.info(
            "Field match found for {}/{} sources.".format(
                len(new_catalog.index), len(nearest_beams)))

        if len(new_catalog.index) - len(nearest_beams) != 0:
            self.logger.warning(
                "No field matches found for sources with index (or name):")
            for i in range(0, len(catalog.index)):
                if i not in new_catalog["original_index"]:
                    if "name" in catalog.columns:
                        self.logger.warning(catalog["name"].iloc[i])
                    else:
                        self.logger.warning("{:03d}".format(i + 1))
        else:
            self.logger.info("All sources found!")

        self.field_cat = new_catalog

        return new_catalog, within_beam

    def write_fields_cat(self, outfile):
        '''
        Write the source-fields catalogue to file

        :param outfile: Name of the file to write to
        :type outfile: str
        '''

        self.field_cat.drop(
            ["original_index"],
            axis=1).to_csv(
            outfile,
            index=False)
        self.logger.info("Written field catalogue to {}.".format(outfile))


class Image:
    '''
    Represent and interact with an Image file from the VAST Pilot Survey.

    Attributes
    ----------
    sbid : int
        The SBID of the image.
    field : str
        The field name.
    epoch : str
        The epoch the image is part of.
    stokes : str
        The Stokes value of the image.
    path : str
        The path to the image file on the system.
    header : astropy.io.fits.Header
        The header of the image
    wcs : astropy.wcs.WCS
        The WCS object generated from the header.
    data : numpy.ndarry
        Array of the image data.
    beam : radio_beam.Beam
        Radio beam object representing the beam of
        the image.
    rmspath : str
        The path to the rms file on the system.
    rms_header : astropy.io.fits.Header
        The header of the RMS image
    rmsname : str
        The name of the RMS image.
    rms_fail : bool
        Becomes `True` if the RMS image is not found.

    Methods
    -------

    get_rms_img()
        Matches and loads the RMS image for the loaded image.

    measure_coord_pixel_values(coords, rms=False)
        Measure pixel values as the coords location. When rms
        is 'True' the values are read from the RMS image.
    '''

    def __init__(self, field, epoch, stokes, base_folder,
                 tiles=False, sbid=None, path=None, rmspath=None,
                 rms_header=None):
        '''
        Constructor method

        :param field: Name of the field
        :type field: str
        :param epoch:
        :type epoch:
        :param stokes: Stokes parameter of interest
        :type stokes: str
        :param base_folder: Path to base folder if using
            default directory structure
        :type base_folder: str
        :param tiles: Whether to use tiles or combined images,
            defaults to `False`
        :type tiles: bool, optional
        :param sbid: SBID of the field, defaults to None
        :type sbid: str, optional
        :param path: , defaults to None
        :type path: , optional
        :param rmspath: , defaults to None
        :type rmspath: , optional
        :param rms_header: Header of rms FITS image, defaults to None
        :type rms_header: astropy.io.fits.Header, optional
        '''

        self.logger = logging.getLogger('vasttools.survey.Image')
        self.logger.debug('Created Image instance')

        self.sbid = sbid
        self.field = field
        self.epoch = epoch
        self.stokes = stokes
        self.rms_header = rms_header
        self.path = path
        self.rmspath = rmspath

        if self.path is None:
            if tiles:
                img_folder = os.path.join(
                    base_folder,
                    "EPOCH{}".format(RELEASED_EPOCHS[epoch]),
                    "TILES",
                    "STOKES{}_IMAGES".format(stokes.upper())
                )
                img_template = (
                    'image.{}.SB{}.cont.{}.linmos.taylor.0.restored.fits'
                )
                self.imgname = img_template.format(stokes.lower(), sbid, field)
            else:
                img_folder = os.path.join(
                    base_folder,
                    "EPOCH{}".format(RELEASED_EPOCHS[epoch]),
                    "COMBINED",
                    "STOKES{}_IMAGES".format(stokes.upper())
                )
                self.imgname = '{}.EPOCH{}.{}.fits'.format(
                    field, RELEASED_EPOCHS[epoch], stokes.upper()
                )

            self.imgpath = os.path.join(img_folder, self.imgname)
        else:
            self.imgpath = path

        if os.path.isfile(self.imgpath):
            self.image_fail = False
        else:
            self.image_fail = True
            self.logger.error(
                "{} does not exist! Unable to create postagestamps".format(
                    self.imgpath
                )
            )
            return

        with fits.open(self.imgpath) as hdul:
            self.header = hdul[0].header
            self.wcs = WCS(self.header, naxis=2)

            try:
                self.data = hdul[0].data[0, 0, :, :]
            except Exception as e:
                self.data = hdul[0].data

            try:
                self.beam = Beam.from_fits_header(self.header)
            except Exception as e:
                self.logger.error("Beam information could not be read!")
                self.beam = None

    def get_rms_img(self):
        '''
        Load the noisemap corresponding to the image.
        '''
        if self.rmspath is None:
            self.rmsname = self.imgname.replace('.fits', '_rms.fits')
            self.rmspath = self.imgpath.replace(
                "_IMAGES", "_RMSMAPS"
            ).replace('.fits', '_rms.fits')

        if os.path.isfile(self.rmspath):
            self.rms_fail = False
        else:
            self.rms_fail = True
            self.logger.error(
                "{} does not exist! Unable to get noise maps.".format(
                    self.rmspath))
            return

        with fits.open(self.rmspath) as hdul:
            self.rms_header = hdul[0].header
            self.rms_wcs = WCS(self.rms_header, naxis=2)

            try:
                self.rms_data = hdul[0].data[0, 0, :, :]
            except Exception as e:
                self.rms_data = hdul[0].data

    def measure_coord_pixel_values(self, coords, rms=False):
        '''

        :param coords: Coordinate of interest
        :type coords:
        :param rms: Query the RMS image, defaults to `False`
        :type rms: bool, optional

        :returns: Pixel values stored in an array at the coords
            locations
        :rtype: numpy.ndarray.
        '''

        if rms is True:
            if self.rms_header is None:
                self.get_rms_img()

            thewcs = self.rms_wcs
            thedata = self.rms_data

        else:

            thewcs = self.wcs
            thedata = self.data

        array_coords = thewcs.world_to_array_index(coords)
        array_coords = np.array([
            np.array(array_coords[0]),
            np.array(array_coords[1]),
        ])

        # leaving this here just in case for now,
        # but sources should always be in image range

        # check for pixel wrapping
        # x_valid = np.logical_or(
        #     array_coords[0] > thedata.shape[0],
        #     array_coords[0] < 0
        # )
        #
        # y_valid = np.logical_or(
        #     array_coords[1] > thedata.shape[1],
        #     array_coords[1] < 0
        # )
        #
        # valid = ~np.logical_or(
        #     x_valid, y_valid
        # )
        #
        # valid_indexes = group[valid].index.values
        # not_valid_indexes = group[~valid].index.values

        values = thedata[
            array_coords[0],
            array_coords[1]
        ]

        return values
