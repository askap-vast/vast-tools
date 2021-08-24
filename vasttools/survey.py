"""Functions and classes related to loading and searching of the survey data.

Attributes:
    RELEASED_EPOCHS (Dict[str, str]): Dictionary containing the released
        epochs with the values being the 0 padded representation that the data
        file names use.
    FILED_FILES (Dict[str, str]): Package paths to the CSV files containing
        the observational information of each epoch.
    FIELD_CENTRES (pandas.core.frame.DataFrame): DataFrame loaded from a
        packaged CSV that contains the field centres for each individual
        pilot field.
    ALLOWED_PLANETS (List[str]): List of accepted planet and other object
        names.
"""
import sys
import os
import pandas as pd
import warnings
import importlib.resources
import itertools
import numpy as np
import re

import logging
import logging.handlers
import logging.config

from astropy.coordinates import Angle, EarthLocation
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from astropy.utils.exceptions import AstropyWarning, AstropyDeprecationWarning
from radio_beam import Beam
from typing import Tuple, Optional

warnings.filterwarnings('ignore', category=AstropyWarning, append=True)
warnings.filterwarnings(
    'ignore',
    category=AstropyDeprecationWarning,
    append=True
)


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
    "11x": "11x",
    "12": "12",
}


# TODO: Not sure this belongs in survey.
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


def load_field_centres() -> pd.DataFrame:
    """
    Loads the field centres csv file as a dataframe for use.

    Columns present are, 'field', 'centre-ra' and 'centre-dec'.
    The coordinates are in units of degrees.

    Returns:
        Dataframe containing the field centres.
    """
    with importlib.resources.path(
        "vasttools.data.csvs", "vast_field_centres.csv") as field_centres_csv:
        field_centres = pd.read_csv(field_centres_csv)

    return field_centres


def load_field_file(epoch: str) -> pd.DataFrame:
    """
    Load the csv field file of the requested epoch as a pandas dataframe.

    Columns present are 'SBID', 'FIELD_NAME', 'BEAM', 'RA_HMS', 'DEC_DMS',
    'DATEOBS', 'DATEEND', 'NINT', 'BMAJ', 'BMIN', 'BPA'

    Args:
        epoch: Epoch to load. Can be entered with or without zero padding.
            E.g. '3x', '9' or '03x' '09'.

    Returns:
        DataFrame containing the field information of the epoch.
    """
    if field not in RELEASED_EPOCHS:
        field = field[1:]

    paths = {
        "0": importlib.resources.path('vasttools.data.csvs', 'racs_info.csv'),
        "1": importlib.resources.path(
            'vasttools.data.csvs', 'vast_epoch01_info.csv'),
        "2": importlib.resources.path(
            'vasttools.data.csvs', 'vast_epoch02_info.csv'),
        "3x": importlib.resources.path(
            'vasttools.data.csvs', 'vast_epoch03x_info.csv'),
        "4x": importlib.resources.path(
            'vasttools.data.csvs', 'vast_epoch04x_info.csv'),
        "5x": importlib.resources.path(
            'vasttools.data.csvs', 'vast_epoch05x_info.csv'),
        "6x": importlib.resources.path(
            'vasttools.data.csvs', 'vast_epoch06x_info.csv'),
        "7x": importlib.resources.path(
            'vasttools.data.csvs', 'vast_epoch07x_info.csv'),
        "8": importlib.resources.path(
            'vasttools.data.csvs', 'vast_epoch08_info.csv'),
        "9": importlib.resources.path(
            'vasttools.data.csvs', 'vast_epoch09_info.csv'),
        "10x": importlib.resources.path(
            'vasttools.data.csvs', 'vast_epoch10x_info.csv'),
        "11x": importlib.resources.path(
            'vasttools.data.csvs', 'vast_epoch11x_info.csv'),
        "12": importlib.resources.path(
            'vasttools.data.csvs', 'vast_epoch12_info.csv'),
    }

    with paths[epoch] as fields_csv:
        fields_df = pd.read_csv(fields_csv, comment='#')

    return fields_df


def get_fields_per_epoch_info() -> pd.DataFrame:
    """
    Function to create a dataframe suitable for fast
    field querying per epoch.

    Returns:
        Dataframe of epoch information
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


def get_askap_observing_location() -> EarthLocation:
    """
    Function to return ASKAP observing location.

    Returns:
        Location of ASKAP.
    """
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

    Attributes:
        fields (pandas.core.frame.DataFrame):
            DataFrame containing the fields information for the selected
            epoch.
        direction (astropy.coordinates.sky_coordinate.SkyCoord):
            SkyCoord object representing the centres of each beam that
            make up each field in the epoch.
    """

    def __init__(self, epoch: str) -> None:
        """
        Constructor method.

        Args:
            epoch: The epoch number of fields to collect.

        Returns:
            None
        """
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

    # TODO: The below methods are no longer used in the code base.
    #       So these should probably be removed.

    # ATTRIBUTE DOCSTRING:
    # field_cat (pandas.core.frame.DataFrame):
    #     A dataframe containing the nearest beam for the sources
    #     queried (created through the 'find' method)

    # def find(
    #     self,
    #     src_coord: SkyCoord,
    #     max_sep: float,
    #     catalog: pd.DataFrame
    # ) -> Tuple[pd.DataFrame, np.ndarray]:
    #     """
    #     Find which field each source in the catalogue is in.
    #
    #     Args:
    #         src_coord: Coordinates of sources to find fields for.
    #         max_sep: Maximum allowable separation between source
    #             and beam centre in degrees.
    #         catalog: Catalogue of sources to find fields for.
    #
    #     Returns:
    #         An updated catalogue with nearest field data for each
    #         source, and a boolean array corresponding to whether the source
    #         is within max_sep.
    #     """
    #     self.logger.debug(src_coord)
    #     self.logger.debug(catalog[np.isnan(src_coord.ra)])
    #     nearest_beams, seps, _d3d = src_coord.match_to_catalog_sky(
    #         self.direction)
    #     self.logger.debug(seps.deg)
    #     self.logger.debug(
    #         "Nearest beams: {}".format(self.fields["BEAM"][nearest_beams]))
    #     within_beam = seps.deg < max_sep
    #     catalog["sbid"] = self.fields["SBID"].iloc[nearest_beams].values
    #     nearest_fields = self.fields["FIELD_NAME"].iloc[nearest_beams]
    #     self.logger.debug(nearest_fields)
    #     catalog["field_name"] = nearest_fields.values
    #     catalog["original_index"] = catalog.index.values
    #     obs_dates = self.fields["DATEOBS"].iloc[nearest_beams]
    #     date_end = self.fields["DATEEND"].iloc[nearest_beams]
    #     catalog["obs_date"] = obs_dates.values
    #     catalog["date_end"] = date_end.values
    #     beams = self.fields["BEAM"][nearest_beams]
    #     catalog["beam"] = beams.values
    #     new_catalog = catalog[within_beam].reset_index(drop=True)
    #     self.logger.info(
    #         "Field match found for {}/{} sources.".format(
    #             len(new_catalog.index), len(nearest_beams)))
    #
    #     if len(new_catalog.index) - len(nearest_beams) != 0:
    #         self.logger.warning(
    #             "No field matches found for sources with index (or name):")
    #         for i in range(0, len(catalog.index)):
    #             if i not in new_catalog["original_index"]:
    #                 if "name" in catalog.columns:
    #                     self.logger.warning(catalog["name"].iloc[i])
    #                 else:
    #                     self.logger.warning("{:03d}".format(i + 1))
    #     else:
    #         self.logger.info("All sources found!")
    #
    #     self.field_cat = new_catalog
    #
    #     return new_catalog, within_beam
    #
    # def write_fields_cat(self, outfile: str) -> None:
    #     """
    #     Write the source-fields catalogue to file.
    #
    #     Args:
    #         outfile: Name of the file to write to.
    #
    #     Returns:
    #         None
    #     """
    #     self.field_cat.drop(
    #         ["original_index"],
    #         axis=1).to_csv(
    #         outfile,
    #         index=False)
    #     self.logger.info("Written field catalogue to {}.".format(outfile))


class Image:
    """
    Represent and interact with an Image file from the VAST Pilot Survey.

    Attributes:
        sbid (int): The SBID of the image.
        field (str): The field name.
        epoch (str): The epoch the image is part of.
        stokes (str): The Stokes value of the image.
        path (str): The path to the image file on the system.
        header (astropy.io.fits.Header): The header of the image
        wcs (astropy.wcs.WCS): The WCS object generated from the header.
        data (numpy.ndarry): Array of the image data.
        beam (radio_beam.Beam): radio_beam.Beam object representing the beam
            of the image. Refer to the
            [radio_beam](https://radio-beam.readthedocs.io/en/latest/)
            documentation for more information.
        rmspath (str): The path to the rms file on the system.
        rms_header (astropy.io.fits.Header): The header of the RMS image
        rmsname (str): The name of the RMS image.
        rms_fail (bool): Becomes `True` if the RMS image is not found.
    """

    def __init__(
        self,
        field: str,
        epoch: str,
        stokes: str,
        base_folder: str,
        tiles: bool = False,
        sbid: Optional[str] = None,
        path: Optional[str] = None,
        rmspath: Optional[str] = None,
        rms_header: Optional[fits.Header] = None
    ) -> None:
        """
        Constructor method.

        Args:
            field: Name of the field.
            epoch: The epoch of the field requested.
            stokes: Stokes parameter of interest.
            base_folder: Path to base folder if using
                default directory structure.
            tiles: Will use 'COMBINED' images when 'True' and 'TILES' when
                'False', defaults to `False`.
            sbid: SBID of the field, defaults to None.
            path: Path to the image file if already known, defaults to None.
            rmspath: The path to the corresponding rms image file if known,
                defaults to None.
            rms_header: Header of rms FITS image if already obtained,
                defaults to None.

        Returns:
            None
        """
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

    def get_rms_img(self) -> None:
        """
        Load the noisemap corresponding to the image.

        Returns:
            None
        """
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

    def measure_coord_pixel_values(
        self,
        coords: SkyCoord,
        rms: bool = False
    ) -> np.ndarray:
        """Measures the pixel values at the provided coordinate values.

        Args:
            coords: Coordinate of interest.
            rms: Query the RMS image, defaults to `False`.

        Returns:
            Pixel values stored in an array at the coords locations.
        """

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
