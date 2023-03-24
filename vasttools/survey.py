"""Functions and classes related to loading and searching of the survey data.
"""
import os
import pandas as pd
import warnings
import importlib.resources
import numpy as np

import logging
import logging.handlers
import logging.config

from astropy.coordinates import Angle, EarthLocation
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.exceptions import AstropyWarning, AstropyDeprecationWarning
from radio_beam import Beam
from typing import Optional, List, Union

from vasttools import RELEASED_EPOCHS, OBSERVED_EPOCHS

warnings.filterwarnings('ignore', category=AstropyWarning, append=True)
warnings.filterwarnings(
    'ignore',
    category=AstropyDeprecationWarning,
    append=True
)


def load_field_centres() -> pd.DataFrame:
    """
    Loads the field centres csv files as a dataframe for use.

    Columns present are, 'field', 'centre-ra' and 'centre-dec'.
    The coordinates are in units of degrees.

    Returns:
        Dataframe containing the field centres.
    """
    with importlib.resources.path(
        "vasttools.data.csvs", "low_field_centres.csv"
    ) as field_centres_csv:
        low_centres = pd.read_csv(field_centres_csv)

    with importlib.resources.path(
        "vasttools.data.csvs", "mid_field_centres.csv"
    ) as field_centres_csv:
        mid_centres = pd.read_csv(field_centres_csv)

    field_centres = pd.concat([low_centres, mid_centres])
    
    _strip_fieldnames(field_centres, 'field')

    return field_centres


def _strip_fieldnames(df: pd.DataFrame, colname) -> None:
    """
    Some field names have historically used the interleaving naming scheme,
    but that has changed as of January 2023. This function removes the "A"
    that is on the end of the field names
    
    Args:
        df: DataFrame to strip field names from
        colname: Name of the column containing the field names
    """
    
    df[colname] = df[colname].str.rstrip('A')
    
def load_fields_file(epoch: str) -> pd.DataFrame:
    """
    Load the csv field file of the requested epoch as a pandas dataframe.

    Columns present are 'SBID', 'FIELD_NAME', 'BEAM', 'RA_HMS', 'DEC_DMS',
    'DATEOBS', 'DATEEND', 'NINT', 'BMAJ', 'BMIN', 'BPA'

    Args:
        epoch: Epoch to load. Can be entered with or without zero padding.
            E.g. '3x', '9' or '03x' '09'.

    Returns:
        DataFrame containing the field information of the epoch.

    Raises:
        ValueError: Raised when epoch requested is not released.
    """
    if epoch not in RELEASED_EPOCHS:
        if len(str(epoch)) > 2 and epoch.startswith('0'):
            epoch = epoch[1:]
        if epoch not in RELEASED_EPOCHS:
            if epoch not in OBSERVED_EPOCHS:
                raise ValueError(
                    f'Epoch {epoch} is not available or is not a valid epoch.'
                )

    paths = {
        "0": importlib.resources.path(
            'vasttools.data.csvs', 'racs_low_info.csv'),
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
        "13": importlib.resources.path(
            'vasttools.data.csvs', 'vast_epoch13_info.csv'),
        "14": importlib.resources.path(
            'vasttools.data.csvs', 'racs_mid_info.csv'),
        "17": importlib.resources.path(
            'vasttools.data.csvs', 'vast_epoch17_info.csv'),
        "18": importlib.resources.path(
            'vasttools.data.csvs', 'vast_epoch18_info.csv'),
        "19": importlib.resources.path(
            'vasttools.data.csvs', 'vast_epoch19_info.csv'),
        "20": importlib.resources.path(
            'vasttools.data.csvs', 'vast_epoch20_info.csv'),
        "21": importlib.resources.path(
            'vasttools.data.csvs', 'vast_epoch21_info.csv'),
        "22": importlib.resources.path(
            'vasttools.data.csvs', 'vast_epoch22_info.csv'),
        "23": importlib.resources.path(
            'vasttools.data.csvs', 'vast_epoch23_info.csv'),
        "24": importlib.resources.path(
            'vasttools.data.csvs', 'vast_epoch24_info.csv'),
        "25": importlib.resources.path(
            'vasttools.data.csvs', 'vast_epoch25_info.csv'),
        "26": importlib.resources.path(
            'vasttools.data.csvs', 'vast_epoch26_info.csv'),
        "28": importlib.resources.path(
            'vasttools.data.csvs', 'racs_high_info.csv'),
        "29": importlib.resources.path(
            'vasttools.data.csvs', 'racs_low2_info.csv'),
    }

    with paths[epoch] as fields_csv:
        fields_df = pd.read_csv(fields_csv, comment='#')
        _strip_fieldnames(fields_df, 'FIELD_NAME')

    return fields_df


def get_fields_per_epoch_info() -> pd.DataFrame:
    """
    Function to create a dataframe suitable for fast
    field querying per epoch.

    Returns:
        Dataframe of epoch information
    """
    epoch_fields = pd.DataFrame()
    for i, e in enumerate(RELEASED_EPOCHS):
        temp = load_fields_file(e)
        temp['EPOCH'] = e
        epoch_fields = pd.concat([epoch_fields, temp])

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


def get_supported_epochs() -> List[str]:
    """
    Returns the user a list of supported VAST Pilot epochs.

    Returns:
        List of supported epochs.
    """
    return list(sorted(RELEASED_EPOCHS.values()))


class Fields:
    """
    Class to represent the VAST Pilot survey fields of a given epoch.

    Attributes:
        fields (pandas.core.frame.DataFrame):
            DataFrame containing the fields information for the selected
            epoch.
        direction (astropy.coordinates.sky_coordinate.SkyCoord):
            SkyCoord object representing the centres of each beam that
            make up each field in the epoch.
    """

    def __init__(self, epochs: Union[str, List[str]]) -> None:
        """
        Constructor method.

        Args:
            epochs: The epoch number(s) of fields to collect.

        Returns:
            None
        """
        self.logger = logging.getLogger('vasttools.survey.Fields')
        self.logger.debug('Created Fields instance')

        if type(epochs) == str:
            epochs = list(epochs)

        field_dfs = []
        for epoch in epochs:
            self.logger.debug(f"Loading epoch {epoch}")
            field_dfs.append(load_fields_file(epoch))

        self.fields = pd.concat(field_dfs)

        self.logger.debug(f"Frequencies: {self.fields.OBS_FREQ.unique()}")

        self.fields.dropna(inplace=True)
        self.fields.reset_index(drop=True, inplace=True)

        self.direction = SkyCoord(
            Angle(self.fields["RA_HMS"], unit=u.hourangle),
            Angle(self.fields["DEC_DMS"], unit=u.deg)
        )


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
        rms_header: Optional[fits.Header] = None,
        corrected_data: bool = True
    ) -> None:
        """
        Constructor method.

        Args:
            field: Name of the field.
            epoch: The epoch of the field requested. Do not zero pad the epoch
                number.
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
            corrected_data: Access the corrected data. Only relevant if
                `tiles` is `True`. Defaults to `True`.

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
        self.tiles = tiles
        self.base_folder = base_folder
        self.corrected_data = corrected_data

        if self.path is None:
            self.get_paths_and_names()
        else:
            self.imgpath = self.path
            self.imgname = os.path.basename(self.path)

        self._check_exists()
        self._loaded_data = False

    def get_paths_and_names(self) -> None:
        """
        Configure the file names if they have no been provided.

        Args:
            None

        Returns:
            None
        """
        if self.tiles:
            img_folder = os.path.join(
                self.base_folder,
                "EPOCH{}".format(RELEASED_EPOCHS[self.epoch]),
                "TILES",
                "STOKES{}_IMAGES_CORRECTED".format(self.stokes.upper())
            )
            img_template = (
                'image.{}.{}.SB{}.cont.taylor.0.restored.corrected.fits'
            )

            if not self.corrected_data:
                img_folder = img_folder.replace("_CORRECTED", "")
                img_template = img_template.replace(".corrected", "")

            self.imgname = img_template.format(
                self.stokes.lower(), self.field, self.sbid
            )
            img_path = os.path.join(img_folder, self.imgname)

            if not os.path.exists(img_path):
                if self.corrected_data:
                    self.imgname = self.imgname.replace(".corrected.",
                                                        ".conv.corrected."
                                                        )
                else:
                    self.imgname = self.imgname.replace(".fits",
                                                        ".conv.fits"
                                                        )
        else:
            img_folder = os.path.join(
                self.base_folder,
                "EPOCH{}".format(RELEASED_EPOCHS[self.epoch]),
                "COMBINED",
                "STOKES{}_IMAGES".format(self.stokes.upper())
            )
            self.imgname = '{}.EPOCH{}.{}.conv.fits'.format(
                self.field,
                RELEASED_EPOCHS[self.epoch],
                self.stokes.upper()
            )

        self.imgpath = os.path.join(img_folder, self.imgname)

    def _check_exists(self) -> bool:
        if os.path.isfile(self.imgpath):
            self.image_fail = False
        else:
            self.image_fail = True
            self.logger.error(
                "{} does not exist! Unable to create postagestamps".format(
                    self.imgpath
                )
            )

    def get_img_data(self) -> None:
        """
        Load the data from the image, including the beam.

        Returns:
            None
        """
        if self.image_fail:
            return

        with fits.open(self.imgpath) as hdul:
            self.header = hdul[0].header
            self.wcs = WCS(self.header, naxis=2)
            self.data = hdul[0].data.squeeze()

        try:
            self.beam = Beam.from_fits_header(self.header)
        except Exception as e:
            self.logger.error("Beam information could not be read!")
            self.logger.error(f"Error: {e}")
            self.beam = None

        self._loaded_data = True

    def get_rms_img(self) -> None:
        """
        Load the noisemap corresponding to the image.

        Returns:
            None
        """
        if self.rmspath is None:
            self.rmsname = "noiseMap.{}".format(self.imgname)
            self.rmspath = self.imgpath.replace(
                "_IMAGES", "_RMSMAPS"
            ).replace(self.imgname, self.rmsname)

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
            self.rms_data = hdul[0].data.squeeze()

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
            if not self._loaded_data:
                self.get_img_data()
            thewcs = self.wcs
            thedata = self.data

        array_coords = thewcs.world_to_array_index(coords)
        array_coords = np.array([
            np.array(array_coords[0]),
            np.array(array_coords[1]),
        ])

        # leaving this here just in case for now,
        # but sources should always be in image range
        # if enabled it should be tested

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
