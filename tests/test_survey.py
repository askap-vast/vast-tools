import astropy.units as u
import numpy as np
import pandas as pd
import pathlib
import pytest

from astropy.coordinates import Angle, EarthLocation, SkyCoord
from astropy.io import fits
from pytest_mock import mocker
from typing import Optional

from vasttools import RELEASED_EPOCHS
import vasttools.survey as vts


@pytest.fixture
def mocked_load_fields_file() -> pd.DataFrame:
    df = pd.DataFrame(
        data = {
            'SBID': ['9876', '9876'],
            'FIELD_NAME': ['VAST_1234+00A', 'VAST_1234+00A'],
            'BEAM': [0, 1],
            'RA_HMS': ["12:34:56", "12:34:12"],
            'DEC_DMS': ["00:12:34", "-01:12:34"],
            'DATEOBS': ['2019-10-29 12:34:02.450', '2019-10-29 12:34:02.450'],
            'DATEEND': ['2019-10-29 12:45:02.450', '2019-10-29 12:45:02.450'],
            'NINT': [100, 100],
            'BMAJ': [15.0, 15.1],
            'BMIN': [12.1, 12.2],
            'BPA': [90., 90.]
        }
    )

    return df


@pytest.fixture
def mocked_fits_open() -> fits.HDUList:
    """
    Docstring
    """
    data = np.zeros((100,100), dtype=np.float32)
    for i in range(100):
        data[i] = np.arange(100*i, 100*(i+1))

    hdu = fits.PrimaryHDU(data=data)

    header = hdu.header

    header['BMAJ'] = 0.00493462835125746
    header['BMIN'] = 0.00300487516378073
    header['BPA'] = -71.0711523845679
    header['CDELT1'] = 1.0
    header['CDELT2'] = 1.0
    header['WCSAXES'] = 2
    header['TELESCOP'] = "ASKAP"
    header['RESTFREQ'] = 887491000.0
    header['DATE-OBS'] = "2020-01-12T05:36:03.834"
    header['DATE-BEG'] = "2020-01-12T05:36:03.834"
    header['DATE-END'] = "2020-01-12T05:47:50.517"
    header['TIMESYS'] = "UTC"
    header['RADESYS'] = "ICRS"
    header['CTYPE1'] = "RA---SIN"
    header['CUNIT1'] = "deg"
    header['CRVAL1'] = 319.6519091667
    header['CRPIX1'] = 4059.5
    header['CD1_1'] = -0.0006944444444444
    header['CD1_2'] = 0.0
    header['CTYPE2'] = "DEC--SIN"
    header['CUNIT2'] = "deg"
    header['CRVAL2'] = -6.2985525
    header['CRPIX2'] = -2537.5
    header['CD2_1'] = 0.0
    header['CD2_2'] = 0.0006944444444444

    hdul = fits.HDUList([hdu])

    return hdul


@pytest.fixture
def init_Image() -> vts.Image:
    def _init_Image(
        field: str = "VAST_0012+00",
        epoch: str = '1',
        stokes: str = 'I',
        path: Optional[str] = None,
        tiles: bool = False,
        base_folder: str = '/mocked/basefolder/',
        rmspath: Optional[str] = None
    ):
        img = vts.Image(
            field=field,
            epoch=epoch,
            stokes=stokes,
            base_folder=base_folder,
            tiles=tiles,
            sbid=9667,
            path=path,
            rmspath=rmspath
        )

        return img

    return _init_Image


def test_load_field_centres(mocker):
    assumed_path = "vasttools.data.csvs"
    assumed_filename = "vast_field_centres.csv"

    importlib_mocker = mocker.patch(
        'importlib.resources.path',
        return_value=pathlib.Path(assumed_filename)
    )
    pandas_mocker = mocker.patch(
        'vasttools.survey.pd.read_csv', return_value=-99
    )

    result = vts.load_field_centres()

    importlib_mocker.assert_called_once_with(assumed_path, assumed_filename)
    pandas_mocker.assert_called_once_with(importlib_mocker.return_value)
    assert result == -99


def test_load_fields_file(mocker):
    assumed_path = "vasttools.data.csvs"
    assumed_filename = 'vast_epoch01_info.csv'
    epoch = '1'

    importlib_mocker = mocker.patch(
        'importlib.resources.path',
        return_value=pathlib.Path(assumed_filename)
    )
    pandas_mocker = mocker.patch(
        'vasttools.survey.pd.read_csv', return_value=-99
    )

    result = vts.load_fields_file(epoch)

    expected_calls = [mocker.call(assumed_path, assumed_filename)]
    importlib_mocker.assert_has_calls(expected_calls)
    pandas_mocker.assert_called_once_with(
        importlib_mocker.return_value, comment='#'
    )
    assert result == -99


def test_load_fields_file_epoch_fail(mocker):
    assumed_path = "vasttools.data.csvs"
    assumed_filename = 'vast_epoch01_info.csv'
    epoch = '96x'

    importlib_mocker = mocker.patch(
        'importlib.resources.path',
        return_value=pathlib.Path(assumed_filename)
    )

    with pytest.raises(ValueError) as excinfo:
        vts.load_fields_file(epoch)

    assert str(excinfo.value) == (
        f'Epoch {epoch} is not available or is not a valid epoch.'
    )


def test_get_fields_per_epoch_info(mocked_load_fields_file, mocker):
    load_fields_file_mocker = mocker.patch(
        'vasttools.survey.load_fields_file',
        return_value=mocked_load_fields_file
    )

    expected_shape = len(RELEASED_EPOCHS)

    result = vts.get_fields_per_epoch_info()

    assert result.shape[0] == expected_shape


def test_get_askap_observing_location():
    ASKAP_latitude = Angle("-26:41:46.0", unit=u.deg)
    ASKAP_longitude = Angle("116:38:13.0", unit=u.deg)

    expected = EarthLocation(
        lat=ASKAP_latitude, lon=ASKAP_longitude
    )

    result = vts.get_askap_observing_location()

    assert result == expected


class TestFields:
    def test_fields_init(self, mocked_load_fields_file, mocker):
        load_fields_file_mocker = mocker.patch(
            'vasttools.survey.load_fields_file',
            return_value=mocked_load_fields_file
        )

        expected_skycoord = SkyCoord(
            Angle(mocked_load_fields_file["RA_HMS"], unit=u.hourangle),
            Angle(mocked_load_fields_file["DEC_DMS"], unit=u.deg)
        )

        fields_result = vts.Fields('1')

        assert fields_result.fields.equals(mocked_load_fields_file)
        assert np.all(fields_result.direction == expected_skycoord)

    def test_fields_init_nan(self, mocked_load_fields_file, mocker):
        mock_result = mocked_load_fields_file.copy()
        mock_result.at[1, 'BMAJ'] = np.nan

        load_fields_file_mocker = mocker.patch(
            'vasttools.survey.load_fields_file',
            return_value=mock_result
        )

        fields_result = vts.Fields('1')

        assert fields_result.fields.shape[0] == 1


class TestImage:
    def test_image_init_combined_nopath(self, init_Image, mocker):
        mock_isfile = mocker.patch('os.path.isfile', return_value=True)

        image = init_Image()

        # Use the defaults on the init_Image fixture.
        expected_filename = "VAST_0012+00.EPOCH01.I.fits"
        expected_path = (
            "/mocked/basefolder/EPOCH01/COMBINED/"
            f"STOKESI_IMAGES/{expected_filename}"
        )

        assert image.imgpath == expected_path
        assert image.imgname == expected_filename
        assert image.image_fail == False

    def test_image_init_tiles_nopath(self, init_Image, mocker):
        mock_isfile = mocker.patch('os.path.isfile', return_value=True)

        image = init_Image(tiles=True)

        # Use the defaults on the init_Image fixture.
        expected_filename = (
            "image.i.SB9667.cont.VAST_0012+00.linmos.taylor.0.restored.fits"
        )
        expected_path = (
            "/mocked/basefolder/EPOCH01/TILES/"
            f"STOKESI_IMAGES/{expected_filename}"
        )

        assert image.imgpath == expected_path
        assert image.imgname == expected_filename
        assert image.image_fail == False

    def test_image_init_combined_nopath_stokesv(self, init_Image, mocker):
        mock_isfile = mocker.patch('os.path.isfile', return_value=True)

        image = init_Image(stokes='V')

        # Use the defaults on the init_Image fixture.
        expected_filename = "VAST_0012+00.EPOCH01.V.fits"
        expected_path = (
            "/mocked/basefolder/EPOCH01/COMBINED/"
            f"STOKESV_IMAGES/{expected_filename}"
        )

        assert image.imgpath == expected_path
        assert image.imgname == expected_filename

    def test_image_init_tiles_nopath_stokesv(self, init_Image, mocker):
        mock_isfile = mocker.patch('os.path.isfile', return_value=True)

        image = init_Image(tiles=True, stokes='V')

        # Use the defaults on the init_Image fixture.
        expected_filename = (
            "image.v.SB9667.cont.VAST_0012+00.linmos.taylor.0.restored.fits"
        )
        expected_path = (
            "/mocked/basefolder/EPOCH01/TILES/"
            f"STOKESV_IMAGES/{expected_filename}"
        )

        assert image.imgpath == expected_path
        assert image.imgname == expected_filename

    def test_image_init_path(self, init_Image, mocker):
        mock_isfile = mocker.patch('os.path.isfile', return_value=True)

        expected_filename = "image1.fits"
        expected_path = (
            "/mocked/basefolder/my/data/"
            f"images/{expected_filename}"
        )

        image = init_Image(path=expected_path)

        assert image.imgpath == expected_path
        assert image.imgname == expected_filename

    def test_image_init_image_fail(self, init_Image, mocker):
        mock_isfile = mocker.patch('os.path.isfile', return_value=False)

        expected_filename = "image1.fits"
        expected_path = (
            "/mocked/basefolder/my/data/"
            f"images/{expected_filename}"
        )

        image = init_Image(path=expected_path)

        assert image.image_fail == True

    def test_image_get_data(self, init_Image, mocked_fits_open, mocker):
        mock_isfile = mocker.patch('os.path.isfile', return_value=True)
        mock_fits_open = mocker.patch(
            'vasttools.survey.fits.open',
            return_value=mocked_fits_open
        )

        image = init_Image()

        expected_filename = "VAST_0012+00.EPOCH01.I.fits"
        expected_path = (
            "/mocked/basefolder/EPOCH01/COMBINED/"
            f"STOKESI_IMAGES/{expected_filename}"
        )

        image.get_img_data()

        mock_fits_open.assert_called_once_with(expected_path)

    def test_image_get_rms_img_combined_nopath(
        self, init_Image, mocked_fits_open, mocker
    ):
        """
        No need to test tiles as it does the same replacements.
        """
        mock_isfile = mocker.patch('os.path.isfile', return_value=True)
        mock_fits_open = mocker.patch(
            'vasttools.survey.fits.open',
            return_value=mocked_fits_open
        )

        # Use the defaults on the init_Image fixture.
        expected_filename = "VAST_0012+00.EPOCH01.I_rms.fits"
        expected_path = (
            "/mocked/basefolder/EPOCH01/COMBINED/"
            f"STOKESI_RMSMAPS/{expected_filename}"
        )

        image = init_Image()
        image.get_rms_img()

        assert image.rmspath == expected_path
        assert image.rmsname == expected_filename
        assert image.rms_fail == False
        mock_fits_open.assert_called_once_with(expected_path)

    def test_image_get_rms_img_path(
        self, init_Image, mocked_fits_open, mocker
    ):
        """
        Docstring
        """
        mock_isfile = mocker.patch('os.path.isfile', return_value=True)
        mock_fits_open = mocker.patch(
            'vasttools.survey.fits.open',
            return_value=mocked_fits_open
        )

        expected_filename = "image1_rms.fits"
        expected_path = (
            "/mocked/basefolder/my/data/"
            f"rmsmaps/{expected_filename}"
        )

        image = init_Image(rmspath=expected_path)
        image.get_rms_img()

        assert image.rmspath == expected_path
        assert image.rms_fail == False
        mock_fits_open.assert_called_once_with(expected_path)

    def test_image_measure_coord_pixel_values(
        self, init_Image, mocked_fits_open, mocker
    ):
        mock_isfile = mocker.patch('os.path.isfile', return_value=True)
        mock_fits_open = mocker.patch(
            'vasttools.survey.fits.open',
            return_value=mocked_fits_open
        )

        image = init_Image()

        coords_to_measure = SkyCoord(
            ra=[
                '21:29:45.338',
                '21:29:51.246',
                '21:29:54.231'
            ],
            dec = [
                '-04:28:08.24',
                '-04:31:20.36',
                '-04:29:35.07'
            ],
            unit=(u.hourangle, u.deg)
        )

        expected_values = np.array([8559, 824, 5006], dtype=np.float32)

        values = image.measure_coord_pixel_values(coords_to_measure)

        assert np.all(values == expected_values)

    def test_image_measure_coord_pixel_values_rms(
        self, init_Image, mocked_fits_open, mocker
    ):
        mock_isfile = mocker.patch('os.path.isfile', return_value=True)
        mock_fits_open = mocker.patch(
            'vasttools.survey.fits.open',
            return_value=mocked_fits_open
        )

        image = init_Image()
        image.get_rms_img()

        coords_to_measure = SkyCoord(
            ra=[
                '21:29:52.705',
                '21:29:47.035',
                '21:29:40.032'
            ],
            dec = [
                '-04:28:17.65',
                '-04:29:40.65',
                '-04:31:26.26'
            ],
            unit=(u.hourangle, u.deg)
        )

        expected_values = np.array([8115, 4849, 691], dtype=np.float32)

        values = image.measure_coord_pixel_values(coords_to_measure, rms=True)

        assert np.all(values == expected_values)
