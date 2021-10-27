import astropy.units as u
import numpy as np
import pandas as pd
import pytest

from astropy.coordinates import SkyCoord, Angle
from astropy.table import Table
from astropy.time import Time
from astropy.io import fits
from pytest_mock import mocker
from pathlib import Path
from mocpy import MOC, STMOC
from typing import Union

import vasttools.tools as vtt

TEST_DATA_DIR = Path(__file__).resolve().parent / 'data'


@pytest.fixture
def source_df() -> pd.DataFrame:
    """
    Produces a dataframe containing source coordinates for testing. Contains
    ra and dec.

    Returns:
        Dataframe with coordinates.
    """
    source_df = pd.DataFrame(
        data={
            'ra': [
                12.59305887,
                12.68310734,
                0.0,
                345.0,
                30.0,
                9.538331008
            ],
            'dec': [
                -23.06359886,
                -24.26722689,
                0.0,
                -46.12,
                -2.3,
                -23.29754621
            ]
        }
    )

    return source_df


@pytest.fixture
def dummy_load_fields_file() -> pd.DataFrame:
    """
    Produces a dummy fields file.

    Returns:
        Dummy fields dataframe.
    """
    df = pd.DataFrame(
        data={
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
def dummy_fits_open() -> fits.HDUList:
    """
    Produces a dummy fits file (hdulist).

    Returns:
        The fits file as an hdulist instance.
    """
    data = np.zeros((100, 100), dtype=np.float32)

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
def dummy_moc() -> MOC:
    """
    Produces a MOC.

    Returns:
        The MOC.
    """
    json = {'9': [1215087, 1215098]}
    moc = MOC.from_json(json)

    return moc


@pytest.fixture
def dummy_stmoc() -> Union[MOC, STMOC]:
    """
    Produces a MOC and corresponding STMOC.

    Returns:
        The STMOC.
    """
    json = {'9': [1215087, 1215098]}
    moc = MOC.from_json(json)

    start = Time(['2019-10-29 12:34:02.450'])
    end = Time(['2019-10-29 12:45:02.450'])

    stmoc = STMOC.from_spatial_coverages(
        start, end, [moc]
    )

    return stmoc


def test_find_in_moc(source_df: pd.DataFrame) -> None:
    """
    Tests finding which sources are contained within a MOC.

    Args:
        source_df: The dataframe containing the source data.

    Returns:
        None
    """
    filename = TEST_DATA_DIR / 'test_skymap_gw190814.fits.gz'
    results = np.array([0, 1])

    moc_skymap = vtt.skymap2moc(filename, 0.9)

    func_output = vtt.find_in_moc(moc_skymap, source_df, pipe=False)

    assert (func_output == results).all()


def test_add_credible_levels(source_df: pd.DataFrame) -> None:
    """
    Tests adding the credible levels from a skymap.

    Args:
        source_df: The dataframe containing the source data.

    Returns:
        None
    """
    credible_levels = np.array([0.80250182,
                                0.28886045,
                                1.0,
                                1.0,
                                1.0,
                                0.97830106,
                                ])

    filename = TEST_DATA_DIR / 'test_skymap_gw190814.fits.gz'
    vtt.add_credible_levels(filename, source_df, pipe=False)

    assert source_df['credible_level'].values == pytest.approx(
        credible_levels, rel=1e-1)


def test_create_fields_csv(mocker) -> None:
    """
    Tests creating the fields csv for a single epoch.

    Args:
        mocker: The pytest mock mocker object.

    Returns:
        None
    """

    mocker_file_exists = mocker.patch(
        'vasttools.tools.Path.exists',
        return_value=True
    )
    mocker_fields_df = mocker.patch(
        'vasttools.tools._create_fields_df',
        return_value=pd.DataFrame()
    )
    mocker_to_csv = mocker.patch(
        'pandas.DataFrame.to_csv'
    )

    epoch_num = '2'
    outfile = 'vast_epoch{}_info.csv'.format(epoch_num)

    vtt.create_fields_csv(epoch_num, TEST_DATA_DIR / 'surveys_db')

    mocker_to_csv.assert_called_once_with(Path(outfile), index=False)


def test__create_fields_df() -> None:
    """
    Tests creating the fields csv for a single epoch.

    Args:
        None

    Returns:
        None
    """

    out_df = vtt._create_fields_df('2', TEST_DATA_DIR / 'surveys_db')

    expected_df = pd.read_csv(TEST_DATA_DIR / 'vast_epoch2_info.csv')

    pd.testing.assert_frame_equal(out_df, expected_df)


# TODO: Update name of images if standard changes.
@pytest.mark.parametrize(
    "image_type,image_name",
    [
        ('COMBINED', 'VAST_1234+00A.EPOCH01.I.TEST.fits'),
        (
            'TILES',
            "image.i.SB9876.cont.VAST_1234+00A.linmos.taylor.0.restored.fits"
        )
    ]
)
def test_add_obs_date(
    image_type: str,
    image_name: str,
    dummy_fits_open: fits.HDUList,
    dummy_load_fields_file: pd.DataFrame,
    mocker
) -> None:
    """
    Tests adding observation dates to fits images.

    Args:
        image_type: The image_type argument passed to 'add_obs_date' from the
            parametrize.
        image_name: The dummy image name used for 'add_obs_date' from the
            parametrize.
        dummy_fits_open: The dummy HDUList object that represents an open
            FITS file.
        dummy_load_fields_file: The dummy fields file.
        mocker: The pytest mock mocker object.

    Returns:
        None
    """
    mocker_load_fields_file = mocker.patch(
        'vasttools.tools.load_fields_file',
        return_value=dummy_load_fields_file
    )

    mocker_get_epoch_images = mocker.patch(
        'vasttools.tools._get_epoch_images',
        return_value=[image_name]
    )

    mocker_fits_open = mocker.patch(
        'vasttools.tools.fits.open',
        return_value=dummy_fits_open
    )

    start = Time(dummy_load_fields_file['DATEOBS'].iloc[0])
    end = Time(dummy_load_fields_file['DATEEND'].iloc[0])
    duration = end - start

    expected_new_headers = {
        "DATE-OBS": start.fits,
        "MJD-OBS": start.mjd,
        "DATE-BEG": start.fits,
        "DATE-END": end.fits,
        "MJD-BEG": start.mjd,
        "MJD-END": end.mjd,
        "TELAPSE": duration.sec,
        "TIMEUNIT": "s"
    }

    vtt.add_obs_date('1', image_type, '', '.')

    new_header = mocker_fits_open.return_value[0].header

    mocker_fits_open.assert_called_once_with(image_name, mode='update')
    assert np.all([
        expected_new_headers[i] == new_header[i] for i in expected_new_headers
    ])


def test_gen_mocs_image(
        dummy_fits_open: fits.HDUList,
        dummy_load_fields_file: pd.DataFrame,
        dummy_moc: MOC,
        tmp_path: Path,
        mocker) -> None:
    """
    Tests the generation of a MOC and STMOC for a single fits file

    Args:
        dummy_fits_open: The dummy HDUList object that represents an open
            FITS file.
        dummy_load_fields_file: The dummy fields file.
        dummy_moc: The dummy MOC object representing an open MOC.
        tmp_path: The default pytest temporary path.
        mocker: The pytest mock mocker object.

    Returns:
        None
    """

    mocker_load_fields_file = mocker.patch(
        'vasttools.tools.load_fields_file',
        return_value=dummy_load_fields_file
    )

    mocker_create_moc_from_fits = mocker.patch(
        'vasttools.tools.create_moc_from_fits',
        return_value=dummy_moc
    )

    mocker_file_exists = mocker.patch(
        'vasttools.tools.Path.exists',
        return_value=True
    )

    start = Time(dummy_load_fields_file['DATEOBS'].iloc[0])
    end = Time(dummy_load_fields_file['DATEEND'].iloc[0])
    duration = end - start

    dummy_header = {
        "DATE-OBS": start.fits,
        "MJD-OBS": start.mjd,
        "DATE-BEG": start.fits,
        "DATE-END": end.fits,
        "MJD-BEG": start.mjd,
        "MJD-END": end.mjd,
        "TELAPSE": duration.sec,
        "TIMEUNIT": "s"
    }

    mocker_fits_open = mocker.patch(
        'vasttools.tools.fits.getheader',
        return_value=dummy_header
    )
    mocker_isfile = mocker.patch(
        'os.path.isfile',
        return_value=True
    )
    mocker_moc_write = mocker.patch(
        'vasttools.tools.MOC.write'
    )
    mocker_stmoc_write = mocker.patch(
        'vasttools.tools.STMOC.write'
    )

    fits_file = 'test.fits'
    moc_file = fits_file.replace('.fits', '.moc.fits')
    stmoc_file = fits_file.replace('.fits', '.stmoc.fits')

    moc, stmoc = vtt.gen_mocs_image(fits_file, outdir=tmp_path)

    mocker_moc_write.assert_called_once_with(tmp_path / moc_file,
                                             overwrite=True)

    mocker_stmoc_write.assert_called_once_with(tmp_path / stmoc_file,
                                               overwrite=True)

    assert stmoc.max_time.jd == end.jd
    assert stmoc.min_time.jd == start.jd


def test_gen_mocs_epoch(dummy_moc: MOC,
                        dummy_stmoc: STMOC,
                        mocker) -> None:
    """
    Tests the generation of all MOCs and STMOCs for a single epoch.
    Also tests the update of the full STMOC.

    Args:
        dummy_moc: The dummy MOC object representing an open MOC.
        dummy_stmoc: The dummy STMOC object representing an open STMOC.
        mocker: The pytest mock mocker object.

    Returns:
        None
    """

    mocker_get_epoch_images = mocker.patch(
        'vasttools.tools._get_epoch_images',
        return_value=['test.fits']
    )
    mocker_gen_mocs_image = mocker.patch(
        'vasttools.tools.gen_mocs_image',
        return_value=(dummy_moc, dummy_stmoc)
    )
    mocker_moc_write = mocker.patch(
        'vasttools.tools.MOC.write'
    )
    mocker_stmoc_write = mocker.patch(
        'vasttools.tools.STMOC.write'
    )
    epoch = '1'
    vtt.gen_mocs_epoch(epoch, '', '', epoch_path='.')

    master_name = "VAST_PILOT_EPOCH{}.moc.fits".format(epoch)
    master_stmoc_name = master_name.replace("moc", "stmoc")
    pilot_stmoc_name = "VAST_PILOT.stmoc.fits"

    stmoc_calls = [mocker.call(Path(master_stmoc_name), overwrite=True),
                   mocker.call(Path(pilot_stmoc_name), overwrite=True)]

    mocker_moc_write.assert_called_once_with(Path(master_name),
                                             overwrite=True)
    mocker_stmoc_write.assert_has_calls(stmoc_calls)


def test__set_epoch_path_failure(mocker) -> None:
    """
    Tests the set_epoch_path function when `VAST_DATA_DIR` has not been set.

    Args:
        mocker: The pytest-mock mocker object.

    Returns:
        None
    """
    mocker_getenv = mocker.patch(
        'os.getenv', return_value=None
    )

    with pytest.raises(Exception) as excinfo:
        vtt._set_epoch_path('1')
    print(str(excinfo.value))
    assert str(excinfo.value).startswith(
        "The path to the requested epoch could not be determined!"
    )
