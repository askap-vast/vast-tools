import astropy.units as u
import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.io import fits
from astropy.wcs import WCS
from pytest_mock import mocker, MockerFixture  # noqa: F401
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


@pytest.fixture
def dummy_ax(dummy_fits_open):
    hdu = dummy_fits_open[0]
    wcs = WCS(hdu.header)
    fig = plt.figure()
    ax = fig.add_subplot(projection=wcs)

    return ax


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


def test_create_fields_metadata(mocker: MockerFixture) -> None:
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
    mocker_fields_sc = mocker.patch(
        'vasttools.tools._create_fields_sc',
        return_value=-99
    )
    mocker_to_csv = mocker.patch(
        'pandas.DataFrame.to_csv'
    )
    mocker_pickle_dump = mocker.patch(
        'pickle.dump'
    )

    mocker_open = mocker.patch('builtins.open', new_callable=mocker.mock_open)

    epoch_num = '2'
    csv_outfile = f'vast_epoch{int(epoch_num):02}_info.csv'
    sc_outfile = f'vast_epoch{int(epoch_num):02}_fields_sc.pickle'

    vtt.create_fields_metadata(epoch_num, TEST_DATA_DIR / 'surveys_db')

    mocker_to_csv.assert_called_once_with(Path(csv_outfile), index=False)

    mocker_open.assert_called_once_with(Path(sc_outfile), 'wb')

    mocker_pickle_dump.assert_called_once_with(
        mocker_fields_sc.return_value, mocker_open.return_value)


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
    mocker: MockerFixture
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


@pytest.mark.parametrize(
    "write",
    [(True), (False)],
    ids=("write", "no-write")
)
def test_gen_mocs_image(
        dummy_fits_open: fits.HDUList,
        dummy_load_fields_file: pd.DataFrame,
        dummy_moc: MOC,
        write: bool,
        mocker: MockerFixture) -> None:
    """
    Tests the generation of a MOC and STMOC for a single fits file

    Args:
        dummy_fits_open: The dummy HDUList object that represents an open
            FITS file.
        dummy_load_fields_file: The dummy fields file.
        dummy_moc: The dummy MOC object representing an open MOC.
        write: Whether to test the write to file or not.
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

    moc, stmoc = vtt.gen_mocs_image(fits_file, write=write)

    if write:
        mocker_moc_write.assert_called_once_with(Path(moc_file),
                                                 overwrite=True)

        mocker_stmoc_write.assert_called_once_with(Path(stmoc_file),
                                                   overwrite=True)

    assert stmoc.max_time.isclose(end, atol=1e-5 * u.s)
    assert stmoc.min_time.isclose(start, atol=1e-5 * u.s)


def test_gen_mocs_epoch(dummy_moc: MOC,
                        dummy_stmoc: STMOC,
                        mocker: MockerFixture) -> None:
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


def test_gen_mocs_epoch_outdir_failure(mocker: MockerFixture) -> None:
    """
    Tests the generation of all MOCs and STMOCs for a single epoch.
    Also tests the update of the full STMOC.

    Args:
        mocker: The pytest mock mocker object.

    Returns:
        None
    """

    outdir = '/path/to/outdir'

    mocker_exists = mocker.patch('vasttools.tools.Path.exists',
                                 return_value=False
                                 )

    with pytest.raises(Exception) as excinfo:
        vtt.gen_mocs_epoch('1',
                           '',
                           '',
                           outdir=outdir
                           )

    exc_str = "{} does not exist".format(outdir)

    assert str(excinfo.value) == exc_str


def test_gen_mocs_epoch_stmoc_failure(mocker: MockerFixture) -> None:
    """
    Tests the generation of all MOCs and STMOCs for a single epoch.
    Also tests the update of the full STMOC.

    Args:
        mocker: The pytest mock mocker object.

    Returns:
        None
    """

    base_stmoc = '/path/to/stmoc.fits'

    mocker_exists = mocker.patch('vasttools.tools.Path.exists',
                                 side_effect=[True, False]
                                 )

    with pytest.raises(Exception) as excinfo:
        vtt.gen_mocs_epoch('1',
                           '',
                           '',
                           epoch_path='.',
                           base_stmoc=base_stmoc
                           )

    exc_str = "{} does not exist".format(base_stmoc)
    assert str(excinfo.value) == exc_str


def test__set_epoch_path_failure(mocker: MockerFixture) -> None:
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

    assert str(excinfo.value).startswith(
        "The path to the requested epoch could not be determined!"
    )


@pytest.mark.parametrize(
    "ra_units,dec_units",
    [
        (u.m, u.arcsec),
        (u.arcsec, u.m),
        (u.m, u.m)
    ],
    ids=('bad-ra',
         'bad-dec',
         'bad-both',
         )
)
def test_offset_postagestamp_axes_errors(dummy_ax: plt.Axes,
                                         ra_units: u.core.Unit,
                                         dec_units: u.core.Unit
                                         ):
    """
    Tests the failures when using offset_postagestamp_axes.

    Args:
        dummy_ax: The dummy axis object to be updated.
        ra_units: The units to use for the Right Ascension axis
        dec_units: The units to use for the Declination axis

    Returns:
        None
    """

    coord = SkyCoord(319.6519091667, -6.2985525, unit=u.deg)

    with pytest.raises(Exception) as excinfo:
        vtt.offset_postagestamp_axes(dummy_ax,
                                     coord,
                                     ra_units=ra_units,
                                     dec_units=dec_units
                                     )
    assert str(excinfo.value) == "R.A. and Dec. units must be angles."


@pytest.mark.parametrize(
    "ra_units,dec_units,ra_label,dec_label,"
    "major_tick_length,minor_tick_length",
    [
        (u.arcsec, u.arcsec, 'R.A. Offset', 'Dec. Offset', 6, 3),
        (u.deg, u.deg, 'R.A. Offset', 'Dec. Offset', 6, 3),
        (u.arcmin, u.arcmin, 'R.A. Offset', 'Dec. Offset', 6, 3),
        (u.arcsec, u.arcsec, '$\\Delta$ R.A.', '$\\Delta$ Dec.', 6, 3),
    ],
    ids=('base',
         'units_deg',
         'units_arcsec',
         'latex_labels'
         )
)
def test_offset_postagestamp_axes(dummy_ax: plt.Axes,
                                  ra_units: u.core.Unit,
                                  dec_units: u.core.Unit,
                                  ra_label: str,
                                  dec_label: str,
                                  major_tick_length: Union[int, float],
                                  minor_tick_length: Union[int, float],
                                  ):
    """
    Tests the offset_postagestamp_axes function.

    Args:
        dummy_ax: The dummy axis object to be updated.
        ra_units: The units to use for the Right Ascension axis
        dec_units: The units to use for the Declination axis
        ra_label: The label to use for the Right Ascension axis
        dec_label: The label to use for the Declination axis
        major_tick_length: Major tick length in points
        minor_tick_length: Minor tick length in points

    Returns:
        None
    """

    centre_sc = SkyCoord(319.6519091667, -6.2985525, unit=u.deg)

    vtt.offset_postagestamp_axes(dummy_ax,
                                 centre_sc,
                                 ra_units=ra_units,
                                 dec_units=dec_units,
                                 ra_label=ra_label,
                                 dec_label=dec_label,
                                 major_tick_length=major_tick_length,
                                 minor_tick_length=minor_tick_length
                                 )

    ra_off, dec_off = dummy_ax.overlay_coords

    assert ra_off.get_axislabel() == ra_label
    assert dec_off.get_axislabel() == dec_label

    assert ra_off.get_format_unit() == ra_units
    assert dec_off.get_format_unit() == dec_units
