import astropy.units as u
import numpy as np
import pandas as pd
import pytest

from astropy.coordinates import SkyCoord, Angle
from astropy.table import Table
from astropy.io import fits
from pytest_mock import mocker, MockerFixture  # noqa: F401

import vasttools.utils as vtu


@pytest.fixture
def coords_df() -> pd.DataFrame:
    """
    Produces a dataframe of ra and dec coordinates in degrees.

    Returns:
        Dataframe of ra and dec coordinates.
    """
    df = pd.DataFrame.from_dict(
        {
            "ra": [180., 270., 180.],
            "dec": [60., -60., 60.5]
        }
    )

    return df


@pytest.fixture
def coords_skycoord(coords_df: pd.DataFrame) -> SkyCoord:
    """
    Produces a SkyCoord object from a ra and dec dataframe.

    Args:
        coords_df: The input dataframe that contains the ra and dec columns.

    Returns:
        SkyCoord object produced from dataframe.
    """
    df_skycoord = SkyCoord(
        coords_df['ra'].to_numpy(),
        coords_df['dec'].to_numpy(),
        unit=(u.deg, u.deg)
    )

    return df_skycoord


@pytest.fixture
def catalog_deg_float() -> pd.DataFrame:
    """
    Produces a dataframe of ra and dec coordinates in degrees as floats
    with source names attached.

    Returns:
        Dataframe of ra and dec coordinates and source name.
    """
    df = pd.DataFrame(
        data={
            'ra': [322.4387083, 180., 270.],
            'dec': [-4.4866389, 60., -60.],
            'name': ['PSR J2129-04', 'Test1', 'Test2']
        }
    )

    return df


@pytest.fixture
def catalog_deg_string() -> pd.DataFrame:
    """
    Produces a dataframe of ra and dec coordinates in degrees as strings
    with source names attached.

    Returns:
        Dataframe of ra and dec coordinates and source name.
    """
    df = pd.DataFrame(
        data={
            'ra': ['322.4387083', '180.0', '270.0'],
            'dec': ['-4.4866389', '60.0', '-60.0'],
            'name': ['PSR J2129-04', 'Test1', 'Test2']
        },
    )

    return df


@pytest.fixture
def catalog_hms_string() -> pd.DataFrame:
    """
    Produces a dataframe of ra and dec coordinates in hms format as strings
    with source names attached.

    Returns:
        Dataframe of ra and dec coordinates and source name.
    """
    df = pd.DataFrame(
        data={
            'ra': ['21:29:45.29', '12:00:00.00', '18:00:00.00'],
            'dec': ['-04:29:11.90', '60:00:00.00', '-60:00:00.00'],
            'name': ['PSR J2129-04', 'Test1', 'Test2']
        },
    )

    return df


@pytest.fixture
def catalog_skycoord(catalog_deg_float: pd.DataFrame) -> SkyCoord:
    """
    Produces a SkyCoord object from a ra and dec catalog dataframe.

    Args:
        catalog_deg_float:
            The input dataframe that contains the ra and dec columns.

    Returns:
        SkyCoord object produced from dataframe.
    """
    df_skycoord = SkyCoord(
        catalog_deg_float['ra'].to_numpy(),
        catalog_deg_float['dec'].to_numpy(),
        unit=(u.deg, u.deg)
    )

    return df_skycoord


@pytest.fixture
def catalog_skycoord_hms(catalog_hms_string: pd.DataFrame) -> SkyCoord:
    """
    Produces a SkyCoord object from a ra and dec catalog dataframe, which is
    in hms dms format.

    Args:
        catalog_hms_string:
            The input dataframe that contains the ra and dec columns in the
            format of hms/dms strings.

    Returns:
        SkyCoord object produced from dataframe.
    """
    df_skycoord = SkyCoord(
        catalog_hms_string['ra'].to_numpy(),
        catalog_hms_string['dec'].to_numpy(),
        unit=(u.hourangle, u.deg)
    )

    return df_skycoord


@pytest.fixture
def planet_fields() -> pd.DataFrame:
    """
    The expected fields dataframe for a jupiter planet search.

    Returns:
        Dataframe of Jupiter planet search.
    """
    fields = pd.DataFrame(
        data={
            'epoch': ['3x', '8'],
            'FIELD_NAME': ['VAST_1739-25A', 'VAST_0241+00A'],
            'SBID': [10335, 11383],
            'DATEOBS': ['2019-10-29 12:34:02.450', '2020-01-17 11:45:29.346'],
            'centre-ra': [264.90476978971117, 40.34340569444702],
            'centre-dec': [-25.136207332844116, 0.0037709444443794287],
            'planet': ['jupiter', 'jupiter']
        }
    )

    return fields


@pytest.fixture
def source_df() -> pd.DataFrame:
    """
    Produces a dataframe containing source flux values for testing. Contains
    peak, integrated and errors.

    Returns:
        Dataframe with flux values.
    """
    source_df = pd.DataFrame(
        data={
            'flux_peak': [
                -0.1536501944065094,
                -0.11119924485683441,
                -0.7933286428451538,
                1.0169366598129272,
                3.0840001106262207,
                7.142000198364258
            ],
            'flux_peak_err': [
                0.2477360963821411,
                0.24757413566112518,
                0.2197069227695465,
                0.2355700582265854,
                0.26184606552124023,
                0.2906762361526489
            ],
            'flux_int': [
                -0.1536501944065094,
                -0.11119924485683441,
                -0.7933286428451538,
                1.0169366598129272,
                3.947999954223633,
                8.831000328063965
            ],
            'flux_int_err': [
                0.2477360963821411,
                0.24757413566112518,
                0.2197069227695465,
                0.2355700582265854,
                0.5366007089614868,
                0.5815157890319824
            ]
        }
    )

    return source_df


@pytest.fixture
def dummy_selavy_components_astropy() -> Table:
    """
    Provides a dummy set of selavy components containing only the columns
    required for testing.
    Returned as a pandas dataframe.
    Returns:
        The dataframe containing the dummy selavy components.
    """
    df = pd.DataFrame(data={
        'island_id': {
            0: 'SB9667_island_1000',
            1: 'SB9667_island_1001',
            2: 'SB9667_island_1002',
            3: 'SB9667_island_1003',
            4: 'SB9667_island_1004'
        },
        'ra_deg_cont': {
            0: 321.972731,
            1: 317.111595,
            2: 322.974588,
            3: 315.077869,
            4: 315.56781
        },
        'dec_deg_cont': {
            0: 0.699851,
            1: 0.53981,
            2: 1.790072,
            3: 3.011253,
            4: -0.299919
        },
        'maj_axis': {0: 15.6, 1: 18.48, 2: 21.92, 3: 16.77, 4: 14.67},
        'min_axis': {0: 14.23, 1: 16.03, 2: 16.67, 3: 12.4, 4: 13.64},
        'pos_ang': {0: 111.96, 1: 43.18, 2: 22.71, 3: 57.89, 4: 63.43},
        'flux_peak': {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0},
        'flux_peak_err': {0: 0.5, 1: 0.2, 2: 0.1, 3: 0.2, 4: 0.3}
    })

    return Table.from_pandas(df)


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
def dummy_fits_open_large() -> fits.HDUList:
    border_width = 50
    image_width = 1000

    data = np.ones((image_width, image_width))
    data[:border_width, :] = np.nan
    data[-border_width:, :] = np.nan
    data[:, :border_width] = np.nan
    data[:, -border_width:] = np.nan

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
def dummy_fits_open_large_hole() -> fits.HDUList:
    border_width = 50
    image_width = 1000
    hole_width = 100

    centre = int(image_width / 2)
    hole_rad = int(hole_width / 2)

    data = np.ones((image_width, image_width))
    data[:border_width, :] = np.nan
    data[-border_width:, :] = np.nan
    data[:, :border_width] = np.nan
    data[:, -border_width:] = np.nan

    data[centre - hole_rad:centre + hole_rad,
         centre - hole_rad:centre + hole_rad
         ] = np.nan

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


def test_gen_skycoord_from_df(
    coords_df: pd.DataFrame,
    coords_skycoord: SkyCoord
) -> None:
    """
    Tests the creation of a SkyCoord object from a dataframe.

    Args:
        coords_df: Dataframe containing the coordinate columns.
        coords_skycoord: Expected SkyCoord.

    Returns:
        None.
    """
    vtu_sc = vtu.gen_skycoord_from_df(coords_df)

    assert np.all(coords_skycoord == vtu_sc)


def test_gen_skycoord_from_df_hms(
    catalog_hms_string: pd.DataFrame,
    catalog_skycoord_hms: SkyCoord
) -> None:
    """
    Tests the creation of a SkyCoord object from a dataframe with coordinates
    in the format of hms dms string.

    Args:
        catalog_hms_string: Dataframe containing the coordinate columns.
        catalog_skycoord_hms: Expected SkyCoord.

    Returns:
        None.
    """
    vtu_sc = vtu.gen_skycoord_from_df(
        catalog_hms_string,
        ra_unit=u.hourangle,
    )

    assert np.all(catalog_skycoord_hms == vtu_sc)


def test_gen_skycoord_from_df_colnames(
    coords_df: pd.DataFrame,
    coords_skycoord: SkyCoord
) -> None:
    """
    Tests the creation of a SkyCoord object from a dataframe with custom
    column names.

    Args:
        coords_df: Dataframe containing the coordinate columns.
        coords_skycoord: Expected SkyCoord.

    Returns:
        None.
    """
    coords_df = coords_df.rename(columns={
        'ra': 'theRA',
        'dec': 'theDEC'
    })
    vtu_sc = vtu.gen_skycoord_from_df(
        coords_df, ra_col='theRA', dec_col='theDEC'
    )

    assert np.all(coords_skycoord == vtu_sc)


def test_check_file(mocker: MockerFixture) -> None:
    """
    Tests check file returns correctly.

    Args:
        mocker: Pytest mock mocker object.

    Returns:
        None.
    """
    mocker_isfile = mocker.patch('os.path.isfile', return_value=True)

    test_file = '/test/file/existence.txt'

    exists = vtu.check_file(test_file)

    mocker_isfile.assert_called_once_with(test_file)


def test_check_racs_exists(mocker: MockerFixture) -> None:
    """
    Tests the RACS check.

    Args:
        mocker: Pytest mock mocker object.

    Returns:
        None.
    """
    mocker_isdir = mocker.patch('os.path.isdir', return_value=True)
    exists = vtu.check_racs_exists('/data/release/path')

    mocker_isdir.assert_called_once_with('/data/release/path/EPOCH00')
    assert exists is True


def test_create_source_directories(mocker: MockerFixture) -> None:
    """
    Tests the source directories creation.

    Args:
        mocker: Pytest mock mocker object.

    Returns:
        None.
    """
    mocker_makedirs = mocker.patch('os.makedirs', return_value=None)

    base_dir = '/path/to/base/dir/'
    source_names = ['source1', 'source2']

    vtu.create_source_directories(base_dir, source_names)

    calls = [
        mocker.call('/path/to/base/dir/source1'),
        mocker.call('/path/to/base/dir/source2')
    ]

    mocker_makedirs.assert_has_calls(calls)


def test_filter_selavy_components(
    coords_df: pd.DataFrame,
    coords_skycoord: SkyCoord
) -> None:
    """
    Tests filter selavy components function that should return selavy
    components only contained with in a specified location.

    Args:
        coords_df: Dataframe containing the coordinate columns.
        coords_skycoord: SkyCoord to use.

    Returns:
        None.
    """
    target_skycoord = coords_skycoord[0]

    size = Angle(1 * u.deg)

    filtered = vtu.filter_selavy_components(
        coords_df,
        coords_skycoord,
        size,
        target_skycoord
    )

    expected = coords_df.drop(1).reset_index(drop=True)

    assert filtered.equals(expected)


def test_build_catalog_file(
    catalog_deg_float: pd.DataFrame,
    mocker: MockerFixture,
) -> None:
    """
    Tests the build catalog function.

    Args:
        coords_deg_float: Dataframe containing the coordinate columns.
            This is the expected result.
        mocker: Pytest mock mocker object.

    Returns:
        None.
    """
    filename = 'testinput.csv'
    source_names = ""

    mocker_abspath = mocker.patch('os.path.abspath', return_value=filename)
    mocker_isfile = mocker.patch('os.path.isfile', return_value=True)
    mocker_read_csv = mocker.patch(
        'vasttools.utils.pd.read_csv',
        return_value=catalog_deg_float
    )

    catalog = vtu.build_catalog(filename, source_names)

    mocker_read_csv.assert_called_once_with(filename, comment='#')
    assert catalog.equals(catalog_deg_float)


def test_build_catalog_string_deg(catalog_deg_string: pd.DataFrame) -> None:
    """
    Tests catalog creation with string coordinates in degrees.

    Args:
        catalog_deg_string: Dataframe containing the coordinate columns.

    Returns:
        None.
    """
    coords = "322.4387083 -4.4866389,180.0 60.0,270.0 -60.0"
    source_names = "PSR J2129-04,Test1,Test2"

    catalog = vtu.build_catalog(coords, source_names)

    assert catalog[['ra', 'dec', 'name']].equals(catalog_deg_string)


def test_build_catalog_string_hms(catalog_hms_string: pd.DataFrame) -> None:
    """
    Tests catalog creation with string coordinates in hms format.

    Args:
        catalog_hms_string: Dataframe containing the coordinate columns.

    Returns:
        None.
    """
    coords = (
        "21:29:45.29 -04:29:11.90,12:00:00.00 60:00:00.00,"
        "18:00:00.00 -60:00:00.00"
    )
    source_names = "PSR J2129-04,Test1,Test2"

    catalog = vtu.build_catalog(coords, source_names)

    assert catalog[['ra', 'dec', 'name']].equals(catalog_hms_string)


def test_build_SkyCoord_float_deg(
    catalog_deg_float: pd.DataFrame,
    catalog_skycoord: SkyCoord
) -> None:
    """
    Tests build SkyCoord from the dataframe in float deg format.

    Args:
        catalog_deg_float: Dataframe containing the coordinate columns.
        catalog_skycoord: Expected SkyCoord result.

    Returns:
        None.
    """
    result = vtu.build_SkyCoord(catalog_deg_float)

    assert np.all(result == catalog_skycoord)


def test_build_SkyCoord_string_deg(
    catalog_deg_string,
    catalog_skycoord
) -> None:
    """
    Tests build SkyCoord from the dataframe in string deg format.

    Args:
        catalog_deg_string: Dataframe containing the coordinate columns.
        catalog_skycoord: Expected SkyCoord result.

    Returns:
        None.
    """
    result = vtu.build_SkyCoord(catalog_deg_string)

    assert np.all(result == catalog_skycoord)


def test_build_SkyCoord_string_hms(
    catalog_hms_string: pd.DataFrame,
    catalog_skycoord_hms: SkyCoord
) -> None:
    """
    Tests build SkyCoord from the dataframe in string hms format.

    Args:
        catalog_hms_string: Dataframe containing the coordinate columns.
        catalog_skycoord: Expected SkyCoord result.

    Returns:
        None.
    """
    result = vtu.build_SkyCoord(catalog_hms_string)

    assert np.all(result == catalog_skycoord_hms)


def test_read_selavy_xml(mocker: MockerFixture):
    """
    Tests read_selavy for a file with xml formatting.
    Args:
        mocker: Pytest mock mocker object.
    Returns:
        None.
    """
    mock_table_read = mocker.patch(
        'vasttools.utils.Table.read')

    test_filename = 'test.xml'
    vtu.read_selavy(test_filename)
    mock_table_read.assert_called_once_with(test_filename,
                                            format="votable",
                                            use_names_over_ids=True
                                            )


def test_read_selavy_xml_usecols(dummy_selavy_components_astropy,
                                 mocker: MockerFixture
    ) -> None:
    """
    Tests read_selavy for a file with xml formatting, requesting a subset
    of the available columns.
    Args:
        dummy_selavy_components_astropy: A dummy astropy Table containing
            the necessary selavy columns
        mocker: Pytest mock mocker object.
    Returns:
        None.
    """
    mock_table_read = mocker.patch(
        'vasttools.utils.Table.read',
        return_value=dummy_selavy_components_astropy
    )

    test_filename = 'test.xml'
    usecols = ['island_id',
               'ra_deg_cont',
               'dec_deg_cont',
               'maj_axis',
               'min_axis',
               'pos_ang'
               ]

    df = vtu.read_selavy(test_filename, cols=usecols)

    assert list(df.columns) == usecols


def test_read_selavy_fwf(mocker: MockerFixture) -> None:
    """
    Tests read_selavy for a file with standard fixed-width formatting.
    Args:
        mocker: Pytest mock mocker object.
    Returns:
        None.
    """
    mock_table_read = mocker.patch(
        'vasttools.utils.pd.read_fwf')

    test_filename = 'test.txt'
    vtu.read_selavy(test_filename)
    mock_table_read.assert_called_once_with(test_filename,
                                            skiprows=[1],
                                            usecols=None
                                            )


def test_read_selavy_csv(mocker) -> None:
    """
    Tests read_selavy for a file with csv formatting.
    Args:
        mocker: Pytest mock mocker object.
    Returns:
        None.
    """
    mock_table_read = mocker.patch(
        'vasttools.utils.pd.read_csv')

    test_filename = 'test.csv'
    vtu.read_selavy(test_filename)
    mock_table_read.assert_called_once_with(test_filename,
                                            usecols=None
                                            )


def test_simbad_search(mocker: MockerFixture) -> None:
    """
    Test the SIMBAD search.

    This test doesn't actually call the SIMBAD service, it only tests
    using a mocked astropy table that has the three columns of interest.
    Any changes to the SIMBAD astroquery service will need to accounted for.

    Args:
        mocker: Pytest mock mocker object.

    Returns:
        None
    """
    objects = ['PSR J2129-04', 'SN 2012dy']
    ra = np.array([322.43871000, 319.71125000], dtype=np.float64) * u.deg
    dec = np.array([-4.48664000, -57.64514000], dtype=np.float64) * u.deg

    simbad_response = Table(
        [ra, dec, objects],
        names=('RA_d', 'DEC_d', 'TYPED_ID')
    )

    mocker_simbad_query = mocker.patch(
        'vasttools.utils.Simbad.query_objects',
        return_value=simbad_response
    )

    simbad_skycoord = SkyCoord.guess_from_table(simbad_response)

    result_skycoord, result_names = vtu.simbad_search(objects)

    mocker_simbad_query.assert_called_once_with(objects)
    assert np.all(result_skycoord == simbad_skycoord)
    assert np.all(result_names == np.array(objects))


def test_simbad_search_none(mocker: MockerFixture) -> None:
    """
    Test the SIMBAD search None result.

    Args:
        mocker: Pytest mock mocker object.

    Returns:
        None
    """
    objects = ['PSR J2129-04', 'SN 2012dy']

    mocker_simbad_query = mocker.patch(
        'vasttools.utils.Simbad.query_objects',
        return_value=None
    )

    result_skycoord, result_names = vtu.simbad_search(objects)

    mocker_simbad_query.assert_called_once_with(objects)
    assert result_skycoord == result_names is None


@pytest.mark.parametrize("sep_thresh", [4.0, 3.0])
def test_match_planet_to_field_sep_4deg(
    sep_thresh: float,
    planet_fields: pd.DataFrame
) -> None:
    """
    Tests the matching of a planet to a field.

    The only columns that matter are: planet, DATEOBS, centre-ra
    and centre-dec. This te

    Args:
        planet_fields: Dataframe containing an example planets search.
        sep_thresh: The separation threshold to test (from parametrize).

    Returns:
        None
    """
    matches = vtu.match_planet_to_field(planet_fields, sep_thresh=sep_thresh)

    if sep_thresh == 4.0:
        assert matches.shape[0] == 1
        assert matches['epoch'].iloc[0] == '3x'
    else:
        assert matches.empty


@pytest.mark.parametrize(
    "peak,expected",
    [
        (True, 121.89841839024031),
        (False, 59.80572010933274),
        (False, 0.0)
    ]
)
def test_pipeline_get_eta_metric_peak(
    peak: bool,
    expected: float,
    source_df: pd.DataFrame
) -> None:
    """
    Tests the calculation of the eta metric.

    Args:
        peak: Whether to use peak flux when calculating the eta.
        expected: Expected eta value.
        source_df: The dataframe containing the source data.

    Returns:
        None
    """
    if expected == 0.0:
        source_df = source_df.drop(source_df.index[1:])

    eta = vtu.pipeline_get_eta_metric(source_df, peak=peak)

    if expected == 0.0:
        assert eta == expected
    else:
        assert eta == pytest.approx(expected)


def test_pipeline_get_variable_metrics(source_df: pd.DataFrame) -> None:
    """
    Tests the calculation of all variable metrics.

    Args:
        source_df: The dataframe containing the source data.

    Returns:
        None
    """
    results = vtu.pipeline_get_variable_metrics(source_df)

    assert results['eta_peak'] == pytest.approx(121.89841839024031)
    assert results['v_peak'] == pytest.approx(1.7659815567107082)
    assert results['eta_int'] == pytest.approx(59.80572010933274)
    assert results['v_int'] == pytest.approx(1.740059768483511)


def test_pipeline_get_variable_metrics_zero(source_df: pd.DataFrame) -> None:
    """
    Tests the calculation of the eta metric when zero is expected.

    Args:
        peak: Whether to use peak flux when calculating the eta.
        expected: Expected eta value.
        source_df: The dataframe containing the source data.

    Returns:
        None
    """
    results = vtu.pipeline_get_variable_metrics(
        source_df.drop(source_df.index[1:])
    )

    assert np.all(results.to_numpy() == 0.0)


def test_calculate_vs_metric() -> None:
    """
    Tests the calculation of the vs two epoch metric.

    Returns:
        None
    """
    flux_a = 200.
    flux_b = 100.
    flux_err_a = 3.
    flux_err_b = 4.

    result = vtu.calculate_vs_metric(flux_a, flux_b, flux_err_a, flux_err_b)

    assert result == 20.


def test_calculate_m_metric() -> None:
    """
    Tests the calculation of the m two epoch metric.

    Returns:
        None
    """
    flux_a = 2.
    flux_b = 1.

    result = vtu.calculate_m_metric(flux_a, flux_b)

    assert result == 2. / 3.


def test_create_moc_from_fits(
    dummy_fits_open: fits.HDUList,
    mocker: MockerFixture
) -> None:
    """
    Tests the generation of a MOC for a single fits file.

    Args:
        dummy_fits_open: The dummy HDUList object that represents an open
            FITS file.
        mocker: The pytest mock mocker object.

    Returns:
        None
    """

    mocker_fits_open = mocker.patch(
        'vasttools.utils.fits.open',
        return_value=dummy_fits_open
    )
    mocker_fits_open = mocker.patch(
        'os.path.isfile',
        return_value=True
    )

    moc_out_json = {'9': [1215087, 1215098]}

    moc = vtu.create_moc_from_fits('test.fits', 9)
    moc_json = moc.serialize(format='json')

    assert moc_json == moc_out_json


def test_mocs_with_holes(dummy_fits_open_large,
                         dummy_fits_open_large_hole,
    """
    Tests that gen_mocs_field produces the same output regardless of whether
    there are NaN holes within the image.

    Args:
        dummy_fits_open_large: The dummy HDUList object that represents an open
            FITS file with a large data array of ones.
        dummy_fits_open_large_hole: The dummy HDUList object that represents
            an open FITS file with a large data array of ones, with a hole of
            NaN values in the centre.
        mocker: The pytest mock mocker object.
    Returns:
        None
    """
    mocker_fits_open = mocker.patch(
        'os.path.isfile',
        return_value=True
    )

    mocker_fits_open = mocker.patch(
        'vasttools.utils.fits.open',
        return_value=dummy_fits_open_large
    )
    full_moc = vtu.create_moc_from_fits('test.fits')

    mocker_fits_open = mocker.patch(
        'vasttools.utils.fits.open',
        return_value=dummy_fits_open_large_hole
    )
    hole_moc = vtu.create_moc_from_fits('test.fits')

    assert full_moc == hole_moc


def test__distance_from_edge() -> None:
    """
    Tests the distance from edge method.

    The function works by calculating how far the pixel is from the edge
    (i.e. zero pixels). The expected result is defined in the test.

    Args:
        None

    Returns:
        None
    """
    input_array = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0]
        ]
    )

    expected = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 2, 2, 1, 0],
            [0, 1, 2, 2, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0]
        ]
    )

    result = vtu._distance_from_edge(input_array)

    assert np.all(result == expected)
