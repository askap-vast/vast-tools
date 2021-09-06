import astropy.units as u
import numpy as np
import pandas as pd
import pytest

from astropy.coordinates import SkyCoord, Angle
from astropy.table import Table
from pytest_mock import mocker

import vasttools.utils as vtu


@pytest.fixture
def coords_df():
    df = pd.DataFrame.from_dict(
        {
            "ra": [180., 270., 180.],
            "dec": [60., -60., 60.5]
        }
    )

    return df


@pytest.fixture
def coords_skycoord(coords_df):
    df_skycoord = SkyCoord(
        coords_df['ra'].to_numpy(),
        coords_df['dec'].to_numpy(),
        unit=(u.deg, u.deg)
    )

    return df_skycoord


@pytest.fixture
def catalog_deg_float():
    df = pd.DataFrame(
        data = {
            'ra': [322.4387083, 180., 270.],
            'dec': [-4.4866389, 60., -60.],
            'name': ['PSR J2129-04', 'Test1', 'Test2']
        }
    )

    return df


@pytest.fixture
def catalog_deg_string():
    df = pd.DataFrame(
        data = {
            'ra': ['322.4387083', '180.0', '270.0'],
            'dec': ['-4.4866389', '60.0', '-60.0'],
            'name': ['PSR J2129-04', 'Test1', 'Test2']
        },
    )

    return df


@pytest.fixture
def catalog_hms_string():
    df = pd.DataFrame(
        data = {
            'ra': ['21:29:45.29', '12:00:00.00', '18:00:00.00'],
            'dec': ['-04:29:11.90', '60:00:00.00', '-60:00:00.00'],
            'name': ['PSR J2129-04', 'Test1', 'Test2']
        },
    )

    return df


@pytest.fixture
def catalog_skycoord(catalog_deg_float):
    df_skycoord = SkyCoord(
        catalog_deg_float['ra'].to_numpy(),
        catalog_deg_float['dec'].to_numpy(),
        unit=(u.deg, u.deg)
    )

    return df_skycoord


@pytest.fixture
def catalog_skycoord_hms(catalog_hms_string):
    df_skycoord = SkyCoord(
        catalog_hms_string['ra'].to_numpy(),
        catalog_hms_string['dec'].to_numpy(),
        unit=(u.hourangle, u.deg)
    )

    return df_skycoord


@pytest.fixture
def planet_fields():
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
def source_df():
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


def test_gen_skycoord_from_df(coords_df, coords_skycoord):
    vtu_sc = vtu.gen_skycoord_from_df(coords_df)

    assert np.all(coords_skycoord == vtu_sc)


def test_gen_skycoord_from_df_hms(catalog_hms_string, catalog_skycoord_hms):
    vtu_sc = vtu.gen_skycoord_from_df(
        catalog_hms_string,
        ra_unit=u.hourangle,
    )

    assert np.all(catalog_skycoord_hms == vtu_sc)


def test_gen_skycoord_from_df_colnames(coords_df, coords_skycoord):
    coords_df = coords_df.rename(columns={
        'ra': 'theRA',
        'dec': 'theDEC'
    })
    vtu_sc = vtu.gen_skycoord_from_df(
        coords_df, ra_col='theRA', dec_col='theDEC'
    )

    assert np.all(coords_skycoord == vtu_sc)


def test_check_file(mocker):
    mocker_isfile = mocker.patch('os.path.isfile', return_value=True)

    test_file = '/test/file/existence.txt'

    exists = vtu.check_file(test_file)

    mocker_isfile.assert_called_once_with(test_file)


def test_check_racs_exists(mocker):
    mocker_isdir = mocker.patch('os.path.isdir', return_value=True)
    exists = vtu.check_racs_exists('/data/release/path')

    mocker_isdir.assert_called_once_with('/data/release/path/EPOCH00')
    assert exists == True


def test_create_source_directories(mocker):
    mocker_makedirs = mocker.patch('os.makedirs', return_value=None)

    base_dir = '/path/to/base/dir/'
    source_names = ['source1', 'source2']

    vtu.create_source_directories(base_dir, source_names)

    calls = [
        mocker.call('/path/to/base/dir/source1'),
        mocker.call('/path/to/base/dir/source2')
    ]

    mocker_makedirs.assert_has_calls(calls)


def test_filter_selavy_components(coords_df, coords_skycoord):
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


def test_build_catalog_file(mocker, catalog_deg_float):
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


def test_build_catalog_string_deg(catalog_deg_string):
    coords = "322.4387083 -4.4866389,180.0 60.0,270.0 -60.0"
    source_names = "PSR J2129-04,Test1,Test2"

    catalog = vtu.build_catalog(coords, source_names)

    assert catalog[['ra', 'dec', 'name']].equals(catalog_deg_string)


def test_build_catalog_string_hms(catalog_hms_string):
    coords = (
        "21:29:45.29 -04:29:11.90,12:00:00.00 60:00:00.00,"
        "18:00:00.00 -60:00:00.00"
    )
    source_names = "PSR J2129-04,Test1,Test2"

    catalog = vtu.build_catalog(coords, source_names)

    assert catalog[['ra', 'dec', 'name']].equals(catalog_hms_string)


def test_build_SkyCoord_float_deg(catalog_deg_float, catalog_skycoord):
    result = vtu.build_SkyCoord(catalog_deg_float)

    assert np.all(result == catalog_skycoord)


def test_build_SkyCoord_string_deg(catalog_deg_string, catalog_skycoord):
    result = vtu.build_SkyCoord(catalog_deg_string)


def test_build_SkyCoord_string_hms(catalog_hms_string, catalog_skycoord_hms):
    result = vtu.build_SkyCoord(catalog_hms_string)

    assert np.all(result == catalog_skycoord_hms)


def test_simbad_search(mocker):
    """
    This test doesn't actually call the SIMBAD service, it only tests
    using a mocked astropy table that has the three columns of interest.
    Any changes to the SIMBAD astroquery service will need to accounted for.
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


def test_simbad_search_none(mocker):
    objects = ['PSR J2129-04', 'SN 2012dy']

    mocker_simbad_query = mocker.patch(
        'vasttools.utils.Simbad.query_objects',
        return_value=None
    )

    result_skycoord, result_names = vtu.simbad_search(objects)

    mocker_simbad_query.assert_called_once_with(objects)
    assert result_skycoord == result_names == None


def test_match_planet_to_field_sep_4deg(planet_fields):
    """
    Tests the matching of a planet to a field. The only columns that matter
    are: planet, DATEOBS, centre-ra and centre-dec.
    """
    # use default sep_thresh of 4.0
    matches = vtu.match_planet_to_field(planet_fields)

    assert matches.shape[0] == 1
    assert matches['epoch'].iloc[0] == '3x'


def test_match_planet_to_field_sep_3deg(planet_fields):
    """
    Tests the matching of a planet to a field. The only columns that matter
    are: planet, DATEOBS, centre-ra and centre-dec.
    """
    matches = vtu.match_planet_to_field(planet_fields, sep_thresh=3.0)

    assert matches.empty


def test_pipeline_get_eta_metric_peak(source_df):
    eta = vtu.pipeline_get_eta_metric(source_df, peak=True)

    assert eta == pytest.approx(121.89841839024031)


def test_pipeline_get_eta_metric_int(source_df):
    eta = vtu.pipeline_get_eta_metric(source_df)

    assert eta == pytest.approx(59.80572010933274)


def test_pipeline_get_eta_metric_zero(source_df):
    eta = vtu.pipeline_get_eta_metric(source_df.drop(source_df.index[1:]))

    assert eta == 0.0


def test_pipeline_get_variable_metrics(source_df):
    results = vtu.pipeline_get_variable_metrics(source_df)

    assert results['eta_peak'] == pytest.approx(121.89841839024031)
    assert results['v_peak'] == pytest.approx(1.7659815567107082)
    assert results['eta_int'] == pytest.approx(59.80572010933274)
    assert results['v_int'] == pytest.approx(1.740059768483511)


def test_pipeline_get_variable_metrics_zero(source_df):
    results = vtu.pipeline_get_variable_metrics(
        source_df.drop(source_df.index[1:])
    )

    assert np.all(results.to_numpy() == 0.0)
