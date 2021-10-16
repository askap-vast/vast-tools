import astropy.units as u
import numpy as np
import pandas as pd
import pytest

from astropy.coordinates import SkyCoord, Angle
from astropy.table import Table
from astropy.io import fits
from pytest_mock import mocker
from pathlib import Path

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


def test_create_fields_csv(tmp_path: Path) -> None:
    """
    Tests creating the fields csv for a single epoch.

    Args:
        tmp_path: The default pytest temporary path

    Returns:
        None
    """
    vtt.create_fields_csv('2', TEST_DATA_DIR / 'surveys_db', tmp_path)

    out_df = pd.read_csv(tmp_path / 'vast_epoch2_info.csv')
    expected_df = pd.read_csv(TEST_DATA_DIR / 'vast_epoch2_info.csv')

    pd.testing.assert_frame_equal(out_df, expected_df)


def test_add_obs_date(mocker):
    """
    Tests adding observation dates to fits images

    Args:
        mocker: The pytest mock mocker object.

    Returns:
        None
    """

    def fits_open_no_update(filename):
        return fits.open(filename)

    test_img_path = str(TEST_DATA_DIR / 'VAST_0012-06A.EPOCH01.I.TEST.fits')

    mocker_get_epoch_images = mocker.patch(
        'vasttools.tools._get_epoch_images',
        return_value=[test_img_path]
    )

    mocker_fits_open = mocker.patch(
        'vasttools.tools.fits.open',
        return_value=fits_open_no_update(test_img_path)
    )

    vtt.add_obs_date('1', '', '')


def test_gen_mocs_field(tmp_path: Path) -> None:
    """
    Tests the generation of a MOC and STMOC for a single fits file

    Args:
        tmp_path: The default pytest temporary path
    Returns:
        None
    """

    test_img_path = str(TEST_DATA_DIR / 'VAST_0012-06A.EPOCH01.I.TEST.fits')
    vtt.gen_mocs_field(test_img_path, outdir=tmp_path)


def test_gen_mocs_epoch(mocker, tmp_path: Path) -> None:
    """
    Tests the generation of all MOCs and STMOCs for a single epoch.
    Also tests the update of the full STMOC.

    Args:
        mocker: The pytest mock mocker object.
        tmp_path: The default pytest temporary path

    Returns:
        None
    """

    test_img_path = str(TEST_DATA_DIR / 'VAST_0012-06A.EPOCH01.I.TEST.fits')

    mocker_get_epoch_images = mocker.patch(
        'vasttools.tools._get_epoch_images',
        return_value=[test_img_path]
    )

    vtt.gen_mocs_epoch('1', '', '', outdir=tmp_path)


def test_mocs_with_holes(tmp_path: Path) -> None:
    """
    Tests that gen_mocs_field produces the same output regardless of whether
    there are NaN holes within the image.

    Args:
        tmp_path: The default pytest temporary path

    Returns:
        None
    """

    border_width = 50
    image_width = 1000
    hole_width = 100

    centre = int(image_width/2)
    hole_rad = int(hole_width/2)

    test_img_path = str(TEST_DATA_DIR / 'VAST_0012-06A.EPOCH01.I.TEST.fits')

    hdu = fits.open(test_img_path)[0]
    header = hdu.header

    data = np.ones((image_width, image_width))
    data[:border_width, :] = np.nan
    data[-border_width:, :] = np.nan
    data[:, :border_width] = np.nan
    data[:, -border_width:] = np.nan

    hole_data = data.copy()
    hole_data[centre-hole_rad:centre+hole_rad,
              centre-hole_rad:centre+hole_rad
              ] = np.nan

    full_path = tmp_path / 'TESTFIELD.EPOCH01.I.fits'
    hole_path = tmp_path / 'TESTFIELDHOLE.EPOCH01.I.fits'

    fits.writeto(full_path,
                 data,
                 header
                 )
    fits.writeto(hole_path,
                 hole_data,
                 header
                 )

    full_moc, full_stmoc = vtt.gen_mocs_field(full_path, outdir=tmp_path)
    hole_moc, hole_stmoc = vtt.gen_mocs_field(hole_path, outdir=tmp_path)

    assert full_moc == hole_moc
    assert full_stmoc == hole_stmoc
