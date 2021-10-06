import astropy.units as u
import numpy as np
import pandas as pd
import pytest

from astropy.coordinates import SkyCoord, Angle
from astropy.table import Table
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
