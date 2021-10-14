"""Functions and classes related to VAST that have no specific category
and can be used generically.
"""
import os
import glob

import healpy as hp
import numpy as np
import pandas as pd
import scipy.ndimage as ndi

from pathlib import Path
from mocpy import MOC
from mocpy import STMOC
from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time

from astropy import units as u
from astropy.coordinates import SkyCoord, Angle

from vasttools.survey import load_fields_file
from vasttools.moc import VASTMOCS


def skymap2moc(filename: str, cutoff: float) -> MOC:
    """
    Creates a MOC of the specified credible region of a given skymap.

    Args:
        filename: Path to the healpix skymap file.
        cutoff: Credible level cutoff.

    Returns:
        A MOC containing the credible region.

    Raises:
        ValueError: Credible level cutoff must be between 0 and 1
        FileNotFoundError: File does not exist
    """
    skymap = Path(filename)

    if not 0.0 <= cutoff <= 1.0:
        raise Exception("Credible level cutoff must be between 0 and 1")

    if not skymap.is_file():
        raise Exception("{} does not exist".format(skymap))

    hpx = hp.read_map(filename, nest=True)
    nside = hp.get_nside(hpx)
    level = np.log2(nside)

    i = np.flipud(np.argsort(hpx))
    sorted_credible_levels = np.cumsum(hpx[i])
    credible_levels = np.empty_like(sorted_credible_levels)
    credible_levels[i] = sorted_credible_levels

    idx = np.where(credible_levels < cutoff)[0]
    levels = np.ones(len(idx)) * level

    moc = MOC.from_healpix_cells(idx, depth=levels)

    return moc


def find_in_moc(
    moc: MOC,
    df: pd.DataFrame,
    pipe: bool = True
) -> np.ndarray:
    """
    Find the sources that are contained within a MOC

    Args:
        moc: The MOC of interest.
        df: Dataframe of sources.
        pipe: Whether the dataframe is from the pipeline. Defaults to True.

    Returns:
        Indices of all sources contained within the MOC.
    """

    if pipe:
        ra_col = 'wavg_ra'
        dec_col = 'wavg_dec'
    else:
        ra_col = 'ra'
        dec_col = 'dec'

    ra = Angle(df[ra_col], unit='deg')
    dec = Angle(df[dec_col], unit='deg')

    return np.where(moc.contains(ra, dec))[0]


def add_credible_levels(
    filename: str,
    df: pd.DataFrame,
    pipe: bool = True
) -> None:
    """
    Calculate the minimum credible region containing each source
    and add to the dataframe in-place.

    Args:
        filename: Path to the healpix skymap file.
        df: Dataframe of sources.
        pipe: Whether the dataframe is from the pipeline. Defaults to True.

    Returns:
        None

    Raises:
        FileNotFoundError: File does not exist
    """

    skymap = Path(filename)
    if not skymap.is_file():
        raise Exception("{} does not exist".format(filename))

    if pipe:
        ra_col = 'wavg_ra'
        dec_col = 'wavg_dec'
    else:
        ra_col = 'ra'
        dec_col = 'dec'

    hpx = hp.read_map(filename)
    nside = hp.get_nside(hpx)

    i = np.flipud(np.argsort(hpx))
    sorted_credible_levels = np.cumsum(hpx[i])
    credible_levels = np.empty_like(sorted_credible_levels)
    credible_levels[i] = sorted_credible_levels
    theta = 0.5 * np.pi - np.deg2rad(df[dec_col].values)
    phi = np.deg2rad(df[ra_col].values)

    ipix = hp.ang2pix(nside, theta, phi)

    df.loc[:, 'credible_level'] = credible_levels[ipix]
