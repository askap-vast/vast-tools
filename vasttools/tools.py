"""Functions and classes related to VAST that have no specific category
and can be used generically.
"""
import os
import glob
import logging

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
from typing import Union

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


# New epoch tools
def _create_beam_df(beam_files: list) -> pd.DataFrame:
    """
    Create the dataframe of all beam information from a list of beam files

    Args:
        beam_files: the list of beam files

    Returns:
        A dataframe of complete beam information
    """

    beam_columns = ['BEAM_NUM',
                    'RA_DEG',
                    'DEC_DEG',
                    'PSF_MAJOR',
                    'PSF_MINOR',
                    'PSF_ANGLE'
                    ]

    for i, beam_file in enumerate(beam_files):
        field = "VAST_" + \
            beam_file.name.split('VAST_')[-1].split(beam_file.suffix)[0]
        sbid = int(beam_file.name.split('beam_inf_')[-1].split('-')[0])

        temp = pd.read_csv(beam_file)
        temp = temp.loc[:, beam_columns]
        temp['SBID'] = sbid
        temp['FIELD_NAME'] = field

        if i == 0:
            beam_df = temp.copy()
        else:
            beam_df = beam_df.append(temp)

    return beam_df


def _set_epoch_path(epoch: str) -> Path:
    """
    Set the epoch_path from the VAST_DATA_DIR variable

    Args:
        epoch: The epoch of interest

    Returns:
        Path to the epoch of interest

    Raises:
        OSError: Requested path could not be determined
    """

    base_folder = os.getenv('VAST_DATA_DIR')
    if base_folder is None:
        raise Exception(
            "The path to the requested epoch could not be determined!"
            " Either the system environment 'VAST_DATA_DIR' must be"
            " defined or the 'epoch_path' provided."
        )

    base_folder = Path(base_folder)

    epoch_path = base_folder / 'EPOCH{}'.format(epoch)

    return epoch_path


def create_fields_csv(epoch_num: str,
                      db_path: str,
                      outdir: Union[str, Path] = '.'
                      ) -> None:
    """
    Create the fields csv for a single epoch using the askap_surveys database.

    Args:
        epoch_num: Epoch number of interest.
        db_path: Path to the askap_surveys database.
        outdir: Path to the output directory.
            Defaults to the current directory.

    Returns:
        None
    """

    outdir = Path(outdir)

    field_columns = ['FIELD_NAME', 'SBID', 'SCAN_START', 'SCAN_LEN']

    vast_db = Path(db_path)
    if type(epoch_num) is int:
        epoch_num = str(epoch_num)
    epoch = vast_db / 'epoch_{}'.format(epoch_num.replace('x', ''))

    beam_files = list(epoch.glob('beam_inf_*.csv'))
    beam_df = _create_beam_df(beam_files)

    field_data = epoch / 'field_data.csv'

    field_df = pd.read_csv(field_data)
    field_df = field_df.loc[:, field_columns]

    epoch_csv = beam_df.merge(field_df,
                              left_on=['SBID', 'FIELD_NAME'],
                              right_on=['SBID', 'FIELD_NAME']
                              )

    # convert the coordinates to match format in tools v2.0.0
    coordinates = SkyCoord(
        ra=epoch_csv['RA_DEG'].to_numpy(),
        dec=epoch_csv['DEC_DEG'].to_numpy(),
        unit=(u.deg, u.deg)
    )

    epoch_csv['RA_HMS'] = coordinates.ra.to_string(u.hour,
                                                   sep=":",
                                                   precision=3
                                                   )
    epoch_csv['DEC_DMS'] = coordinates.dec.to_string(sep=":",
                                                     precision=3,
                                                     alwayssign=True
                                                     )

    start_times = epoch_csv['SCAN_START'].to_numpy() / 86400.
    end_times = start_times + epoch_csv['SCAN_LEN'].to_numpy() / 86400.
    start_times = Time(start_times, format='mjd')
    end_times = Time(end_times, format='mjd')

    epoch_csv['DATEOBS'] = start_times.iso
    epoch_csv['DATEEND'] = end_times.iso
    epoch_csv['NINT'] = np.around(epoch_csv['SCAN_LEN'] / 10.).astype(np.int64)

    drop_cols = ['SCAN_START', 'SCAN_LEN', 'RA_DEG', 'DEC_DEG']
    epoch_csv = epoch_csv.drop(drop_cols, axis=1)
    epoch_csv = epoch_csv.rename(columns={'BEAM_NUM': 'BEAM',
                                          'PSF_MAJOR': 'BMAJ',
                                          'PSF_MINOR': 'BMIN',
                                          'PSF_ANGLE': 'BPA'})
    epoch_csv = epoch_csv.loc[:, [
        'SBID',
        'FIELD_NAME',
        'BEAM',
        'RA_HMS',
        'DEC_DMS',
        'DATEOBS',
        'DATEEND',
        'NINT',
        'BMAJ',
        'BMIN',
        'BPA'
    ]]
    outfile = 'vast_epoch{}_info.csv'.format(epoch_num)
    epoch_csv.to_csv(outdir / outfile, index=False)


def add_obs_date(epoch: str,
                 image_type: str,
                 image_dir: str,
                 epoch_path: str = None
                 ) -> None:
    """
    Add datetime information to all fits files in a single epoch.

    Args:
        epoch: The epoch of interest
        image_type: `COMBINED` or `TILES`
        image_dir: The name of the folder containing the images to be updated.
            E.g. `STOKESI_IMAGES`.
        epoch_path: Full path to the folder containing the epoch.
            Defaults to None, which will set the value based on the
            `VAST_DATA_DIR` environment variable and `epoch`.

    Returns:
        None
    """

    epoch_info = load_fields_file(epoch)

    if epoch_path is None:
        epoch_path = _set_epoch_path(epoch)

    raw_images = _get_epoch_images(epoch_path, image_type, image_dir)

    for filename in raw_images:
        field = filename.split("/")[-1].split(".")[0]
        field_info = epoch_info[epoch_info.FIELD_NAME == field].iloc[0]
        field_start = Time(field_info.DATEOBS)
        field_end = Time(field_info.DATEEND)
        duration = field_end - field_start

        hdu = fits.open(filename, mode="update")
        hdu[0].header["DATE-OBS"] = field_start.fits
        hdu[0].header["MJD-OBS"] = field_start.mjd
        hdu[0].header["DATE-BEG"] = field_start.fits
        hdu[0].header["DATE-END"] = field_end.fits
        hdu[0].header["MJD-BEG"] = field_start.mjd
        hdu[0].header["MJD-END"] = field_end.mjd
        hdu[0].header["TELAPSE"] = duration.sec
        hdu[0].header["TIMEUNIT"] = "s"
        hdu.close()


def gen_mocs_field(fits_file: str,
                   outdir: Union[str, Path] = '.'
                   ) -> Union[MOC, STMOC]:
    """
    Generate a MOC and STMOC for a single fits file.

    Args:
        fits_file: path to the fits file.
        outdir: Path to the output directory.
            Defaults to the current directory.

    Returns:
        The MOC and STMOC.

    Raises:
        FileNotFoundError: File does not exist
    """

    outdir = Path(outdir)

    if not os.path.isfile(fits_file):
        raise Exception("{} does not exist".format(fits_file))

    with fits.open(fits_file) as vast_fits:
        vast_data = vast_fits[0].data
        if vast_data.ndim == 4:
            vast_data = vast_data[0, 0, :, :]
        vast_header = vast_fits[0].header
        vast_wcs = WCS(vast_header, naxis=2)

    binary = (~np.isnan(vast_data)).astype(int)
    binary = np.pad(binary, 1, mode='constant')
    dist = ndi.distance_transform_cdt(binary, metric='taxicab')
    mask = dist[1:-1, 1:-1]

    x, y = np.where(mask == 1)
    # need to know when to reverse by checking axis sizes.
    pixels = np.column_stack((y, x))

    coords = SkyCoord(vast_wcs.wcs_pix2world(
        pixels, 0), unit="deg", frame="icrs")

    moc = MOC.from_polygon_skycoord(coords, max_depth=10)
    start = Time([vast_header['DATE-BEG']])
    end = Time([vast_header['DATE-END']])
    stmoc = STMOC.from_spatial_coverages(
        start, end, [moc]
    )

    filename = os.path.split(fits_file)[1]
    moc_name = filename.replace(".fits", ".moc.fits")
    stmoc_name = filename.replace(".fits", ".stmoc.fits")

    moc.write(outdir / moc_name, overwrite=True)
    stmoc.write(outdir / stmoc_name, overwrite=True)

    return moc, stmoc


def gen_mocs_epoch(epoch: str,
                   image_type: str,
                   image_dir: str,
                   epoch_path: str = None,
                   outdir: Union[str, Path] = '.'
                   ) -> None:
    """
    Generate MOCs and STMOCs for all images in a single epoch, and create a new
    full pilot STMOC.

    Args:
        epoch: The epoch of interest.
        image_type: `COMBINED` or `TILES`
        image_dir: The name of the folder containing the images to be updated.
            E.g. `STOKESI_IMAGES`.
        epoch_path: Full path to the folder containing the epoch.
            Defaults to None, which will set the value based on the
            `VAST_DATA_DIR` environment variable and `epoch`.
        outdir: Path to the output directory.
            Defaults to the current directory.

    Returns:
        None
    """

    outdir = Path(outdir)

    vtm = VASTMOCS()
    full_STMOC = vtm.load_pilot_stmoc()

    if epoch_path is None:
        epoch_path = _set_epoch_path(epoch)

    raw_images = _get_epoch_images(epoch_path, image_type, image_dir)

    for i, f in enumerate(raw_images):
        themoc, thestmoc = gen_mocs_field(f)

        if i == 0:
            mastermoc = themoc
            masterstemoc = thestmoc
        else:
            mastermoc = mastermoc.union(themoc)
            masterstemoc = masterstemoc.union(thestmoc)

    master_name = "VAST_PILOT_{}_moc.fits".format(epoch)
    master_stmoc_name = master_name.replace("moc", "stmoc")

    mastermoc.write(outdir / master_name, overwrite=True)
    masterstemoc.write(outdir / master_stmoc_name, overwrite=True)

    full_STMOC = full_STMOC.union(masterstemoc)
    full_STMOC.write(outdir / 'VAST_PILOT.stmoc.fits', overwrite=True)


def _get_epoch_images(epoch_path: Union[str, Path],
                      image_type: str,
                      image_dir: str
                      ) -> list:
    """
    Get all available images in a given epoch.

    Args:
        epoch_path: Path to the epoch of interest.
        image_type: `COMBINED` or `TILES`
        image_dir: The name of the folder containing the images to be updated.
            E.g. `STOKESI_IMAGES`.

    Returns:
        The list of images.

    Raises:
        OSError: Directory path does not exist
    """

    P = Path(epoch_path) / image_type / image_dir
    if not P.exists():
        raise Exception("{} does not exist!".format(P))
    raw_images = sorted(list(P.glob("*.fits")))

    return raw_images
