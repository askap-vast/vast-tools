import os
import healpy as hp
import numpy as np
import pandas as pd

from pathlib import Path
from mocpy import MOC
from astropy import units as u
from astropy.coordinates import SkyCoord, Angle
from astropy.time import Time
from pathlib import Path
import glob

from vasttools.survey import load_fields_file


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


def create_fields_csv(epoch_num: str, db_path: str) -> None:
    """
    Create the fields csv for a single epoch using the askap_surveys database

    Args:
        epoch_num: Epoch number of interest
        db_path: Path to the askap_surveys database
    """

    field_columns = ['FIELD_NAME', 'SBID', 'SCAN_START', 'SCAN_LEN']
    beam_columns = ['BEAM_NUM',
                    'RA_DEG',
                    'DEC_DEG',
                    'PSF_MAJOR',
                    'PSF_MINOR',
                    'PSF_ANGLE'
                    ]

    vast_db = Path(db_path)
    if type(epoch_num) is int:
        epoch_num = str(epoch_num)
    epoch = vast_db / 'epoch_{}'.format(epoch_num.replace('x', ''))

    beam_files = list(epoch.glob('beam_inf_*.csv'))
    field_data = epoch / 'field_data.csv'

    field_df = pd.read_csv(field_data)
    field_df = field_df.loc[:, field_columns]

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

    epoch_csv.to_csv('vast_epoch{}_info.csv'.format(epoch_num), index=False)
    
    

        
def add_obs_date(epoch: str, image_dir: str, epoch_path: str = None):
    """
    
    """
    
    if epoch_path is None:
        base_folder = Path(os.getenv('VAST_DATA_DIR'))
        epoch_path = base_folder / 'EPOCH{}'.format(epoch)
    
    epoch_info = load_fields_file(epoch)

    glob_str = os.path.join(epoch_path,image_dir, "*.fits")
    raw_images = sorted(glob.glob(glob_str))

    for filename in raw_images:
        field = filename.split("/")[-1].split(".")[4]
        field_info = epoch_info[epoch_info.FIELD_NAME == field].iloc[0]
        field_start = Time(field_info.DATEOBS)
        field_end = Time(field_info.DATEEND)
        duration = field_end - field_start
        
        hdu = fits.open(i, mode="update")
        hdu[0].header["DATE-OBS"] = field_start.fits
        hdu[0].header["MJD-OBS"] = field_start.mjd
        hdu[0].header["DATE-BEG"] = field_start.fits
        hdu[0].header["DATE-END"] = field_end.fits
        hdu[0].header["MJD-BEG"] = field_start.mjd
        hdu[0].header["MJD-END"] = field_end.mjd
        hdu[0].header["TELAPSE"] = duration.sec
        hdu[0].header["TIMEUNIT"] = "s"
        hdu.close()
    
    
    
    
    
    
    
