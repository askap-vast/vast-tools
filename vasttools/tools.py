"""Functions and classes related to VAST that have no specific category
and can be used generically.
"""
from copy import deepcopy
from dataclasses import dataclass
import enum
import importlib.resources
import os
import pickle
from xml.dom import minidom

import healpy as hp
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from svgpath2mpl import parse_path

from pathlib import Path
from mocpy import MOC
from mocpy import STMOC
from astropy.io import fits
from astropy.time import Time
from typing import Union, Dict, Any, Tuple, Optional

from astropy import units as u
from astropy.coordinates import SkyCoord, Angle

from vasttools.survey import load_fields_file
from vasttools.moc import VASTMOCS
from vasttools.utils import create_moc_from_fits
from vasttools import FREQ_CONVERSION


def skymap2moc(filename: str, cutoff: float) -> MOC:
    """
    Creates a MOC of the specified credible region of a given skymap.

    Args:
        filename: Path to the healpix skymap file.
        cutoff: Credible level cutoff.

    Returns:
        A MOC containing the credible region.

    Raises:
        ValueError: Credible level cutoff must be between 0 and 1.
        Exception: Path does not exist.
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

    moc = MOC.from_healpix_cells(idx, depth=levels, max_depth=9)

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
        Exception: Path does not exist.
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
    Create the dataframe of all beam information from a list of beam files.

    Args:
        beam_files: the list of beam files.

    Returns:
        A dataframe of complete beam information.
    """

    beam_columns = ['BEAM_NUM',
                    'RA_DEG',
                    'DEC_DEG',
                    'PSF_MAJOR',
                    'PSF_MINOR',
                    'PSF_ANGLE'
                    ]

    for i, beam_file in enumerate(beam_files):
        survey_str = "VAST_"
        if "RACS" in beam_file.name:
            survey_str = "RACS_"
        field = survey_str + \
            beam_file.name.split(survey_str)[-1].split(beam_file.suffix)[0]
        sbid = int(beam_file.name.split('beam_inf_')[-1].split('-')[0])

        temp = pd.read_csv(beam_file)
        temp = temp.loc[:, beam_columns]
        temp['SBID'] = sbid
        temp['FIELD_NAME'] = field

        if i == 0:
            beam_df = temp.copy()
        else:
            beam_df = pd.concat([beam_df, temp])

    return beam_df


def _set_epoch_path(epoch: str) -> Path:
    """
    Set the epoch_path from the VAST_DATA_DIR variable.

    Args:
        epoch: The epoch of interest.

    Returns:
        Path to the epoch of interest.

    Raises:
        Exception: Requested path could not be determined.
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


def _create_fields_df(epoch_num: str,
                      db_path: str,
                      ) -> pd.DataFrame:
    """
    Create the the fields DataFrame for a single epoch using the
    askap_surveys database.

    Args:
        epoch_num: Epoch number of interest.
        db_path: Path to the askap_surveys database.

    Returns:
        The fields DataFrame.

    Raises:
        Exception: Path does not exist.
        Exception: No data available for epoch.
    """
    field_columns = ['FIELD_NAME', 'SBID', 'SCAN_START', 'SCAN_LEN']

    vast_db = Path(db_path)
    if not vast_db.exists():
        raise Exception("{} does not exist!".format(vast_db))

    if isinstance(epoch_num, str):
        epoch_num = int(epoch_num.replace('x', ''))

    descrip_df = pd.read_csv(vast_db / 'description.csv', index_col='EPOCH')

    if epoch_num not in descrip_df.index:
        raise Exception("No data available for epoch {}".format(epoch_num))

    sky_freq = descrip_df.OBS_FREQ.loc[epoch_num]

    if sky_freq not in FREQ_CONVERSION.keys():
        raise Exception("Sky frequency is not in a recognised VAST band")
    else:
        obs_freq = FREQ_CONVERSION[sky_freq]

    epoch = vast_db / 'epoch_{}'.format(epoch_num)

    beam_files = list(epoch.glob('beam_inf_*.csv'))
    beam_df = _create_beam_df(beam_files)

    field_data = epoch / 'field_data.csv'

    if not field_data.exists():
        raise Exception("{} does not exist!".format(field_data))

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

    epoch_csv['OBS_FREQ'] = [obs_freq] * len(epoch_csv)
    epoch_csv = epoch_csv.loc[:, [
        'SBID',
        'OBS_FREQ',
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

    return epoch_csv


def _create_fields_sc(fields_df: pd.DataFrame) -> SkyCoord:
    """
    Create the fields direction Skycoord objects from the fields_df dataframe.

    Args:
        fields_df: Fields dataframe.

    Returns:
        Skycoord containing the beam centres
    """

    fields_sc = SkyCoord(
        Angle(fields_df["RA_HMS"], unit=u.hourangle),
        Angle(fields_df["DEC_DMS"], unit=u.deg)
    )
    return fields_sc


def create_fields_metadata(epoch_num: str,
                           db_path: str,
                           outdir: Union[str, Path] = '.'
                           ) -> None:
    """
    Create and write the fields csv and skycoord pickle for a single epoch.

    Args:
        epoch_num: Epoch number of interest.
        db_path: Path to the askap_surveys database.
        outdir: Path to the output directory.
            Defaults to the current directory.

    Returns:
        None

    Raises:
        Exception: Path does not exist.
    """

    if isinstance(outdir, str):
        outdir = Path(outdir)

    if not outdir.exists():
        raise Exception("{} does not exist!".format(outdir))

    fields_df = _create_fields_df(epoch_num, db_path)
    fields_sc = _create_fields_sc(fields_df)

    if len(epoch_num.rstrip('x')) == 1:
        epoch_num = f'0{epoch_num}'
    fields_outfile = f'vast_epoch{epoch_num}_info.csv'
    sc_outfile = f'vast_epoch{epoch_num}_fields_sc.pickle'

    fields_df.to_csv(outdir / fields_outfile, index=False)

    with open(outdir / sc_outfile, 'wb') as picklefile:
        pickle.dump(fields_sc, picklefile)


def add_obs_date(epoch: str,
                 image_type: str,
                 image_dir: str,
                 epoch_path: str = None
                 ) -> None:
    """
    Add datetime information to all fits files in a single epoch.

    Args:
        epoch: The epoch of interest.
        image_type: `COMBINED` or `TILES`.
        image_dir: The name of the folder containing the images to be updated.
            E.g. `STOKESI_IMAGES`.
        epoch_path: Full path to the folder containing the epoch.
            Defaults to None, which will set the value based on the
            `VAST_DATA_DIR` environment variable and `epoch`.

    Returns:
        None

    Raises:
        ValueError: When image_type is not 'TILES' or 'COMBINED'.
    """
    epoch_info = load_fields_file(epoch)

    if epoch_path is None:
        epoch_path = _set_epoch_path(epoch)

    raw_images = _get_epoch_images(epoch_path, image_type, image_dir)

    for filename in raw_images:
        split_name = filename.split("/")[-1].split(".")
        if image_type == 'TILES':
            field = split_name[4]
        elif image_type == 'COMBINED':
            field = split_name[0]
        else:
            raise ValueError(
                "Image type not recognised, "
                "must be either 'TILES' or 'COMBINED'."
            )

        field_info = epoch_info[epoch_info.FIELD_NAME == field].iloc[0]
        field_start = Time(field_info.DATEOBS)
        field_end = Time(field_info.DATEEND)
        duration = field_end - field_start

        hdu = fits.open(filename, mode="update")
        hdu_index = 0
        if filename.endswith('.fits.fz'):
            hdu_index = 1
        hdu[hdu_index].header["DATE-OBS"] = field_start.fits
        hdu[hdu_index].header["MJD-OBS"] = field_start.mjd
        hdu[hdu_index].header["DATE-BEG"] = field_start.fits
        hdu[hdu_index].header["DATE-END"] = field_end.fits
        hdu[hdu_index].header["MJD-BEG"] = field_start.mjd
        hdu[hdu_index].header["MJD-END"] = field_end.mjd
        hdu[hdu_index].header["TELAPSE"] = duration.sec
        hdu[hdu_index].header["TIMEUNIT"] = "s"
        hdu.close()


def gen_mocs_image(fits_file: str,
                   outdir: Union[str, Path] = '.',
                   write: bool = False
                   ) -> Union[MOC, STMOC]:
    """
    Generate a MOC and STMOC for a single fits file.

    Args:
        fits_file: path to the fits file.
        outdir: Path to the output directory.
            Defaults to the current directory.
        write: Write the MOC/STMOC to file.

    Returns:
        The MOC and STMOC.

    Raises:
        Exception: Path does not exist.
    """

    if isinstance(outdir, str):
        outdir = Path(outdir)

    if not outdir.exists():
        raise Exception("{} does not exist".format(outdir))

    if not Path(fits_file).exists():
        raise Exception("{} does not exist".format(fits_file))

    moc = create_moc_from_fits(fits_file)

    header = fits.getheader(fits_file, 0)
    start = Time([header['DATE-BEG']])
    end = Time([header['DATE-END']])
    stmoc = STMOC.from_spatial_coverages(
        start, end, [moc]
    )

    if write:
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
                   outdir: Union[str, Path] = '.',
                   base_stmoc: Union[str, Path] = None
                   ) -> None:
    """
    Generate MOCs and STMOCs for all images in a single epoch, and create a new
    full pilot STMOC.

    Args:
        epoch: The epoch of interest.
        image_type: `COMBINED` or `TILES`.
        image_dir: The name of the folder containing the images to be updated.
            E.g. `STOKESI_IMAGES`.
        epoch_path: Full path to the folder containing the epoch.
            Defaults to None, which will set the value based on the
            `VAST_DATA_DIR` environment variable and `epoch`.
        outdir: Path to the output directory.
            Defaults to the current directory.
        base_stmoc: Path to the STMOC to use as the base. Defaults to `None`,
            in which case the VAST STMOC installed with vast-tools will
            be used.

    Returns:
        None

    Raises:
        Exception: Path does not exist.
    """

    if isinstance(outdir, str):
        outdir = Path(outdir)

    if not outdir.exists():
        raise Exception("{} does not exist".format(outdir))

    if base_stmoc is None:
        vtm = VASTMOCS()
        full_STMOC = vtm.load_pilot_stmoc()
    else:
        if isinstance(base_stmoc, str):
            base_stmoc = Path(base_stmoc)

        if not base_stmoc.exists():
            raise Exception("{} does not exist".format(base_stmoc))

        full_STMOC = STMOC.from_fits(base_stmoc)

    if epoch_path is None:
        epoch_path = _set_epoch_path(epoch)

    raw_images = _get_epoch_images(epoch_path, image_type, image_dir)

    for i, f in enumerate(raw_images):
        themoc, thestmoc = gen_mocs_image(f)

        if i == 0:
            mastermoc = themoc
            masterstemoc = thestmoc
        else:
            mastermoc = mastermoc.union(themoc)
            masterstemoc = masterstemoc.union(thestmoc)

    master_name = "VAST_PILOT_EPOCH{}.moc.fits".format(epoch)
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
        image_type: `COMBINED` or `TILES`.
        image_dir: The name of the folder containing the images to be updated.
            E.g. `STOKESI_IMAGES`.

    Returns:
        The list of images.

    Raises:
        Exception: Path does not exist.
    """

    P = Path(epoch_path) / image_type / image_dir
    if not P.exists():
        raise Exception("{} does not exist!".format(P))
    raw_images = sorted(list(P.glob("*.fits")))

    return raw_images


def offset_postagestamp_axes(ax: plt.Axes,
                             centre_sc: SkyCoord,
                             ra_units: u.core.Unit = u.arcsec,
                             dec_units: u.core.Unit = u.arcsec,
                             ra_label: str = 'R.A. Offset',
                             dec_label: str = 'Dec. Offset',
                             major_tick_length: Union[int, float] = 6,
                             minor_tick_length: Union[int, float] = 3,
                             ) -> None:
    """
    Display axis ticks and labels as offsets from a given coordinate,
    rather than in absolute coordinates.

    Args:
        ax: The axis of interest
        centre_sc: SkyCoord to calculate offsets from
        ra_units: Right Ascension axis ticklabel units
        dec_units: Declination axis ticklabel units
        ra_label: Right Ascension axis label
        dec_label: Declination axis label
        major_tick_length: Major tick length in points
        minor_tick_length: Minor tick length in points

    Returns:
        None

    Raises:
        Exception: R.A. and Dec. units must be angles.
    """

    if ra_units.physical_type != 'angle' or dec_units.physical_type != 'angle':
        raise Exception("R.A. and Dec. units must be angles.")

    ra_offs, dec_offs = ax.get_coords_overlay(centre_sc.skyoffset_frame())
    plt.minorticks_on()
    ra_offs.set_coord_type('longitude', 180)
    ra_offs.set_format_unit(ra_units, decimal=True)
    dec_offs.set_format_unit(dec_units, decimal=True)
    ra_offs.tick_params(direction='in', color='black')
    ra_offs.tick_params(which='major', length=major_tick_length)
    ra_offs.tick_params(which='minor', length=minor_tick_length)

    dec_offs.tick_params(direction='in', color='black')
    dec_offs.tick_params(which='major', length=major_tick_length)
    dec_offs.tick_params(which='minor', length=minor_tick_length)

    ra, dec = ax.coords

    ra.set_ticks_visible(False)
    ra.set_ticklabel_visible(False)
    dec.set_ticks_visible(False)
    dec.set_ticklabel_visible(False)
    ra_offs.display_minor_ticks(True)
    dec_offs.display_minor_ticks(True)

    dec_offs.set_ticks_position('lr')
    dec_offs.set_ticklabel_position('l')
    dec_offs.set_axislabel_position('l')
    dec_offs.set_axislabel(dec_label)

    ra_offs.set_ticks_position('tb')
    ra_offs.set_ticklabel_position('b')
    ra_offs.set_axislabel_position('b')
    ra_offs.set_axislabel(ra_label)

    return


class WiseClass(enum.Enum):
    """WISE object classes defined in the WISE color-color plot."""

    COOL_T_DWARFS = "CoolTDwarfs"
    STARS = "Stars"
    ELLIPTICALS = "Ellipticals"
    SPIRALS = "Spirals"
    LIRGS = "LIRGs"
    STARBURSTS = "Starbursts"
    SEYFERTS = "Seyferts"
    QSOS = "QSOs"
    OBSCURED_AGN = "ObscuredAGN"


@dataclass
class WisePatchConfig:
    """Style and annotation configurations for the patch drawn to represent a
    WISE object class in the WISE color-color plot.

    Attributes:
        style (Dict[str, Any]): Any style keyword arguments and values
            supported by `matplotlib.patches.PathPatch`.
        annotation_text (str): Text to annotate the patch.
        annotation_position (Tuple[float, float]): Position in data coordinates
        for the annotation text.
    """

    style: Dict[str, Any]
    annotation_text: str
    annotation_position: Tuple[float, float]

    def copy(self):
        return deepcopy(self)


WISE_DEFAULT_PATCH_CONFIGS = {
    WiseClass.COOL_T_DWARFS: WisePatchConfig(
        style=dict(fc="#cb4627", ec="none"),
        annotation_text="Cool\nT-Dwarfs",
        annotation_position=(1.15, 3.0),
    ),
    WiseClass.STARS: WisePatchConfig(
        style=dict(fc="#e8e615", ec="none"),
        annotation_text="Stars",
        annotation_position=(0.5, 0.4),
    ),
    WiseClass.ELLIPTICALS: WisePatchConfig(
        style=dict(fc="#95c53d", ec="none"),
        annotation_text="Ellipticals",
        annotation_position=(1.0, -0.25),
    ),
    WiseClass.SPIRALS: WisePatchConfig(
        style=dict(fc="#bbdeb5", ec="none", alpha=0.7),
        annotation_text="Spirals",
        annotation_position=(2.5, 0.35),
    ),
    WiseClass.LIRGS: WisePatchConfig(
        style=dict(fc="#ecc384", ec="none"),
        annotation_text="LIRGs",
        annotation_position=(5.0, -0.1),
    ),
    WiseClass.STARBURSTS: WisePatchConfig(
        style=dict(fc="#e8e615", ec="none", alpha=0.7),
        annotation_text="ULIRGs\nLINERs\nStarbursts",
        annotation_position=(4.75, 0.5),
    ),
    WiseClass.SEYFERTS: WisePatchConfig(
        style=dict(fc="#45c7f0", ec="none", alpha=0.7),
        annotation_text="Seyferts",
        annotation_position=(3.5, 0.9),
    ),
    WiseClass.QSOS: WisePatchConfig(
        style=dict(fc="#b4e2ec", ec="none"),
        annotation_text="QSOs",
        annotation_position=(3.1, 1.25),
    ),
    WiseClass.OBSCURED_AGN: WisePatchConfig(
        style=dict(fc="#faa719", ec="none", alpha=0.8),
        annotation_text="ULIRGs/LINERs\nObscured AGN",
        annotation_position=(4.5, 1.75),
    ),
}


def wise_color_color_plot(
    patch_style_overrides: Optional[Dict[WiseClass, WisePatchConfig]] = None,
    annotation_text_size: Union[float, str] = "x-small",
) -> matplotlib.figure.Figure:
    """Make an empty WISE color-color plot with common object classes drawn as
    patches. The patches have default styles that may be overridden. To
    override a patch style, pass in a dict containing the desired style and
    annotation settings. The overrides must be complete, i.e. a complete
    `WisePatchConfig` object must be provided for each `WiseClass` you wish to
    modify. Partial updates to the style or annotation of individual patches is
    not supported.

    For example, to change the color of the stars patch to blue:
    ```python
    fig = wise_color_color_plot({
        WiseClass.STARS: WisePatchConfig(
            style=dict(fc="blue", ec="none"),
            annotation_text="Stars",
            annotation_position=(0.5, 0.4),
        )
    })
    ```

    Args:
        patch_style_overrides (Optional[Dict[WiseClass, WisePatchConfig]],
            optional): Override the default patch styles for the given WISE
            object class. If None, use defaults for each patch. Defaults to
            None.
        annotation_text_size (Union[float, str]): Font size for the patch
            annotations. Accepts a font size (float) or a matplotlib font scale
            string (e.g. "xx-small", "medium", "xx-large"). Defaults to
            "x-small".
    Returns:
        `matplotlib.figure.Figure`: the WISE color-color figure. Access the
            axes with the `.axes` attribute.
    """
    # set the WISE object classification patch styles
    if patch_style_overrides is not None:
        patch_styles = WISE_DEFAULT_PATCH_CONFIGS.copy()
        patch_styles.update(patch_style_overrides)
    else:
        patch_styles = WISE_DEFAULT_PATCH_CONFIGS

    # parse the WISE color-color SVG that contains the object class patches
    with importlib.resources.path(
        "vasttools.data", "WISE-color-color.svg"
    ) as svg_path:
        doc = minidom.parse(str(svg_path))
    # define the transform from the SVG frame to the plotting frame
    transform = (
        matplotlib.transforms.Affine2D().scale(sx=1, sy=-1).translate(-1, 4)
    )

    fig, ax = plt.subplots()
    # add WISE object classification patches
    for svg_path in doc.getElementsByTagName("path"):
        name = svg_path.getAttribute("id")
        patch_style = patch_styles[getattr(WiseClass, name)]
        path_mpl = parse_path(svg_path.getAttribute("d")).transformed(
            transform
        )
        patch = matplotlib.patches.PathPatch(path_mpl, **patch_style.style)
        ax.add_patch(patch)
        ax.annotate(
            patch_style.annotation_text,
            patch_style.annotation_position,
            ha="center",
            fontsize=annotation_text_size,
        )
    ax.set_xlim(-1, 6)
    ax.set_ylim(-0.5, 4)
    ax.set_aspect(1)
    ax.set_xlabel("[4.6] - [12] (mag)")
    ax.set_ylabel("[3.4] - [4.6] (mag)")
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    return fig
