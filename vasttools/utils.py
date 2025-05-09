"""Utility functions used throughout the package.

Attributes:
    use_colorlog (bool): Whether the logging should use colorlog or not.
"""
import os
import sys
import logging
import logging.handlers
import logging.config
import matplotlib.markers
import matplotlib.lines
import numpy as np
import pandas as pd
import dask.dataframe as dd
import scipy.ndimage as ndi
import gc

from astropy import units as u
from astropy.coordinates import SkyCoord, Angle
from astroquery.simbad import Simbad
from astropy.time import Time
from astropy.table import Table
from astropy.coordinates import solar_system_ephemeris
from astropy.coordinates import get_body
from astropy.io import fits
from astropy.wcs import WCS
from multiprocessing_logging import install_mp_handler
from typing import Optional, Union, Tuple, List
from pathlib import Path
from mocpy import MOC

# crosshair imports
from matplotlib.transforms import Affine2D
import matplotlib.path as path


try:
    import colorlog
    use_colorlog = True
except ImportError:
    use_colorlog = False


import vasttools.survey as vts


def get_logger(
    debug: bool,
    quiet: bool,
    logfile: str = None
) -> logging.RootLogger:
    """
    Set up the logger.

    Args:
        debug: Set stream level to debug.
        quiet: Suppress all non-essential output.
        logfile: File to output log to.

    Returns:
        Logger object.
    """
    logger = logging.getLogger()
    s = logging.StreamHandler()
    if logfile is not None:
        fh = logging.FileHandler(logfile)
        fh.setLevel(logging.DEBUG)
    logformat = '[%(asctime)s] - %(levelname)s - %(message)s'

    if use_colorlog:
        formatter = colorlog.ColoredFormatter(
            "%(log_color)s[%(asctime)s] - %(levelname)s - %(blue)s%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            reset=True,
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white', },
            secondary_log_colors={},
            style='%'
        )
    else:
        formatter = logging.Formatter(logformat, datefmt="%Y-%m-%d %H:%M:%S")

    s.setFormatter(formatter)

    if debug:
        s.setLevel(logging.DEBUG)
    else:
        if quiet:
            s.setLevel(logging.WARNING)
        else:
            s.setLevel(logging.INFO)

    logger.addHandler(s)

    if logfile is not None:
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    logger.setLevel(logging.DEBUG)
    install_mp_handler(logger=logger)

    return logger


def _set_crosshair(self) -> None:
    """This function adds a true crosshair marker to matplotlib.

    ============================== ===========================================
    marker                         description
    ============================== ===========================================
    `"c"`                          crosshair

    Usage:
        ```python
        import matplotlib.pyplot as plt
        import crosshair
        plt.scatter(0,0, marker='c', s=100)
        plt.show()
        ```

    Notes:
        I tried to stay as close to the style of `matplotlib/lib/markers.py`,
        so it can easily implemented in mpl after further testing.

        How to implement this in matplotlib via a module was inspired by:
        https://stackoverflow.com/a/16655800/5064815

        Be aware that for small sizes the crosshair looks like four dots or
        even a circle.  This is due to the fact that in this case the linewidth
        is larger then the length of the 'hairs' of the crosshair. This is know
        and similar behaviour is seen for other markers at small sizes.

    Author:
        L. A. Boogaard (13/07/2017)

    Returns:
        None
    """
    _crosshair_path = path.Path([(0.0, -0.5),  # center, bottom
                                 (0.0, -0.25),  # center, q_bot
                                 (-0.5, 0.0),  # left, center
                                 (-0.25, 0.0),  # q_left, center
                                 (0.0, 0.25),  # center, q_top
                                 (0.0, 0.5),  # center, top
                                 (0.25, 0.0),  # q_right, center
                                 (0.5, 0.0)],  # right, center
                                [path.Path.MOVETO,
                                 path.Path.LINETO,
                                 path.Path.MOVETO,
                                 path.Path.LINETO,
                                 path.Path.MOVETO,
                                 path.Path.LINETO,
                                 path.Path.MOVETO,
                                 path.Path.LINETO])

    self._transform = Affine2D().scale(1.0)
    self._snap_threshold = 1.0
    self._filled = False
    self._path = _crosshair_path


def crosshair() -> None:
    """
    A wrapper function to set the crosshair marker in
    matplotlib using the function written by L. A. Boogaard.

    See https://stackoverflow.com/a/16655800/5064815.

    Returns:
        None
    """

    matplotlib.markers.MarkerStyle._set_crosshair = _set_crosshair
    matplotlib.markers.MarkerStyle.markers['c'] = 'crosshair'
    matplotlib.lines.Line2D.markers = matplotlib.markers.MarkerStyle.markers


def check_file(path: str) -> bool:
    """
    Check if logging file exists.

    Args:
        path: filepath to check

    Returns:
        Boolean representing the file existence, 'True' if present, otherwise
            'False'.
    """
    logger = logging.getLogger()
    exists = os.path.isfile(path)
    if not exists:
        logger.critical(
            "Cannot find file '%s'!", path
        )
    return exists


def build_catalog(coords: str, source_names: str) -> pd.DataFrame:
    """
    Build the catalogue of target sources.

    Args:
        coords: The coordinates (comma-separated) or filename entered.
        source_names: Comma-separated source names.

    Returns:
        Catalogue of target sources.
    """
    logger = logging.getLogger()

    if " " not in coords:
        logger.info("Loading file {}".format(coords))
        # Give explicit check to file existence
        user_file = os.path.abspath(coords)
        if not os.path.isfile(user_file):
            logger.critical("{} not found!".format(user_file))
            logger.critical("Exiting.")
            sys.exit()
        try:
            catalog = pd.read_csv(user_file, comment="#")
            catalog.dropna(how="all", inplace=True)
            logger.debug(catalog)
            catalog.columns = map(str.lower, catalog.columns)
            logger.debug(catalog.columns)
            no_ra_col = "ra" not in catalog.columns
            no_dec_col = "dec" not in catalog.columns
            if no_ra_col or no_dec_col:
                logger.critical(
                    "Cannot find one of 'ra' or 'dec' in input file.")
                logger.critical("Please check column headers!")
                sys.exit()
            if "name" not in catalog.columns:
                catalog["name"] = [
                    "{}_{}".format(
                        i, j) for i, j in zip(
                        catalog['ra'], catalog['dec'])]
            else:
                catalog['name'] = catalog['name'].astype(str)
        except Exception as e:
            logger.critical(
                "Pandas reading of {} failed!".format(coords))
            logger.critical("Check format!")
            sys.exit()
    else:
        catalog_dict = {'ra': [], 'dec': []}
        coords = coords.split(",")
        for i in coords:
            ra_str, dec_str = i.split(" ")
            catalog_dict['ra'].append(ra_str)
            catalog_dict['dec'].append(dec_str)

        if source_names != "":
            source_names = source_names.split(",")
            if len(source_names) != len(catalog_dict['ra']):
                logger.critical(
                    ("All sources must be named "
                     "when using '--source-names'."))
                logger.critical("Please check inputs.")
                sys.exit()
        else:
            source_names = [
                "{}_{}".format(
                    i, j) for i, j in zip(
                    catalog_dict['ra'], catalog_dict['dec'])]

        catalog_dict['name'] = source_names

        catalog = pd.DataFrame.from_dict(catalog_dict)
        catalog = catalog[['name', 'ra', 'dec']]

    catalog['name'] = catalog['name'].astype(str)

    return catalog


def build_SkyCoord(catalog: pd.DataFrame) -> SkyCoord:
    """
    Create a SkyCoord array for each target source.

    Args:
        catalog: Catalog of source coordinates.

    Returns:
        Target source(s) SkyCoord.
    """
    logger = logging.getLogger()

    ra_str = catalog['ra'].iloc[0]
    if catalog['ra'].dtype == np.float64:
        hms = False
        deg = True

    elif ":" in ra_str or " " in ra_str:
        hms = True
        deg = False
    else:
        deg = True
        hms = False

    if hms:
        src_coords = SkyCoord(
            catalog['ra'],
            catalog['dec'],
            unit=(
                u.hourangle,
                u.deg))
    else:
        src_coords = SkyCoord(
            catalog['ra'],
            catalog['dec'],
            unit=(
                u.deg,
                u.deg))

    return src_coords


def read_selavy(
    selavy_path: str,
    cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load a selavy catalogue from file. Can handle VOTables and csv files.

    Args:
        selavy_path: Path to the file.
        cols: Columns to use. Defaults to None, which returns all columns.

    Returns:
        Dataframe containing the catalogue.
    """

    if selavy_path.endswith(".xml") or selavy_path.endswith(".vot"):
        df = Table.read(
            selavy_path, format="votable", use_names_over_ids=True
        ).to_pandas()
        if cols is not None:
            df = df[df.columns.intersection(cols)]
    elif selavy_path.endswith(".csv"):
        # CSVs from CASDA have all lowercase column names
        df = pd.read_csv(selavy_path, usecols=cols).rename(
            columns={"spectral_index_from_tt": "spectral_index_from_TT"}
        )
    else:
        df = pd.read_fwf(selavy_path, skiprows=[1], usecols=cols)

    # Force all flux values to be positive
    for colname in ['flux_peak', 'flux_peak_err', 'flux_int', 'flux_int_err']:
        if colname in df.columns:
            df[colname] = df[colname].abs()
    return df


def filter_selavy_components(
    selavy_df: pd.DataFrame,
    selavy_sc: SkyCoord,
    imsize: Union[Angle, Tuple[Angle, Angle]],
    target: SkyCoord
) -> pd.DataFrame:
    """
    Create a shortened catalogue by filtering out selavy components
    outside of the image.

    Args:
        selavy_df: Dataframe of selavy components.
        selavy_sc: SkyCoords containing selavy components.
        imsize: Size of the image along each axis. Can be a single Angle
            object or a tuple of two Angle objects.
        target: SkyCoord of target centre.

    Returns:
        Shortened catalogue.
    """
    seps = target.separation(selavy_sc)
    mask = seps <= imsize / 1.4
    return selavy_df[mask].reset_index(drop=True)


def simbad_search(
    objects: List[str],
    logger: Optional[logging.RootLogger] = None
) -> Union[Tuple[SkyCoord, List[str]], Tuple[None, None]]:
    """
    Searches SIMBAD for object coordinates and returns coordinates and names

    Args:
        objects: List of object names to query.
        logger: Logger to use, defaults to None.

    Returns:
        Coordinates and source names. Each will be NoneType if search fails.

    Raises:
        Exception: Simbad table length exceeds number of objects queried.
    """
    if logger is None:
        logger = logging.getLogger()

    Simbad.add_votable_fields('ra(d)', 'dec(d)', 'typed_id')

    try:
        result_table = Simbad.query_objects(objects)
        if result_table is None:
            return None, None

        ra = result_table['RA_d']
        dec = result_table['DEC_d']

        c = SkyCoord(ra, dec, unit=(u.deg, u.deg))

        simbad_names = np.array(result_table['TYPED_ID'])

        if len(simbad_names) > len(objects):
            raise Exception("Returned Simbad table is longer than the number "
                            "of queried objects. You likely have a malformed "
                            "object name in your query."
                            )

        return c, simbad_names

    # TODO: This needs better handling below.
    except Exception as e:
        logger.debug(
            "Error in performing the SIMBAD object search!\nError: %s",
            e, exc_info=True
        )
        return None, None


def match_planet_to_field(
    group: pd.DataFrame, sep_thresh: float = 4.0
) -> pd.DataFrame:
    """
    Processes a dataframe that contains observational info
    and calculates whether a planet is within 'sep_thresh' degrees of the
    observation.

    Used as part of groupby functions hence the argument
    is a group.

    Args:
        group: Required columns are planet, DATEOBS, centre-ra and centre-dec.
        sep_thresh: The separation threshold for the planet position to the
            field centre. If the planet is lower than this value then the
            planet is considered to be in the field. Unit is degrees.

    Returns:
        The group with planet location information added and filtered for only
            those which are within 'sep_thresh' degrees. Hence an empty
            dataframe could be returned.
    """

    if group.empty:
        return

    planet = group.iloc[0]['planet']
    dates = Time(group['DATEOBS'].tolist())
    fields_skycoord = SkyCoord(
        group['centre-ra'].values,
        group['centre-dec'].values,
        unit=(u.deg, u.deg)
    )

    ol = vts.get_askap_observing_location()
    with solar_system_ephemeris.set('builtin'):
        planet_coords = get_body(planet, dates, ol)

    seps = planet_coords.separation(
        fields_skycoord
    )

    group['ra'] = planet_coords.ra.deg
    group['dec'] = planet_coords.dec.deg
    group['sep'] = seps.deg

    group = group.loc[
        group['sep'] < sep_thresh
    ]

    return group


def check_racs_exists(base_dir: str) -> bool:
    """
    Check if RACS directory exists

    Args:
        base_dir: Path to base directory

    Returns:
        True if exists, False otherwise.
    """
    return os.path.isdir(os.path.join(base_dir, "EPOCH00"))


def create_source_directories(outdir: str, sources: List[str]) -> None:
    """
    Create directory for all sources in a list.

    Args:
        outdir: Base directory.
        sources: List of source names.

    Returns:
        None
    """
    logger = logging.getLogger()

    for i in sources:
        name = i.replace(" ", "_").replace("/", "_")
        name = os.path.join(outdir, name)
        os.makedirs(name)


def gen_skycoord_from_df(
    df: pd.DataFrame,
    ra_col: str = 'ra',
    dec_col: str = 'dec',
    ra_unit: u.Unit = u.degree,
    dec_unit: u.Unit = u.degree
) -> SkyCoord:
    """
    Create a SkyCoord object from a provided dataframe.

    Args:
        df: A dataframe containing the RA and Dec columns.
        ra_col: The column to use for the Right Ascension, defaults to 'ra'.
        dec_col: The column to use for the Declination, defaults to 'dec'.
        ra_unit: The unit of the RA column, defaults to degrees. Must be
            an astropy.unit value.
        dec_unit: The unit of the Dec column, defaults to degrees. Must be
            an astropy.unit value.

    Returns:
        A SkyCoord object containing the coordinates of the requested sources.
    """
    sc = SkyCoord(
        df[ra_col].values, df[dec_col].values, unit=(ra_unit, dec_unit)
    )

    return sc


def pipeline_get_eta_metric(df: pd.DataFrame, peak: bool = False) -> float:
    """
    Calculates the eta variability metric of a source.
    Works on the grouped by dataframe using the fluxes
    of the associated measurements.

    Args:
        df: A dataframe containing the grouped measurements, i.e. only
            the measurements from one source. Requires the flux_int/peak and
            flux_peak/int_err columns.
        peak: Whether to use peak flux instead of integrated, defaults to
            False.

    Returns:
        The eta variability metric.
    """
    if df.shape[0] == 1:
        return 0.

    suffix = 'peak' if peak else 'int'
    weights = 1. / df[f'flux_{suffix}_err'].values**2
    fluxes = df[f'flux_{suffix}'].values
    eta = (df.shape[0] / (df.shape[0] - 1)) * (
        (weights * fluxes**2).mean() - (
            (weights * fluxes).mean()**2 / weights.mean()
        )
    )
    return eta


def pipeline_get_variable_metrics(df: pd.DataFrame) -> pd.Series:
    """
    Calculates the variability metrics of a source. Works on the grouped by
    dataframe using the fluxes of the associated measurements.

    Args:
        df: A dataframe containing the grouped measurements, i.e. only
            the measurements from one source. Requires the flux_int/peak and
            flux_peak/int_err columns.

    Returns:
        The variability metrics, v_int, v_peak, eta_int and eta_peak
            as a pandas series.
    """
    d = {}

    if df.shape[0] == 1:
        d['v_int'] = 0.
        d['v_peak'] = 0.
        d['eta_int'] = 0.
        d['eta_peak'] = 0.
    else:
        d['v_int'] = df['flux_int'].std() / df['flux_int'].mean()
        d['v_peak'] = df['flux_peak'].std() / df['flux_peak'].mean()
        d['eta_int'] = pipeline_get_eta_metric(df)
        d['eta_peak'] = pipeline_get_eta_metric(df, peak=True)

    return pd.Series(d)


def calculate_vs_metric(
    flux_a: float, flux_b: float, flux_err_a: float, flux_err_b: float
) -> float:
    """
    Calculate the Vs variability metric which is the t-statistic that the
    provided fluxes are variable. See Section 5 of Mooley et al. (2016)
    for details, DOI: 10.3847/0004-637X/818/2/105.

    Args:
        flux_a: flux value "A".
        flux_b: flux value "B".
        flux_err_a: error of `flux_a`.
        flux_err_b: error of `flux_b`.

    Returns:
        The Vs metric for flux values "A" and "B".
    """
    return (flux_a - flux_b) / np.hypot(flux_err_a, flux_err_b)


def calculate_m_metric(flux_a: float, flux_b: float) -> float:
    """
    Calculate the m variability metric which is the modulation index between
    two fluxes.
    This is proportional to the fractional variability.
    See Section 5 of Mooley et al. (2016) for details,
    DOI: 10.3847/0004-637X/818/2/105.

    Args:
        flux_a: flux value "A".
        flux_b: flux value "B".

    Returns:
        The m metric for flux values "A" and "B".
    """
    return 2 * ((flux_a - flux_b) / (flux_a + flux_b))


def _distance_from_edge(x: np.ndarray) -> np.ndarray:
    """
    Analyses the binary array x and determines the distance from
    the edge (0).

    Args:
        x: The binary array to analyse.

    Returns:
        Array each cell containing distance from the edge.
    """
    x = np.pad(x, 1, mode='constant')
    dist = ndi.distance_transform_cdt(x, metric='taxicab')

    return dist[1:-1, 1:-1]


def create_moc_from_fits(fits_file: str, max_depth: int = 9) -> MOC:
    """
    Creates a MOC from (assuming) an ASKAP fits image
    using the cheat method of analysing the edge pixels of the image.

    Args:
        fits_file: The path of the ASKAP FITS image to generate the MOC from.
        max_depth: Max depth parameter passed to the
            MOC.from_polygon_skycoord() function, defaults to 9.

    Returns:
        The MOC generated from the FITS file.

    Raises:
        Exception: The FITS file does not exist.
    """
    if not os.path.isfile(fits_file):
        raise Exception("{} does not exist".format(fits_file))

    with open_fits(fits_file) as vast_fits:
        data = vast_fits[0].data
        if data.ndim == 4:
            data = data[0, 0, :, :]
        header = vast_fits[0].header
        wcs = WCS(header, naxis=2)

    binary = (~np.isnan(data)).astype(int)
    mask = _distance_from_edge(binary)

    x, y = np.where(mask == 1)
    # need to know when to reverse by checking axis sizes.
    pixels = np.column_stack((y, x))

    coords = SkyCoord(wcs.wcs_pix2world(
        pixels, 0), unit="deg", frame="icrs")

    moc = MOC.from_polygon_skycoord(coords, max_depth=max_depth)

    del binary
    gc.collect()

    return moc


def strip_fieldnames(fieldnames: pd.Series) -> pd.Series:
    """
    Some field names have historically used the interleaving naming scheme,
    but that has changed as of January 2023. This function removes the "A"
    that is on the end of the field names

    Args:
        fieldnames: Series to strip field names from

    Returns:
        Series with stripped field names
    """

    return fieldnames.str.rstrip('A')


def open_fits(
    fits_path: Union[str, Path],
    memmap: Optional[bool] = True,
    comp_nan_fill: Optional[bool]= True,
    comp_nan_fill_cut = -1e4,
) -> fits.HDUList:
    """
    This function opens both compressed and uncompressed fits files.

    Args:
        fits_path: Path to the fits file
        memmap: Open the fits file with mmap. Defaults to True.
        comp_nan_fill: Fill formerly-NaN values with NaNs in compressed images.
            Defaults to True.
        comp_nan_fill_cut: The cutoff value for replacing negative numbers
            with NaNs. Only relevant if `comp_nan_fill=True`. Defaults to -1e4.

    Returns:
        HDUList loaded from the fits file

    Raises:
        ValueError: File extension must be .fits or .fits.fz
    """

    if isinstance(fits_path, Path):
        fits_path = str(fits_path)

    hdul = fits.open(fits_path, memmap=memmap)

    if len(hdul) == 1:
        return hdul
    elif isinstance(hdul[1], fits.hdu.compressed.CompImageHDU):
        if comp_nan_fill:
            data = hdul[1].data
            data[data<comp_nan_fill_cut] = np.nan
        return fits.HDUList(hdul[1:])
    else:
        return hdul


def pandas_to_dask(
    df: pd.DataFrame,
    partition_size: Optional[int] = 100
) -> dd.DataFrame:
    """
    Converts a pandas dataframe to a dask dataframe.

    Args:
        df: The pandas dataframe to convert.
        partition_size: The size of each partition in MB.

    Returns:
        The dask dataframe
    """

    mem_usage = df.memory_usage(deep=True).sum()
    npartitions = int(np.ceil(mem_usage / (1024**2) / partition_size))
    ddf = dd.from_pandas(df, npartitions=npartitions)

    return ddf
