import os
import logging
import logging.handlers
import logging.config

try:
    import colorlog
    use_colorlog = True
except ImportError:
    use_colorlog = False

# crosshair imports
from matplotlib.transforms import Affine2D
from matplotlib.path import Path
import matplotlib.markers
import matplotlib.lines
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
import numpy as np
from multiprocessing_logging import install_mp_handler
from astroquery.simbad import Simbad
from vasttools.survey import get_askap_observing_location
from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris
from astropy.coordinates import get_body, get_moon


def get_logger(debug, quiet, logfile=None):
    '''
    Set up the logger

    :param debug: Set stream level to debug
    :type debug: bool
    :param quiet: Suppress all non-essential output
    :type quiet: bool
    :param logfile: File to output log to
    :type logfile: str

    :returns: Logger
    :rtype: `logging.RootLogger`
    '''

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


def _set_crosshair(self):
    """This module adds a true crosshair marker to matplotlib.

    ============================== ===========================================
    marker                         description
    ============================== ===========================================
    `"c"`                          crosshair

    Usage
    -----

    import matplotlib.pyplot as plt
    import crosshair
    plt.scatter(0,0, marker='c', s=100)
    plt.show()

    Notes
    -----

    I tried to stay as close to the style of `matplotlib/lib/markers.py`,
    so it can easily implemented in mpl after further testing.

    How to implement this in matplotlib via a module was inspired by:
    https://stackoverflow.com/a/16655800/5064815

    Be aware that for small sizes the crosshair looks like four dots or
    even a circle.  This is due to the fact that in this case the linewidth
    is larger then the length of the 'hairs' of the crosshair. This is know
    and similar behaviour is seen for other markers at small sizes.

    Author
    ------
    L. A. Boogaard (13/07/2017)

    """

    _crosshair_path = Path([(0.0, -0.5),  # center, bottom
                            (0.0, -0.25),  # center, q_bot
                            (-0.5, 0.0),  # left, center
                            (-0.25, 0.0),  # q_left, center
                            (0.0, 0.25),  # center, q_top
                            (0.0, 0.5),  # center, top
                            (0.25, 0.0),  # q_right, center
                            (0.5, 0.0)],  # right, center
                           [Path.MOVETO,
                            Path.LINETO,
                            Path.MOVETO,
                            Path.LINETO,
                            Path.MOVETO,
                            Path.LINETO,
                            Path.MOVETO,
                            Path.LINETO])

    self._transform = Affine2D().scale(1.0)
    self._snap_threshold = 1.0
    self._filled = False
    self._path = _crosshair_path


def crosshair():
    """
    A wrapper function to set the crosshair marker in
    matplotlib using the function written by L. A. Boogaard.
    """

    matplotlib.markers.MarkerStyle._set_crosshair = _set_crosshair
    matplotlib.markers.MarkerStyle.markers['c'] = 'crosshair'
    matplotlib.lines.Line2D.markers = matplotlib.markers.MarkerStyle.markers


def check_file(path):
    '''
    Check if logging file exists
    
    :param path: filepath to check
    :type path: str
    '''
    
    logger = logging.getLogger()
    exists = os.path.isfile(path)
    if not exists:
        logger.critical(
            "Cannot find file '%s'!", path
        )
    return exists


def build_catalog(coords, source_names):
    '''
    Build the catalogue of target sources
    
    :param coords: 
    :type coords: 
    :param source_names: 
    :type source_names: 

    :returns: Catalogue of target sources
    :rtype: `pandas.core.frame.DataFrame`
    '''

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


def build_SkyCoord(catalog):
    '''
    Create a SkyCoord array for each target source
    
    :param catalog: Catalog of source coordinates
    :type catalog: `pandas.core.frame.DataFrame`

    :returns: Target source SkyCoord
    :rtype: `astropy.coordinates.sky_coordinate.SkyCoord`
    '''

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


def filter_selavy_components(selavy_df, selavy_sc, imsize, target):
    '''
    Create a shortened catalogue by filtering out selavy components
    outside of the image
    
    :param selavy_df: Dataframe of selavy components
    :type selavy_df: `pandas.core.frame.DataFrame`
    :param selavy_sc: SkyCoords containing selavy components
    :type selavy_sc: `astropy.coordinates.sky_coordinate.SkyCoord
    :param imsize: Size of the image along each axis
    :type imsize: `astropy.coordinates.angles.Angle` or tuple of two
        `Angle` objects
    :param target: SkyCoord of target centre
    :type target: `astropy.coordinates.sky_coordinate.SkyCoord`

    :returns: Shortened catalogue
    :rtype: `pandas.core.frame.DataFrame`
    '''

    seps = target.separation(selavy_sc)
    mask = seps <= imsize / 1.4
    return selavy_df[mask].reset_index(drop=True)


def simbad_search(objects, logger=None):
    """
    Searches SIMBAD for object coordinates and returns coordinates and names
    
    :param objects: 
    :type objects: 
    :param logger: Logger to use, defaults to None
    :type logger: , optional
    
    :returns:
    :rtype:
    """

    Simbad.add_votable_fields('ra(d)', 'dec(d)')

    try:
        result_table = Simbad.query_objects(objects)
        if result_table is None:
            return None, None

        ra = result_table['RA_d']
        dec = result_table['DEC_d']

        c = SkyCoord(ra, dec, unit=(u.deg, u.deg))

        names = [i.decode("utf-8") for i in result_table['MAIN_ID']]

        return c, names

    except Exception as e:
        logger.debug(
            "Error in performing the SIMBAD object search!", exc_info=True
        )
        return None, None


def match_planet_to_field(group):
    '''
    
    :param group:
    :type group:
    
    :returns:
    :rtype:
    '''
    planet = group.iloc[0]['planet']
    dates = Time(group['DATEOBS'].tolist())
    fields_skycoord = SkyCoord(
        group['centre-ra'].values,
        group['centre-dec'].values,
        unit=(u.deg, u.deg)
    )

    ol = get_askap_observing_location()
    with solar_system_ephemeris.set('builtin'):
        planet_coords = get_body(planet, dates, ol)

    seps = planet_coords.separation(
        fields_skycoord
    )

    group['ra'] = planet_coords.ra.deg
    group['dec'] = planet_coords.dec.deg
    group['sep'] = seps.deg

    group = group.loc[
        group['sep'] < 4.0
    ]

    return group


def check_racs_exists(base_dir):
    '''
    Check if RACS directory exists
    
    :param base_dir: Path to base directory
    :type base_dir: str
    
    :returns: True if exists, False otherwise
    :rtype: bool
    '''

    return os.path.isdir(os.path.join(base_dir, "EPOCH00"))


def create_source_directories(outdir, sources):
    '''
    Create directory for all sources in a list

    :param outdir: Base directory
    :type outdir: str
    :param sources: List of sources
    :type sources: list
    '''

    logger = logging.getLogger()

    for i in sources:
        name = i.replace(" ", "_").replace("/", "_")
        name = os.path.join(outdir, name)
        os.makedirs(name)
