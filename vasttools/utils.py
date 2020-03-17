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


def get_logger(debug, quiet, logfile=None):
    '''
    Set up the logger

    :param logfile: File to output log to
    :type logfile: str
    :param debug: Set stream level to debug
    :type debug: bool
    :param quiet: Suppress all non-essential output
    :type quiet: bool
    :param usecolorlog: Use colourful logging scheme, defaults to False
    :type usecolorlog: bool, optional

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
