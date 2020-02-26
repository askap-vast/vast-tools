import logging
import logging.handlers
import logging.config

try:
    import colorlog
    use_colorlog = True
except ImportError:
    use_colorlog = False


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
