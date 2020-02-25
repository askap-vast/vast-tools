#!/usr/bin/env python

# Example command:

# ./build_lightcurves.py

import argparse
import sys
import numpy as np
import os
import glob
import datetime
import pandas as pd
import warnings

import logging
import logging.handlers
import logging.config

import matplotlib.pyplot as plt

from astropy.coordinates import Angle
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.nddata.utils import Cutout2D
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from astropy.utils.exceptions import AstropyWarning, AstropyDeprecationWarning

from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection
from astropy.visualization import ZScaleInterval, ImageNormalize
from astropy.visualization import PercentileInterval
from astropy.visualization import AsymmetricPercentileInterval
from astropy.visualization import LinearStretch
import matplotlib.axes as maxes
from mpl_toolkits.axes_grid1 import make_axes_locatable

warnings.filterwarnings('ignore', category=AstropyWarning, append=True)
warnings.filterwarnings('ignore',
                        category=AstropyDeprecationWarning, append=True)

try:
    import colorlog
    use_colorlog = True
except ImportError:
    use_colorlog = False


class Lightcurve:
    '''
    This is a class representation of the lightcurve of a source
    :param name: Name of the source
    :type name: str
    :param num_obs: Total number of observations
    :type num_obs: int
    '''

    def __init__(self, name, num_obs):
        '''Constructor method
        '''
        self.name = name
        self.observations = pd.DataFrame(
            columns=[
                'obs_start',
                'obs_end',
                'S_int',
                'S_err',
                'img_rms',
                'upper_lim'],
            index=np.arange(
                0,
                num_obs))

    def add_observation(self, i, row):
        '''
        Add a single observation to the lightcurve
        :param i: Observation number (i.e. the row in the observations table)
        :type i: int
        :param row: Row containing flux measurements, background estimates \
        and observation date
        :type row: `pandas.core.series.Series`
        '''
        S_int = row['flux_int']
        S_err = row['rms_image']
        img_rms = row['SELAVY_rms']*1e3
        obs_start = pd.to_datetime(row['obs_date'])
        obs_end = pd.to_datetime(row['date_end'])

        if np.isnan(S_int):
            logger.debug("Observation is a non-detection")
            upper_lim = True
        else:
            logger.debug("Observation is a detection")
            upper_lim = False

        self.observations.iloc[i] = [
            obs_start, obs_end, S_int, S_err, img_rms, upper_lim]

    def plot_lightcurve(self, sigma_thresh=5, savefile=None, figsize=(8,4), min_points=2, min_detections=1):
        '''
        Plot source lightcurves and save to file
        :param sigma_thresh: Threshold to use for upper limits, defaults to 5
        :type sigma_thresh: int or float
        :param savefile: Filename to save plot, defaults to None
        :type savefile: str
        :param min_points: Minimum number of points for plotting, defaults to 2
        :type min_points: int, optional
        :param min_detections: Minimum number of detections for plotting, defaults to 1
        :type min_detections: int, optional
        '''
        
        num_obs = self.observations['obs_start'].count()
        num_detections = (self.observations['upper_lim'] == False).sum()
        
        if num_obs < min_points:
            return
            
        if num_detections < min_detections:
            return
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.set_title(self.name)

        ax.set_xlabel('Date')
        ax.set_ylabel('Flux Density (mJy)')
        

        for i, row in self.observations.iterrows():
            if row['upper_lim']:
                logger.debug("Plotting upper limit")
                ax.errorbar(
                    row['obs_start'],
                    sigma_thresh *
                    row['img_rms'],
                    yerr=row['img_rms'],
                    uplims=True,
                    lolims=False,
                    marker='_',
                    c='k')
            else:
                logger.debug("Plotting detection")
                ax.errorbar(
                    row['obs_start'],
                    row['S_int'],
                    yerr=row['S_err'],
                    marker='o',
                    c='k')
        
        ax.set_ylim(bottom=0)
        plt.savefig(savefile)
        plt.close()
        
    def write_lightcurve(self, savefile, min_points=2):
        '''
        Output the lightcurve for processing with external scripts
        
        :param savefile: Output filename
        :type savefile: str
        :param min_points: Minimum number of points for plotting, defaults to 2
        :type min_points: int, optional
        '''
        
        num_obs = self.observations['obs_start'].count()
        
        if num_obs < min_points:
            return
        self.observations.to_csv(savefile, index=False)
        


def create_lightcurves(crossmatch_paths):
    '''
    Create a lightcurve for each source by looping over all observation files
    :param crossmatch_paths: List of observation file paths
    :type crossmatch_paths: list
    :return: Dictionary of lightcurve objects
    :rtype: dict
    '''

    num_obs = len(crossmatch_paths)
    lightcurve_dict = {}
    for i, path in enumerate(crossmatch_paths):
        path = os.path.abspath(path)

        if not os.path.isfile(path):
            logger.critical("{} not found!".format(path))
            sys.exit()
        try:
            source_list = pd.read_csv(path)
        except Exception as e:
            logger.critical("Pandas reading of {} failed!".format(path))
            logger.critical("Check format!")
            continue
        for j, row in source_list.iterrows():
            name = row['name']
            if name not in lightcurve_dict.keys():
                lightcurve_dict[name] = Lightcurve(name, num_obs)

            lightcurve_dict[name].add_observation(i, row)

    return lightcurve_dict


def plot_lightcurves(lightcurve_dict, folder=''):
    '''
    Plot a lightcurve for each source
    :param lightcurve_dict:
    :type lightcurve_dict: dict
    '''

    for name, lightcurve in lightcurve_dict.items():
        lightcurve.plot_lightcurve(savefile=os.path.join(folder,name+'.png'))
        
def write_lightcurves(lightcurve_dict, folder=''):
    '''
    Plot a lightcurve for each source
    :param lightcurve_dict:
    :type lightcurve_dict: dict
    '''

    for name, lightcurve in lightcurve_dict.items():
        savefile = os.path.join(folder,name+'_lightcurve.csv')
        lightcurve.write_lightcurve(savefile)


# Force nice
os.nice(5)

runstart = datetime.datetime.now()

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    'folder',
    type=str,
    help='')
#parser.add_argument(
#    'info_file',
#    metavar="input.csv",
#    type=str,
#    help='')
parser.add_argument(
    '--quiet',
    action="store_true",
    help='Turn off non-essential terminal output.')
parser.add_argument(
    '--debug',
    action="store_true",
    help='Turn on debug output.')

args = parser.parse_args()

logger = logging.getLogger()
s = logging.StreamHandler()
fh = logging.FileHandler(
    "build_lightcurves_{}.log".format(
        runstart.strftime("%Y%m%d_%H:%M:%S")))
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
fh.setFormatter(formatter)

if args.debug:
    s.setLevel(logging.DEBUG)
else:
    if args.quiet:
        s.setLevel(logging.WARNING)
    else:
        s.setLevel(logging.INFO)

logger.addHandler(s)
logger.addHandler(fh)
logger.setLevel(logging.DEBUG)

'''
info_file = os.path.abspath(args.info_file)
logger.info("Getting file list from {}".format(info_file))
if not os.path.isfile(info_file):
    logger.critical("{} not found!".format(info_file))
    sys.exit()
try:
    crossmatch_paths = pd.read_csv(info_file)
except Exception as e:
    logger.critical("Pandas reading of {} failed!".format(info_file))
    logger.critical("Check format!")
    sys.exit()
'''
crossmatch_paths = glob.glob(os.path.join(args.folder,'*crossmatch*.csv'))

lightcurve_dict = create_lightcurves(crossmatch_paths)

write_lightcurves(lightcurve_dict, folder=args.folder)
#plot_lightcurves(lightcurve_dict, folder=args.folder)
