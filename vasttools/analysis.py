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

import matplotlib
import matplotlib.pyplot as plt

from astropy.coordinates import Angle
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.nddata.utils import Cutout2D
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time
from astropy.timeseries import TimeSeries
from astropy.wcs.utils import skycoord_to_pixel
from astropy.utils.exceptions import AstropyWarning, AstropyDeprecationWarning

from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection
from astropy.visualization import ZScaleInterval, ImageNormalize
from astropy.visualization import PercentileInterval
from astropy.visualization import AsymmetricPercentileInterval
from astropy.visualization import LinearStretch
import matplotlib.axes as maxes
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

matplotlib.pyplot.switch_backend('Agg')


class Lightcurve:
    '''
    This is a class representation of the lightcurve of a source

    :param name: Name of the source
    :type name: str
    :param num_obs: Total number of observations
    :type num_obs: int
    '''

    def __init__(self, name, num_obs, islands=False):
        '''Constructor method
        '''
        self.logger = logging.getLogger(
            'vasttools.build_lightcurves.Lightcurve')
        self.name = name.strip()
        self.islands = islands
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
        if self.islands:
            S_err = row['background_noise']
        else:
            S_err = row['rms_image']
        img_rms = row['SELAVY_rms'] * 1e3
        obs_start = pd.to_datetime(row['obs_date'])
        obs_end = pd.to_datetime(row['date_end'])

        if np.isnan(S_int):
            self.logger.debug("Observation is a non-detection")
            upper_lim = True
        else:
            self.logger.debug("Observation is a detection")
            upper_lim = False

        self.observations.iloc[i] = [
            obs_start, obs_end, S_int, S_err, img_rms, upper_lim]

    def _drop_empty(self):
        '''
        Drop empty rows
        '''

        self.observations = self.observations.dropna(how='all')

    def _infer_objects(self):
        '''
        Infer the dtype objects of the dataframe.
        '''

        self.observations = self.observations.infer_objects()

    def plot_lightcurve(self, sigma_thresh=5, savefile=None, figsize=(8, 4),
                        min_points=2, min_detections=1, mjd=False,
                        grid=False):
        '''
        Plot source lightcurves and save to file

        :param sigma_thresh: Threshold to use for upper limits, defaults to 5
        :type sigma_thresh: int or float
        :param savefile: Filename to save plot, defaults to None
        :type savefile: str
        :param min_points: Minimum number of points for plotting, defaults to 2
        :type min_points: int, optional
        :param min_detections: Minimum number of detections for plotting, \
        defaults to 1
        :type min_detections: int, optional
        :param mjd: Plot x-axis in MJD rather than datetime, defaults to False
        :type mjd: bool, optional
        :param grid: Turn on matplotlib grid, defaults to False
        :type grid: bool, optional
        '''

        num_obs = self.observations['obs_start'].count()
        num_detections = (~self.observations['upper_lim']).sum()

        if num_obs < min_points:
            self.logger.warning(
                "Minimum number of data points ({}) not met"
                " for {} ({}).".format(min_points, self.name, num_obs)
            )
            self.logger.warning("Skipping plot.")
            return False

        if num_detections < min_detections:
            self.logger.warning(
                "Minimum number of detections ({}) not met"
                " for {} ({}).".format(
                    min_detections, self.name, num_detections
                )
            )
            self.logger.warning("Skipping plot.")
            return False

        plot_date = self.observations['obs_start']
        if mjd:
            self.observations['plot_date'] = Time(plot_date.to_numpy()).mjd
        else:
            self.observations['plot_date'] = plot_date

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.set_title(self.name)

        ax.set_ylabel('Flux Density (mJy)')

        for i, row in self.observations.iterrows():
            if row['upper_lim']:
                self.logger.debug("Plotting upper limit")
                ax.errorbar(
                    row['plot_date'],
                    sigma_thresh *
                    row['img_rms'],
                    yerr=row['img_rms'],
                    uplims=True,
                    lolims=False,
                    marker='_',
                    c='k')
            else:
                self.logger.debug("Plotting detection")
                ax.errorbar(
                    row['plot_date'],
                    row['S_int'],
                    yerr=row['S_err'],
                    marker='o',
                    c='k')

        ax.set_ylim(bottom=0)

        if mjd:
            ax.set_xlabel('Date (MJD)')
        else:
            fig.autofmt_xdate()
            ax.set_xlabel('Date')

            date_form = mdates.DateFormatter("%Y-%m-%d")
            ax.xaxis.set_major_formatter(date_form)
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))

        plt.grid(grid)
        plt.savefig(savefile)
        plt.close()

        return True

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
            self.logger.warning(
                "Minimum number of data points ({}) not met"
                " for {} ({}).".format(min_points, self.name, num_obs)
            )
            self.logger.warning("Skipping plot.")
            return False

        self.observations.to_csv(savefile, index=False)

        return True


class BuildLightcurves:
    '''
    This is a class representation of various information about a \
    series of lightcurves.

    :param args: Arguments namespace
    :type args: `argparse.Namespace`
    '''

    def __init__(self, args):
        '''Constructor method
        '''

        self.logger = logging.getLogger(
            'vasttools.build_lightcurves.BuildLightcurves')
        self.args = args

        self.crossmatch_paths = self.build_paths()

    def create_lightcurves(self):
        '''
        Create a lightcurve for each source by looping over all \
        observation files

        :return: Dictionary of lightcurve objects
        :rtype: dict
        '''

        num_obs = len(self.crossmatch_paths)
        self.logger.info(
            "Creating lightcurves from {} observations".format(num_obs))

        lightcurve_dict = {}
        for i, path in enumerate(self.crossmatch_paths):
            path = os.path.abspath(path)

            if not os.path.isfile(path):
                self.logger.critical("{} not found!".format(path))
                sys.exit()
            try:
                source_list = pd.read_csv(path)
            except Exception as e:
                self.logger.critical(
                    "Pandas reading of {} failed!".format(path))
                self.logger.critical("Check format!")
                continue
 
            if i == 0:
                if "component_id" in source_list.columns:
                    self.logger.debug("Component mode.")
                    self.islands = False
                else:
                    self.logger.debug("Island mode.")
                    self.islands = True

            for j, row in source_list.iterrows():
                name = row['name']
                if name not in lightcurve_dict.keys():
                    lightcurve_dict[name] = Lightcurve(
                        name, num_obs, islands=self.islands
                    )
                    self.logger.info("Building lightcurve for {}".format(name))

                lightcurve_dict[name].add_observation(i, row)

        for name in lightcurve_dict.keys():
            lightcurve_dict[name]._drop_empty()
            lightcurve_dict[name]._infer_objects()

        self.logger.info("Lightcurve creation complete")

        return lightcurve_dict

    def plot_lightcurves(self, lightcurve_dict, folder=''):
        '''
        Plot a lightcurve for each source

        :param lightcurve_dict:
        :type lightcurve_dict: dict
        '''

        min_points = self.args.min_points
        min_detections = self.args.min_detections

        self.logger.debug(
            "mjd: {}"
            "; grid: {}".format(self.args.mjd, self.args.grid)
        )

        for name, lightcurve in lightcurve_dict.items():
            savefile = os.path.join(folder, name + '.png')
            savefile = savefile.replace(' ', '_')

            success = lightcurve.plot_lightcurve(
                savefile=savefile,
                min_points=min_points,
                min_detections=min_detections,
                mjd=self.args.mjd,
                grid=self.args.grid
            )
            if success:
                self.logger.info(
                    "Wrote {} lightcurve plot to {}".format(
                        name, savefile))

    def write_lightcurves(self, lightcurve_dict, folder=''):
        '''
        Plot a lightcurve for each source

        :param lightcurve_dict:
        :type lightcurve_dict: dict
        '''

        min_points = self.args.min_points

        for name, lightcurve in lightcurve_dict.items():
            savefile = os.path.join(folder, name + '_lightcurve.csv')
            savefile = savefile.replace(' ', '_')

            success = lightcurve.write_lightcurve(
                savefile, min_points=min_points
            )
            if success:
                self.logger.info(
                    "Wrote {} lightcurve to {}".format(
                        name, savefile))

    def build_paths(self):
        '''
        Build list of paths to crossmatch files

        :return: list of crossmatch paths
        :rtype: list
        '''

        crossmatch_paths = sorted(glob.glob(
            os.path.join(
                self.args.folder,
                '*crossmatch*.csv')))
        self.logger.info(
            'Getting lightcurve info from:\n{}'.format(
                '\n'.join(crossmatch_paths)))

        return crossmatch_paths

    def run_query(self):
        '''
        Run the query
        '''

        lightcurve_dict = self.create_lightcurves()

        self.logger.info("Writing lightcurves to file")
        self.write_lightcurves(lightcurve_dict, folder=self.args.folder)

        if self.args.no_plotting:
            self.logger.info("Not plotting lightcurves")
        else:
            self.plot_lightcurves(lightcurve_dict, folder=self.args.folder)
