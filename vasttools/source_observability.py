#!/usr/bin/env python

import astropy
import numpy as np
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import astropy.units as u
import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt

import logging
import logging.config

runstart = datetime.datetime.now()

logger = logging.getLogger()
s = logging.StreamHandler()
fh = logging.FileHandler("source_observability_{}.log".format(runstart.strftime("%Y%m%d_%H:%M:%S")))
fh.setLevel(logging.DEBUG)
logformat='[%(asctime)s] - %(levelname)s - %(message)s'

formatter = logging.Formatter(logformat, datefmt="%Y-%m-%d %H:%M:%S")

s.setFormatter(formatter)
fh.setFormatter(formatter)

s.setLevel(logging.INFO)

logger.addHandler(s)
logger.addHandler(fh)
logger.setLevel(logging.DEBUG)


def load_file(filepath):
    logger.info("Loading file {}".format(filepath))
    #Give explicit check to file existence
    user_file = os.path.abspath(filepath)
    if not os.path.isfile(user_file):
        logger.critical("{} not found!")
        sys.exit()
    try:
        catalog = pd.read_csv(user_file, comment="#")
        catalog.columns = map(str.lower, catalog.columns)
        if ("ra" not in catalog.columns) or ("dec" not in catalog.columns):
            logger.critical("Cannot find one of 'ra' or 'dec' in input file.")
            logger.critical("Please check column headers!")
            sys.exit()
        if "name" not in catalog.columns:
            catalog["name"] = ["{}_{}".format(i,j) for i,j in zip(catalog['ra'], catalog['dec'])]
    except:
        logger.critical("Pandas reading of {} failed!".format(filepath))
        logger.critical("Check format!")
        sys.exit()
        
    return catalog
        
def make_skycoord(catalog):
    if catalog['ra'].dtype == np.float64:
      hms = False
      deg = True
      
    elif ":" in catalog['ra'].iloc[0]:
        hms = True
        deg = False
    else:
        deg = True
        hms = False
        
    if hms:
        src_coords = SkyCoord(catalog['ra'], catalog['dec'], unit=(u.hourangle, u.deg))
    else:
        src_coords = SkyCoord(catalog['ra'], catalog['dec'], unit=(u.deg, u.deg))
        
    return src_coords
    
def get_times(start, end, delta_t=5*u.min):
    obs_length = end-start
    
    num_points = obs_length.to(u.s)/delta_t.to(u.s)
    
    times = start + np.linspace(0,obs_length.to(u.s), num_points+1)
    
    return times
    

def radio_telescope_info():
    ATCA = EarthLocation(lat=-30.31277778*u.deg, lon=149.55000000*u.deg, height=236.87*u.m)
    ASKAP = EarthLocation(lat=-26.70416667*u.deg, lon=116.65888889*u.deg, height=100*u.m)
    VLA = EarthLocation(lat=34.07874917*u.deg, lon=-107.61777778*u.deg, height=100*u.m)
    MeerKAT = EarthLocation(lat=-30.7130*u.deg, lon=21.4430*u.deg, height=100*u.m)
    Apertif = EarthLocation(lat=52.9145*u.deg, lon=6.6027*u.deg, height=100*u.m)
    GMRT = EarthLocation(lat=19.093495*u.deg, lon=74.050333*u.deg, height=100*u.m)

    return locals()
  

def calc_altitude(coords, times, telescope_location):
    alts = []    
    if not coords.shape:
        coord_altaz = coords.transform_to(AltAz(obstime=times, location=telescope_location))
        return coord_altaz.alt
        
    for coord in coords:
        coord_altaz = coord.transform_to(AltAz(obstime=times, location=telescope_location))
        alts.append(coord_altaz.alt)
    
    return alts

def plot_altitude(target_cat, times, alts, prical=None):
    times_dt = times.datetime

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylim(0,90)
    ax.set_xlim(times_dt[0],  times_dt[-1])
    
    for i, alt in enumerate(alts):
        target = target_cat.iloc[i]
        ax.plot(times_dt, alt, label=target['name'])
        
    if prical:
        ax.plot(times_dt, prical, label='Primary Calibrator', c='k', linestyle='--')
         
    ax.axhline(12,c='0.7', zorder=0)
    ax.axhline(30,c='0.4', zorder=0)
    plt.legend()
    
    plt.show() 


if __name__ == '__main__':
    telescope = 'ATCA'
    start_time = Time('2019-12-08 13:00:00')
    end_time = Time('2019-12-08 19:00:00')
    
    
    target_cat = load_file('lensing_targets_gt.csv')
    coords = make_skycoord(target_cat)
    for coord in coords:
        print(coord.to_string('hmsdms',sep=':'))
    

    
    time_list = get_times(start_time, end_time)
      
    telescope_dict = radio_telescope_info()
    
    telescope_location = telescope_dict[telescope]
    
    print(time_list.sidereal_time('mean',telescope_location.lon))
    
    alts = calc_altitude(coords, time_list, telescope_location)
    
    prical_alts = calc_altitude(SkyCoord('19:39:25.026','-63:42:45.63', unit=(u.hourangle, u.deg)), time_list, telescope_location)
    
    plot_altitude(target_cat, time_list, alts, prical=prical_alts)
    
    
    
    
    
