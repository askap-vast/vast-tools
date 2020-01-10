# Functions and classes related to loading and searching survey data

import argparse
import sys
import numpy as np
import os
import datetime
import pandas as pd
import warnings
import shutil

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

class Dropbox:
    '''
    
    
    '''
    
    def __init__(self, dbx):
        '''Constructor method
        '''
        
        self.dbx = dbx
        
    def recursive_build_files(base_file_list, preappend="", legacy=False):
        '''
        Very annoyingling recursive file lists do not work on shared folders.
        This function is to fetch every single file available by iterating over
        all folders found to build up a unique file list.

        :param base_file_list: a list of files in the root dropbox folder
        :type base_file_list:
        :param preappend: defaults to an empty str
        :type preappend: str, optional
        :param legacy: Whether to read legacy directoy, defaults to False
        :type legacy: bool, optional
        :returns: lists of all folders files in the dropbox
        :rtype: list, list
        '''

        folders = []
        searched_folders = []
        files = []
        for i in base_file_list.entries:
            if isinstance(i, dropbox.files.FolderMetadata):
                if preappend == "":
                    folders.append("/{}".format(i.name))
                else:
                    folders.append("/{}/{}".format(preappend, i.name))
            else:
                if preappend == "":
                    files.append("/{}".format(i.name))
                else:
                    files.append("/{}/{}".format(preappend, i.name))

        while folders != searched_folders:
            for i in folders:
                if logger.level != 10:
                    # write the next character
                    sys.stdout.write(next(spinner))
                    # flush stdout buffer (actual character display)
                    sys.stdout.flush()
                    sys.stdout.write('\b')
                # Ignore legacy folder when searching unless specified by user.
                logger.debug("Folder: {}".format(i))
                if i == "/LEGACY" and legacy is False:
                    logger.debug(
                        "Skipping LEGACY folder, "
                        "include_legacy = {}".format(legacy)
                        )
                    searched_folders.append(i)
                    continue
                if i not in searched_folders:
                    these_files = self.dbx.files_list_folder(
                        "/{}".format(i), shared_link=shared_link)
                    for j in these_files.entries:
                        if isinstance(j, dropbox.files.FolderMetadata):
                            if preappend == "" or i.startswith(
                                    "/{}".format(preappend)):
                                folders.append("{}/{}".format(i, j.name))
                            else:
                                folders.append(
                                    "/{}/{}/{}".format(preappend, i, j.name))
                        else:
                            if preappend == "" or i.startswith(
                                    "/{}".format(preappend)):
                                files.append("{}/{}".format(i, j.name))
                            else:
                                files.append(
                                    "/{}/{}/{}".format(preappend, i, j.name))
                    searched_folders.append(i)
                    logger.debug("Searched {}".format(i))
                    logger.debug("Folders: {}".format(folders))
                    logger.debug("Searched Folders: {}".format(searched_folders))
        # flush stdout buffer (actual character display)
        sys.stdout.flush()
        logger.info("Finished!")
        return files, folders


    def download_files(
            files_list,
            pwd,
            output_dir,
            shared_url,
            password,
            overwrite=False):
        '''
        Iterate over a list of files and download them from the dropbox folder

        :param files_list:
        :type files_list:
        :param pwd:
        :type pwd:
        :param output_dir:
        :type output_dir:
        :param shared_url:
        :type shared_url:
        :param password:
        :type password:
        :param overwrite: whether to overwrite existing files, defaults to False
        :type overwrite: bool, optional
        '''

        for vast_file in files_list:
            download_path = os.path.join(pwd, output_dir, vast_file[1:])
            if not overwrite:
                if os.path.isfile(download_path):
                    logger.error(
                        "{} already exists and overwrite is set to {}.".format(
                            download_path, overwrite))
                    logger.info("Skipping file.")
                    continue
            dropbox_path = "{}".format(vast_file)
            logger.debug("Download path: {}".format(download_path))
            logger.info("Downloading {}...".format(dropbox_path))
            self.dbx.sharing_get_shared_link_file_to_file(
                download_path, shared_url, path=dropbox_path,
                link_password=password)

class Fields:
    '''
    Store the coordinates of all survey fields

    :param fname: The name of the csv file containing the survey field list
    :type fname: str
    '''

    def __init__(self, fname):
        '''Constructor method
        '''
        self.fields = pd.read_csv(fname)
        self.direction = SkyCoord(Angle(self.fields["RA_HMS"],
                                        unit=u.hourangle),
                                  Angle(self.fields["DEC_DMS"], unit=u.deg))

    def find(self, src_dir, max_sep, catalog):
        '''
        Find which field each source in the catalogue is in.

        :param src_dir: Coordinates of sources to find fields for
        :type src_dir: `astropy.coordinates.sky_coordinate.SkyCoord`
        :param max_sep: Maximum allowable separation between source \
        and beam centre in degrees
        :type max_sep: float
        :param catalog: Catalogue of sources to find fields for
        :type catalog: `pandas.core.frame.DataFrame`

        :returns: An updated catalogue with nearest field data for each \
        source, and a boolean array corresponding to whether the source \
        is within max_sep
        :rtype: `pandas.core.frame.DataFrame`, `numpy.ndarray`
        '''
        nearest_beams, seps, _d3d = src_dir.match_to_catalog_sky(
            self.direction)
        within_beam = seps.deg < max_sep
        catalog["sbid"] = self.fields["SBID"].iloc[nearest_beams].values
        nearest_fields = self.fields["FIELD_NAME"].iloc[nearest_beams]
        catalog["field_name"] = nearest_fields.values
        catalog["original_index"] = catalog.index.values
        new_catalog = catalog[within_beam].reset_index(drop=True)
        logger.info(
            "Field match found for {}/{} sources.".format(
                len(new_catalog.index), len(nearest_beams)))

        if len(new_catalog.index) - len(nearest_beams) != 0:
            logger.warning(
                "No field matches found for sources with index (or name):")
            for i in range(0, len(catalog.index)):
                if i not in new_catalog["original_index"]:
                    if "name" in catalog.columns:
                        logger.warning(catalog["name"].iloc[i])
                    else:
                        logger.warning("{:03d}".format(i + 1))
        else:
            logger.info("All sources found!")

        self.field_cat = new_catalog

        return new_catalog, within_beam

    def write_fields_cat(self, outfile):
        '''
        Write the source-fields catalogue to file

        :param outfile: Name of the file to write to
        :type outfile: str
        '''

        self.field_cat.drop(
            ["original_index"],
            axis=1).to_csv(
            outfile,
            index=False)
        logger.info("Written field catalogue to {}.".format(outfile))


class Image:
    '''
    Store image data for a survey field
    Store image data for a RACS field

    :param sbid: SBID of the field
    :type sbid: str
    :param field: Name of the field
    :type field: str
    :param tiles: Use image tiles instead of mosaics, defaults to `False`
    :type tiles: bool, optional
    '''

    def __init__(self, sbid, field, tiles=False):
        '''Constructor method
        '''
        self.sbid = sbid
        self.field = field

        if tiles:
            img_template = 'image.i.SB{}.cont.{}.linmos.taylor.0.restored.fits'
            self.imgname = img_template.format(sbid, field)
        else:
            self.imgname = '{}.fits'.format(field)

        self.imgpath = os.path.join(IMAGE_FOLDER, self.imgname)

        if os.path.isfile(self.imgpath):
            self.image_fail = False
        else:
            self.image_fail = True
            logger.error(
                "{} does not exist! Unable to create postagestamps".format(
                    self.imgpath))
            return

        self.hdu = fits.open(self.imgpath)[0]
        self.wcs = WCS(self.hdu.header, naxis=2)

        try:
            self.data = self.hdu.data[0, 0, :, :]
        except Exception as e:
            self.data = self.hdu.data

    def get_rms_img(self):
        '''
        Load the BANE noisemap corresponding to the image
        '''
        self.rmsname = self.imgname.replace('.fits', '_rms.fits')

        self.rmspath = os.path.join(BANE_FOLDER, self.rmsname)

        if os.path.isfile(self.rmspath):
            self.rms_fail = False
        else:
            self.rms_fail = True
            logger.error(
                "{} does not exist! Unable to create postagestamps".format(
                    self.rmspath))
            return

        self.rms_hdu = fits.open(self.rmspath)[0]
        self.rms_wcs = WCS(self.rms_hdu.header, naxis=2)

        try:
            self.rms_data = self.rms_hdu.data[0, 0, :, :]
        except Exception as e:
            self.rms_data = self.rms_hdu.data
