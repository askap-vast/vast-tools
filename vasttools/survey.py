# Functions and classes related to loading and searching survey data

import sys
import os
import pandas as pd
import warnings
import pkg_resources
import dropbox
import itertools
import hashlib

import logging
import logging.handlers
import logging.config

from astropy.coordinates import Angle
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.exceptions import AstropyWarning, AstropyDeprecationWarning

warnings.filterwarnings('ignore', category=AstropyWarning, append=True)
warnings.filterwarnings('ignore',
                        category=AstropyDeprecationWarning, append=True)


FIELD_FILES = {
    0: pkg_resources.resource_filename(
        __name__, "./data/racs_info.csv"),
    1: pkg_resources.resource_filename(
        __name__, "./data/vast_epoch01_info.csv"),
    2: pkg_resources.resource_filename(
        __name__, "./data/vast_epoch02_info.csv"),
    99: pkg_resources.resource_filename(
        __name__, "./data/vast_epoch03_info.csv")
}

CHECKSUMS_FILE = pkg_resources.resource_filename(
                    __name__, "./data/checksums.h5")

class Dropbox:
    '''
    This is a class for downloading files from Dropbox.
    See `get_vast_pilot_dbx.py` for a full implementation.
    
    :param dbx: a `Dropbox` object containing connection information
    :type dbx: `dropbox.dropbox.Dropbox`
    '''

    def __init__(self, dbx, shared_link):
        '''Constructor method
        '''
        
        self.logger = logging.getLogger('vasttools.survey.Dropbox')
        self.logger.debug('Created Dropbox instance')
        
        self.dbx = dbx
        self.shared_link = shared_link
        
    def load_checksums(self):
        self._checksums_df = pd.read_hdf(CHECKSUMS_FILE)
        self._checksums_df.set_index(
            "file", inplace=True)

    def recursive_build_files(
            self,
            base_file_list,
            preappend="",
            legacy=False):
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

        spinner = itertools.cycle(['-', '/', '|', '\\'])

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
                if self.logger.level != 10:
                    # write the next character
                    sys.stdout.write(next(spinner))
                    # flush stdout buffer (actual character display)
                    sys.stdout.flush()
                    sys.stdout.write('\b')
                # Ignore legacy folder when searching unless specified by user.
                self.logger.debug("Folder: {}".format(i))
                if i == "/LEGACY" and legacy is False:
                    self.logger.debug(
                        "Skipping LEGACY folder, "
                        "include_legacy = {}".format(legacy)
                        )
                    searched_folders.append(i)
                    continue
                if i not in searched_folders:
                    these_files = self.dbx.files_list_folder(
                        "/{}".format(i), shared_link=self.shared_link)
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
                    self.logger.debug("Searched {}".format(i))
                    self.logger.debug("Folders: {}".format(folders))
                    self.logger.debug("Searched Folders: {}".format(searched_folders))
        # flush stdout buffer (actual character display)
        sys.stdout.flush()
        self.logger.info("Finished!")
        return files, folders

    def _checksum_check(self, dropbox_file, local_file):
        try:
            md5_correct = self._checksums_df.loc[dropbox_file].md5_checksum
            self.logger.debug("Dropbox md5: {}".format(md5_correct))
        except Exception as e:
            self.logger.warning(
                "Checksum not known for {}!".format(dropbox_file))
            self.logger.warning(
                "Are you using the latest version of this module?")
            self.logger.warning(
                "No checksum check performed on {}".format(
                    dropbox_file))
            return True

        with open(local_file, 'rb') as file_to_check:
            # read contents of the file
            data = file_to_check.read()
            # pipe contents of the file through
            md5_returned = hashlib.md5(data).hexdigest()
            self.logger.debug("Local md5: {}".format(md5_returned))

        if md5_returned == md5_correct:
            self.logger.debug("Checksum check passed.")
            return True
        else:
            self.logger.warning(
                "Checksum check failed for {}!".format(local_file))
            return False

    def download_files(
            self,
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

        self.load_checksums()

        failures = []

        for vast_file in files_list:
            download_path = os.path.join(pwd, output_dir, vast_file[1:])
            if not overwrite:
                if os.path.isfile(download_path):
                    self.logger.error(
                        "{} already exists and overwrite is set to {}.".format(
                            download_path, overwrite))
                    self.logger.info("Skipping file.")
                    continue
            dropbox_path = "{}".format(vast_file)
            self.logger.debug("Download path: {}".format(download_path))
            self.logger.info("Downloading {}...".format(dropbox_path))
            try:
                self.dbx.sharing_get_shared_link_file_to_file(
                    download_path, shared_url, path=dropbox_path,
                    link_password=password)
                download_complete = True
            except Exception as e:
                self.logger.warning("{} encountered a problem!".format(
                    vast_file
                ))
                self.logger.warning("Will try again after main cycle.")
                download_complete = False
                failures.append(vast_file)

            if download_complete:
                success = self._checksum_check(dropbox_path, download_path)
                if not success:
                    self.logger.warning(
                        "md5 checksum does"
                        " not match for {}!".format(vast_file))
                    failures.append(vast_file)
                    self.logger.warning("Will try again after main cycle.")
                else:
                    self.logger.info("Integrity check passed for {}".format(
                        vast_file
                    ))

        return failures

class Fields:
    '''
    Store the coordinates of all survey fields

    :param fname: The epoch number of fields to collect
    :type fname: int
    '''

    def __init__(self, epoch):
        '''Constructor method
        '''

        self.logger = logging.getLogger('vasttools.survey.Fields')
        self.logger.debug('Created Fields instance')

        self.fields = pd.read_csv(FIELD_FILES[epoch])
        # Epoch 99 has some empty beam directions (processing failures)
        # Drop them and any issue rows in the future.
        self.fields.dropna(inplace=True)
        self.fields.reset_index(drop=True, inplace=True)

        self.direction = SkyCoord(Angle(self.fields["RA_HMS"],
                                        unit=u.hourangle),
                                  Angle(self.fields["DEC_DMS"], unit=u.deg))

    def find(self, src_coord, max_sep, catalog):
        '''
        Find which field each source in the catalogue is in.

        :param src_coord: Coordinates of sources to find fields for
        :type src_coord: `astropy.coordinates.sky_coordinate.SkyCoord`
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

        nearest_beams, seps, _d3d = src_coord.match_to_catalog_sky(
            self.direction)
        within_beam = seps.deg < max_sep
        catalog["sbid"] = self.fields["SBID"].iloc[nearest_beams].values
        nearest_fields = self.fields["FIELD_NAME"].iloc[nearest_beams]
        catalog["field_name"] = nearest_fields.values
        catalog["original_index"] = catalog.index.values
        new_catalog = catalog[within_beam].reset_index(drop=True)
        self.logger.info(
            "Field match found for {}/{} sources.".format(
                len(new_catalog.index), len(nearest_beams)))

        if len(new_catalog.index) - len(nearest_beams) != 0:
            self.logger.warning(
                "No field matches found for sources with index (or name):")
            for i in range(0, len(catalog.index)):
                if i not in new_catalog["original_index"]:
                    if "name" in catalog.columns:
                        self.logger.warning(catalog["name"].iloc[i])
                    else:
                        self.logger.warning("{:03d}".format(i + 1))
        else:
            self.logger.info("All sources found!")

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
        self.logger.info("Written field catalogue to {}.".format(outfile))


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
    :param IMAGE_FOLDER: Path to image directory
    :type IMAGE_FOLDER: str
    :param RMS_FOLDER: Path to RMS map directory
    :type RMS_FOLDER: str
    '''

    def __init__(self, sbid, field, IMAGE_FOLDER, RMS_FOLDER, tiles=False):
        '''Constructor method
        '''
        
        self.logger = logging.getLogger('vasttools.survey.Image')
        self.logger.debug('Created Image instance')
        
        self.sbid = sbid
        self.field = field
        self.RMS_FOLDER = RMS_FOLDER

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
            self.logger.error(
                "{} does not exist! Unable to create postagestamps".format(
                    self.imgpath))
            return

        with fits.open(self.imgpath) as hdul:
            self.header = hdul[0].header
            self.wcs = WCS(self.header, naxis=2)

            try:
                self.data = hdul[0].data[0, 0, :, :]
            except Exception as e:
                self.data = hdul[0].data

    def get_rms_img(self):
        '''
        Load the noisemap corresponding to the image
        '''
        self.rmsname = self.imgname.replace('.fits', '_rms.fits')

        self.rmspath = os.path.join(self.RMS_FOLDER, self.rmsname)

        if os.path.isfile(self.rmspath):
            self.rms_fail = False
        else:
            self.rms_fail = True
            self.logger.error(
                "{} does not exist! Unable to create postagestamps".format(
                    self.rmspath))
            return

        with fits.open(self.rmspath) as hdul:
            self.rms_header = hdul[0].header
            self.rms_wcs = WCS(self.rms_header, naxis=2)

            try:
                self.rms_data = hdul[0].data[0, 0, :, :]
            except Exception as e:
                self.rms_data = hdul[0].rms_hdu.data
