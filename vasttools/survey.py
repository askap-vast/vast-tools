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
from radio_beam import Beam

warnings.filterwarnings('ignore', category=AstropyWarning, append=True)
warnings.filterwarnings('ignore',
                        category=AstropyDeprecationWarning, append=True)


RELEASED_EPOCHS = {
    "1": "01",
    "2": "02",
    "3x": "03x",
    "4x": "04x",
    "5x": "05x",
    "6x": "06x",
    "8": "08",
    "9": "09",
    "10x": "10x",
    "11x": "11x"
}

FIELD_FILES = {
    "0": pkg_resources.resource_filename(
        __name__, "./data/racs_info.csv"),
    "1": pkg_resources.resource_filename(
        __name__, "./data/vast_epoch01_info.csv"),
    "2": pkg_resources.resource_filename(
        __name__, "./data/vast_epoch02_info.csv"),
    "3x": pkg_resources.resource_filename(
        __name__, "./data/vast_epoch03x_info.csv"),
    "4x": pkg_resources.resource_filename(
        __name__, "./data/vast_epoch04x_info.csv"),
    "5x": pkg_resources.resource_filename(
        __name__, "./data/vast_epoch05x_info.csv"),
    "6x": pkg_resources.resource_filename(
        __name__, "./data/vast_epoch06x_info.csv"),
    "8": pkg_resources.resource_filename(
        __name__, "./data/vast_epoch08_info.csv"),
    "9": pkg_resources.resource_filename(
        __name__, "./data/vast_epoch09_info.csv"),
    "10x": pkg_resources.resource_filename(
        __name__, "./data/vast_epoch10x_info.csv"),
    "11x": pkg_resources.resource_filename(
        __name__, "./data/vast_epoch11x_info.csv")
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
        self._checksums_df = None

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
                    self.logger.debug(
                        "Searched Folders: {}".format(searched_folders))
        # flush stdout buffer (actual character display)
        sys.stdout.flush()
        self.logger.info("Finished!")
        return files, folders

    def _checksum_check(self, dropbox_file, local_file):
        '''
        Performs a checksum check on the Dropbox file downloaded.
        The Dropbox checksums are included in an h5 file in the module
        and returns True if they match and False if they do not.

        :param dropbox_file: path of the Dropbox file to check (also
            acts as the 'name')
        :type dropbox_file: str
        :param local_file: The path of the local file to check.
        :type local_file: str
        :returns: True if checksums match, False if they don't match.
        :rtype: bool
        '''
        if self._checksums_df is None:
            self.load_checksums()
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
            output_dir,
            shared_url,
            password,
            max_retries,
            main_overwrite,
            checksum_check=True):
        '''
        A function to download a list of files from Dropbox.
        If a file is not found or corrupted then it retries to
        a user requested number of times. It calls the
        'download_file' function to perform the actual download.

        :param files_list: the list of dropbox files to download
        :type files_list: list
        :param output_dir: The output directory where the downloads go
        :type output_dir: str
        :param shared_url: The Dropbox shared url.
        :type shared_url: str
        :param password: Dropbox link password
        :type password: str
        :param max_retries: Number of times to attempt re-downloads.
        :type max_retries: int
        :param main_overwrite: The user requested overwrite variable
        :type main_overwrite: bool
        :param checksum_check: Select whether to check the checksum of
            the file downloaded, defaults to True.
        :type checksum_check: bool, optional
        :returns: list of files that have failed to download
        :rtype: list
        '''
        failures = ["FILLER"]
        retry_count = 0
        complete_failures = []
        pwd = os.getcwd()
        while len(failures) > 0:
            if retry_count > max_retries:
                complete_failures = files_list
                failures = []
            else:
                if retry_count > 0:
                    self.logger.info(
                        "Retry attempt {}/{}".format(
                            retry_count, max_retries))
                    self.logger.info(
                        "Reattempting to download"
                        " {} files".format(len(files_list)))
                    overwrite = True
                else:
                    overwrite = main_overwrite

                failures = []

                for vast_file in files_list:
                    download_path = os.path.join(
                        pwd, output_dir, vast_file[1:])
                    dropbox_path = "{}".format(vast_file)
                    if not overwrite:
                        if os.path.isfile(download_path):
                            self.logger.warning(
                                "{} already exists and overwrite "
                                "is set to {}.".format(
                                    download_path, overwrite))
                            self.logger.info(
                                "Checking integrity..."
                            )
                            good_file = self._checksum_check(
                                dropbox_path,
                                download_path
                            )
                            if good_file is False:
                                self.logger.warning(
                                    "Redownloading {}".format(
                                        vast_file
                                    )
                                )
                            else:
                                self.logger.info("Checksum check passed!")
                                self.logger.info("Skipping file.")
                                continue
                    self.logger.debug(
                        "Download path: {}".format(download_path))
                    download_success = self.download_file(
                        download_path,
                        dropbox_path,
                        shared_url,
                        password,
                        checksum_check=checksum_check
                    )
                    if download_success is False:
                        failures.append(vast_file)
                files_list = failures
                retry_count += 1

        return complete_failures

    def download_file(
            self,
            download_path,
            dropbox_path,
            shared_url,
            password,
            checksum_check=True):
        '''
        A function to download a single file from Dropbox.

        :param download_path: The path where to download the file to.
        :type files_list: str
        :param dropbox_path: The Dropbox path of the file to download.
        :type dropbox_path: str
        :param shared_url: The Dropbox shared url.
        :type shared_url: str
        :param password: Dropbox link password
        :type password: str
        :param checksum_check: Select whether to check the checksum of
            the file downloaded, defaults to True.
        :type checksum_check: bool, optional
        :returns: True for successful download, False if it goes wrong.
        :rtype: bool
        '''
        self.logger.info("Downloading {}...".format(dropbox_path))
        try:
            self.dbx.sharing_get_shared_link_file_to_file(
                download_path, shared_url, path=dropbox_path,
                link_password=password)
            download_complete = True
        except Exception as e:
            self.logger.warning("{} encountered a problem!".format(
                dropbox_path
            ))
            download_complete = False
            return False

        if download_complete and checksum_check:
            success = self._checksum_check(dropbox_path, download_path)
            if not success:
                self.logger.warning(
                    "md5 checksum does"
                    " not match for {}!".format(dropbox_path))
                self.logger.warning("Will try again after main cycle.")
                return False
            else:
                self.logger.info("Integrity check passed for {}".format(
                    dropbox_path
                ))

        return True


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
        self.logger.debug(FIELD_FILES[epoch])

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
        self.logger.debug(src_coord)
        nearest_beams, seps, _d3d = src_coord.match_to_catalog_sky(
            self.direction)
        self.logger.debug(seps.deg)
        self.logger.debug("Nearest beams: {}".format(self.fields["BEAM"][nearest_beams]))
        within_beam = seps.deg < max_sep
        catalog["sbid"] = self.fields["SBID"].iloc[nearest_beams].values
        nearest_fields = self.fields["FIELD_NAME"].iloc[nearest_beams]
        self.logger.debug(nearest_fields)
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
    :param IMAGE_FOLDER: Path to image directory
    :type IMAGE_FOLDER: str
    :param RMS_FOLDER: Path to RMS map directory
    :type RMS_FOLDER: str
    :param vast_pilot: Survey epoch (if applicable)
    :type vast_pilot: str
    :param tiles: Use image tiles instead of mosaics, defaults to `False`
    :type tiles: bool, optional
    '''

    def __init__(self, sbid, field, IMAGE_FOLDER,
                 RMS_FOLDER, vast_pilot, tiles=False):
        '''Constructor method
        '''

        self.logger = logging.getLogger('vasttools.survey.Image')
        self.logger.debug('Created Image instance')

        self.sbid = sbid
        self.field = field
        self.RMS_FOLDER = RMS_FOLDER
        self.vast_pilot = vast_pilot

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

            try:
                self.beam = Beam.from_fits_header(self.header)
            except Exception as e:
                self.logger.error("Beam information could not be read!")
                self.beam = None

    def get_rms_img(self):
        '''
        Load the noisemap corresponding to the image
        '''
        if self.vast_pilot == "0":
            self.rmsname = self.imgname.replace(
                '.fits', '.taylor.0.noise.fits')
        else:
            self.rmsname = self.imgname.replace('.fits', '_rms.fits')

        self.rmspath = os.path.join(self.RMS_FOLDER, self.rmsname)

        if os.path.isfile(self.rmspath):
            self.rms_fail = False
        else:
            self.rms_fail = True
            self.logger.error(
                "{} does not exist! Unable to get noise maps.".format(
                    self.rmspath))
            return

        with fits.open(self.rmspath) as hdul:
            self.rms_header = hdul[0].header
            self.rms_wcs = WCS(self.rms_header, naxis=2)

            try:
                self.rms_data = hdul[0].data[0, 0, :, :]
            except Exception as e:
                self.rms_data = hdul[0].data
