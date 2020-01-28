#!/usr/bin/env python

import argparse
import dropbox
import os
import sys
import datetime
import configparser
import numpy as np
import pandas as pd

from vasttools.survey import RELEASED_EPOCHS
from vasttools.survey import Dropbox

import logging
import logging.handlers
import logging.config

try:
    import colorlog
    use_colorlog = True
except ImportError:
    use_colorlog = False


def filter_files_list(
        files_list,
        fields=None,
        stokesI_only=False,
        stokesV_only=False,
        skip_xml=False,
        skip_qc=False,
        skip_islands=False,
        skip_field_images=False,
        skip_bkg_images=False,
        skip_rms_images=False,
        skip_all_images=False,
        combined_only=False,
        tile_only=False):
    '''
    Filters the file_list to fetch by the users request.

    :param file_list: the list of dropbox files to filter
    :type file_list: list
    :param fields: list of fields to filter, defaults to None.
    :type fields: list, optional
    :param stokesI_only: Stokes I only boolean, defaults to False
    :type stokesI_only: bool, optional
    :param stokesV_only: Stokes V only boolean, defaults to False
    :type stokesV_only: bool, optional
    :param skip_xml: Filter out .xml files, defaults to False
    :type skip_xml: bool, optional
    :param skip_qc: Filter out QC files, defaults to False
    :type skip_qc: bool, optional
    :param skip_islands: Filter out island selavy
        files, defaults to False
    :type skip_islands: bool, optional
    :param skip_field_images: Filter out field fits
        files, defaults to False
    :type skip_field_images: bool, optional
    :param skip_bkg_images: Filter out bkg fits files, defaults
        to False
    :type skip_bkg_images: bool, optional
    :param skip_rms_images: Filter out rms fits files, defaults
        to False
    :type skip_rms_images: bool, optional
    :param skip_all_images: Filter out .fits files, defaults
        to False
    :type skip_all_images: bool, optional
    :param combined_only: Filter to only combined products,
        defaults to False
    :type combined_only: bool, optional
    :param tile_only: Filter to only tiles products, defaults
        to False
    :type tile_only: bool, optional
    :returns: filtered list of dropbox files
    :rtype: list
    '''
    if fields is None:
        fields = []
    filter_df = pd.DataFrame(data=files_list, columns=["file"])

    if stokesI_only is True and stokesV_only is True:
        logger.warning(
            "Stokes I only and Stokes V only are both set to True.")
        logger.warning("Ignoring.")
        stokesI_only = False
        stokesV_only = False

    if combined_only is True and tile_only is True:
        logger.warning(
            "Combined only and tiles only are both set to True.")
        logger.warning("Ignoring.")
        combined_only = False
        tiles_only = False

    if len(fields) > 0:
        field_pattern = "|".join(fields)
        logger.debug("Filtering fields for {}".format(field_pattern))
        filter_df = filter_df[filter_df.file.str.contains(field_pattern)]
        filter_df.reset_index(drop=True, inplace=True)

    if stokesI_only:
        logger.debug("Filtering to Stokes I only")
        filter_df = filter_df[filter_df.file.str.contains("STOKESI")]
        filter_df.reset_index(drop=True, inplace=True)

    if stokesV_only:
        logger.debug("Filtering to Stokes V only")
        filter_df = filter_df[filter_df.file.str.contains("STOKESV")]
        filter_df.reset_index(drop=True, inplace=True)

    if skip_xml:
        logger.debug("Filtering out XML files.")
        filter_df = filter_df[~filter_df.file.str.endswith(".xml")]
        filter_df.reset_index(drop=True, inplace=True)

    if skip_qc:
        logger.debug("Filtering out QC files.")
        filter_df = filter_df[~filter_df.file.str.contains("/QC")]
        filter_df.reset_index(drop=True, inplace=True)

    if skip_islands:
        logger.debug("Filtering out island files.")
        filter_df = filter_df[~filter_df.file.str.contains(".islands.")]
        filter_df.reset_index(drop=True, inplace=True)

    if not skip_all_images:
        if skip_field_images:
            logger.debug("Filtering out field fits files.")
            pattern = ".I.fits|.V.fits"
            filter_df = filter_df[~filter_df.file.str.contains(pattern)]
            filter_df.reset_index(drop=True, inplace=True)

        if skip_bkg_images:
            logger.debug("Filtering out rms fits files.")
            filter_df = filter_df[~filter_df.file.str.endswith("_bkg.fits")]
            filter_df.reset_index(drop=True, inplace=True)

        if skip_rms_images:
            logger.debug("Filtering out rms fits files.")
            filter_df = filter_df[~filter_df.file.str.endswith("_rms.fits")]
            filter_df.reset_index(drop=True, inplace=True)

    else:
        logger.debug("Filtering out fits files.")
        filter_df = filter_df[~filter_df.file.str.endswith(".fits")]
        filter_df.reset_index(drop=True, inplace=True)

    if combined_only:
        logger.debug("Filtering to combined files only.")
        filter_df = filter_df[filter_df.file.str.contains("/COMBINED/")]
        filter_df.reset_index(drop=True, inplace=True)

    if tile_only:
        logger.debug("Filtering to tiles files only.")
        filter_df = filter_df[filter_df.file.str.contains("/TILES/")]
        filter_df.reset_index(drop=True, inplace=True)

    final_list = filter_df.file.tolist()

    return final_list


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    '--output',
    type=str,
    help='Name of the local output directory where files will be saved',
    default="vast_dropbox")

parser.add_argument(
    '--available-epochs',
    action="store_true",
    help='Print out what Epochs are available.')

parser.add_argument(
    '--available-files',
    action="store_true",
    help='Print out a list of available files on the shared folder.')

parser.add_argument(
    '--download-epoch',
    type=str,
    choices=sorted(RELEASED_EPOCHS),
    help=('Select to download an entire Epoch directory. '
          'Enter as shown in choices.'),
    default=None)

parser.add_argument(
    '--find-fields-input',
    type=str,
    help="Input of fields to fetch (can be obtained from 'find_sources.py').",
    default=None)

parser.add_argument(
    '--find-fields-available-files-input',
    type=str,
    help=(
        "Input already generated list of available files for the download"
        " fields function to save the script gathering all the files"
        " available again. I.e. the output of the '--available-files"
        " option. If not given the script will get the list of files"
        " from Dropbox."),
    default=None)

parser.add_argument(
    '--files-list',
    type=str,
    help='Input of files to fetch.',
    default=None)

parser.add_argument(
    '--overwrite',
    action="store_true",
    help='Overwrite any files that already exist in the output directory.')

parser.add_argument(
    '--debug',
    action="store_true",
    help='Set logging level to debug.')

parser.add_argument(
    '--dropbox-config',
    type=str,
    help=(
        "Dropbox config file to be read in containing the shared url, "
        "password and access token. A template can be generated using "
        "'--write-template-dropbox-config'."
    ),
    default="dropbox.cfg")

parser.add_argument(
    '--write-template-dropbox-config',
    action="store_true",
    help='Create a template dropbox config file.')

parser.add_argument(
    '--include-legacy',
    action="store_true",
    help=(
        "Include the 'LEGACY' directory when searching through files. "
        "Only valid when using the '--available-files' option."
    )
)

parser.add_argument(
    '--max-retries',
    type=int,
    help='How many times to attempt to retry a failed download',
    default=2)

parser.add_argument(
    '--stokesI-only',
    action="store_true",
    help="Only download STOKES I products.")

parser.add_argument(
    '--stokesV-only',
    action="store_true",
    help="Only download STOKES V products.")

parser.add_argument(
    '--skip-xml',
    action="store_true",
    help="Do not download XML files.")

parser.add_argument(
    '--skip-qc',
    action="store_true",
    help="Do not download the QC plots.")

parser.add_argument(
    '--skip-islands',
    action="store_true",
    help="Only download component selavy files.")

parser.add_argument(
    '--skip-field-images',
    action="store_true",
    help="Do not download field images.")

parser.add_argument(
    '--skip-bkg-images',
    action="store_true",
    help="Do not download background images.")

parser.add_argument(
    '--skip-rms-images',
    action="store_true",
    help="Do not download background images.")

parser.add_argument(
    '--skip-all-images',
    action="store_true",
    help="Only download non-image data products.")

parser.add_argument(
    '--combined-only',
    action="store_true",
    help="Only download the combined products.")

parser.add_argument(
    '--tile-only',
    action="store_true",
    help="Only download the combined products.")

args = parser.parse_args()

now = datetime.datetime.now()
now_str = now.strftime("%Y%m%d_%H:%M:%S")

logger = logging.getLogger()
s = logging.StreamHandler()
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
            'CRITICAL': 'red,bg_white',
        },
        secondary_log_colors={},
        style='%'
    )
else:
    formatter = logging.Formatter(logformat, datefmt="%Y-%m-%d %H:%M:%S")

s.setFormatter(formatter)
logger.addHandler(s)

logfilename = "get_vast_pilot_dbx_{}.log".format(now_str)
fileHandler = logging.FileHandler(logfilename)
fileHandler.setFormatter(
    logging.Formatter(
        logformat,
        datefmt="%Y-%m-%d %H:%M:%S"))
logger.addHandler(fileHandler)

if args.debug:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

logging.getLogger("dropbox").setLevel(logging.WARNING)

if args.write_template_dropbox_config:
    config_file = "dropbox.cfg"
    with open(config_file, "w") as f:
        f.write("[dropbox]\n")
        f.write("shared_url = ENTER_URL\n")
        f.write("password = ENTER_PASSWORD\n")
        f.write("access_token = ENTER_ACCESS_TOKEN\n")

    logger.info(
        "Writen an example dropbox config file to '{}'.".format(config_file))
    sys.exit()

if not os.path.isfile(args.dropbox_config):
    logger.critical(
        "Cannot find dropbox config file '{}!".format(
            args.dropbox_config))
    logger.info(
        "A template dropbox file can be generated using "
        "python get_vast_pilot_dbx.py '--write-template-dropbox-config'"
    )

    sys.exit()

config = configparser.ConfigParser()
config.read(args.dropbox_config)

shared_url = config["dropbox"]["shared_url"]
password = config["dropbox"]["password"]
access_token = config["dropbox"]["access_token"]

logger.debug("Shared URL: {}".format(shared_url))
logger.debug("Password: {}".format(password != ""))

output_dir = args.output

# check dir
if not args.available_epochs and not args.available_files:
    if os.path.isdir(output_dir):
        logger.warning(
            "Output directory '{}' already exists!".format(output_dir))
        logger.warning("Files may get overwritten!")
    else:
        os.mkdir(output_dir)


dbx = dropbox.Dropbox(access_token)

shared_link = dropbox.files.SharedLink(url=shared_url, password=password)

base_file_list = dbx.files_list_folder("", shared_link=shared_link)

vast_dropbox = Dropbox(dbx, shared_link)

if args.available_epochs:
    logger.info("The following epochs are available:")
    for i in base_file_list.entries:
        if isinstance(i, dropbox.files.FolderMetadata) and "EPOCH" in i.name:
            logger.info(i.name)

elif args.available_files:
    logger.info(
        "Gathering a list of files - this will take "
        "approximately 4 minutes per epoch."
    )

    files_list, folders_list = vast_dropbox.recursive_build_files(
        base_file_list,
        legacy=args.include_legacy)
    logger.info("Found {} files.".format(len(files_list)))
    vast_list_file_name = "vast_dbx_file_list_{}.txt".format(now_str)

    with open(vast_list_file_name, "w") as f:
        f.write("# File list on VAST Pilot survey dropbox as of {}\n".format(
            now))
        [f.write(i + "\n") for i in files_list]

    logger.info("All available files written to {}".format(
        vast_list_file_name))

elif args.download_epoch is not None:
    epochs = []
    for i in base_file_list.entries:
        if isinstance(i, dropbox.files.FolderMetadata) and "EPOCH" in i.name:
            epochs.append(int(i.name.split('EPOCH')[-1]))
    dropbox_name = RELEASED_EPOCHS[args.download_epoch]
    if dropbox_name not in epochs:
        logger.error(
            "EPOCH{} has not yet been released!".format(
                dropbox_name))
        sys.exit()
    else:
        epoch_string = "EPOCH{}".format(dropbox_name)
        epoch_file_list = dbx.files_list_folder(
            "/{}".format(epoch_string), shared_link=shared_link)
        logger.info(
            "Gathering {} files to download...".format(epoch_string))
        files_list, folders_list = vast_dropbox.recursive_build_files(
            epoch_file_list, dbx, preappend=epoch_string)
        logger.info("{} files to download".format(len(files_list)))

        for folder in folders_list:
            os.makedirs(os.path.join(output_dir, folder[1:]), exist_ok=True)

        files_to_download = filter_files_list(
            files_list,
            stokesI_only=args.stokesI_only,
            stokesV_only=args.stokesV_only,
            skip_xml=args.skip_xml,
            skip_qc=args.skip_qc,
            skip_islands=args.skip_islands,
            skip_field_images=args.skip_field_images,
            skip_bkg_images=args.skip_bkg_images,
            skip_rms_images=args.skip_rms_images,
            skip_all_images=args.skip_all_images,
            combined_only=args.combined_only,
            tile_only=args.tile_only
        )

        logger.info(
            "Downloading {} files for {}...".format(
                len(files_to_download), epoch_string))
        complete_failures = vast_dropbox.download_files(
            files_to_download,
            output_dir,
            shared_url,
            password,
            args.max_retries,
            args.overwrite)
        if len(complete_failures) > 0:
            logger.warning("The following files failed to download correctly:")
            for fail in complete_failures:
                logger.warning(fail)
            logger.warning("These files may be corrupted!")

elif args.find_fields_input is not None:
    if not os.path.isfile(args.find_fields_input):
        logger.error(
            "Supplied file '{}' not found!".format(args.find_fields_input))
        sys.exit()
    fields_df = pd.read_csv(args.find_fields_input)
    fields_to_fetch = fields_df.field_name.unique()
    logger.info("Will download data products of the following fields:")
    for f in fields_to_fetch:
        logger.info(f)
    if args.find_fields_available_files_input is not None:
        if not os.path.isfile(args.find_fields_available_files_input):
            logger.error(
                "Available files list file {} "
                "not found!".format(args.find_fields_available_files_input))
            logger.error(
                "Check file or remove the"
                " '--find-fields-available-files-list' option.")
            sys.exit()
        with open(args.find_fields_available_files_input, 'r') as f:
            lines = f.readlines()
        files_list = [i.strip() for i in lines]
    else:
        # We need to search all files so we need to make the main search
        files_list, folders_list = vast_dropbox.recursive_build_files(
            base_file_list,
            legacy=args.include_legacy)
    files_to_download = filter_files_list(
        files_list,
        fields=fields_to_fetch,
        stokesI_only=args.stokesI_only,
        stokesV_only=args.stokesV_only,
        skip_xml=args.skip_xml,
        skip_qc=args.skip_qc,
        skip_islands=args.skip_islands,
        skip_field_images=args.skip_field_images,
        skip_bkg_images=args.skip_bkg_images,
        skip_rms_images=args.skip_rms_images,
        skip_all_images=args.skip_all_images,
        combined_only=args.combined_only,
        tile_only=args.tile_only
    )

    dirs_to_create = np.unique(
        ["/".join(i.strip().split("/")[1:-1]) for i in files_to_download])

    for i in dirs_to_create:
        if i == "":
            continue
        os.makedirs(os.path.join(output_dir, i), exist_ok=True)

    logger.info(
        "Downloading {} files for {} fields...".format(
            len(files_to_download), len(fields_to_fetch)))
    complete_failures = vast_dropbox.download_files(
        files_to_download,
        output_dir,
        shared_url,
        password,
        args.max_retries,
        args.overwrite)
    if len(complete_failures) > 0:
        logger.warning("The following files failed to download correctly:")
        for fail in complete_failures:
            logger.warning(fail)
        logger.warning("These files may be corrupted!")

elif args.files_list is not None:
    if not os.path.isfile(args.files_list):
        logger.error("Supplied file '{}' not found!".format(args.files_list))
        sys.exit()
    with open(args.files_list, 'r') as f:
        userlines = f.readlines()

    # check files start with / and ignore #
    files_list = []

    for i in userlines:
        if i.startswith("#"):
            continue
        else:
            if i.startswith("/"):
                files_list.append(i.strip())
            else:
                files_list.append("/{}".format(i.strip()))

    files_to_download = filter_files_list(
        files_list,
        stokesI_only=args.stokesI_only,
        stokesV_only=args.stokesV_only,
        skip_xml=args.skip_xml,
        skip_qc=args.skip_qc,
        skip_islands=args.skip_islands,
        skip_field_images=args.skip_field_images,
        skip_bkg_images=args.skip_bkg_images,
        skip_rms_images=args.skip_rms_images,
        skip_all_images=args.skip_all_images,
        combined_only=args.combined_only,
        tile_only=args.tile_only
    )

    dirs_to_create = np.unique(
        ["/".join(i.strip().split("/")[1:-1]) for i in files_to_download])

    for i in dirs_to_create:
        if i == "":
            continue
        os.makedirs(os.path.join(output_dir, i), exist_ok=True)

    logger.info(
        "Downloading {} files from '{}'...".format(
            len(files_to_download),
            args.files_list))
    complete_failures = vast_dropbox.download_files(
        files_to_download,
        output_dir,
        shared_url,
        password,
        args.max_retries,
        args.overwrite)
    if len(complete_failures) > 0:
        logger.warning("The following files failed to download correctly:")
        for fail in complete_failures:
            logger.warning(fail)
        logger.warning("These files may be corrupted!")

else:
    logger.info("Nothing to be done!")

end = datetime.datetime.now()

runtime = end - now

logger.info("Ran for {:.1f} minutes.".format(runtime.seconds / 60.))

logger.info("Log file written to {}".format(logfilename))

logger.info("All done!")
