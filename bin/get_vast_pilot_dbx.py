#!/usr/bin/env python

import argparse
import dropbox
import os
import sys
import datetime
import configparser
import numpy as np
import pandas as pd
import re

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
        stokesQ_only=False,
        stokesU_only=False,
        no_stokesI=False,
        no_stokesV=False,
        no_stokesQ=False,
        no_stokesU=False,
        skip_xml=False,
        skip_qc=False,
        skip_islands=False,
        skip_field_images=False,
        skip_bkg_images=False,
        skip_rms_images=False,
        skip_all_images=False,
        combined_only=False,
        tile_only=False,
        selected_epochs=None):
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
    :param stokesQ_only: Stokes Q only boolean, defaults to False
    :type stokesQ_only: bool, optional
    :param stokesU_only: Stokes U only boolean, defaults to False
    :type stokesU_only: bool, optional
    :param no_stokesI: No Stokes I boolean, defaults to False
    :type no_stokesI: bool, optional
    :param no_stokesV: No Stokes V boolean, defaults to False
    :type no_stokesV: bool, optional
    :param no_stokesQ: No Stokes Q boolean, defaults to False
    :type no_stokesQ: bool, optional
    :param no_stokesU: No Stokes U boolean, defaults to False
    :type no_stokesU: bool, optional
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
    :param selected_epochs: Filter to only the epoch selected,
        defaults to None
    :type selected_epochs: str, optional
    :returns: filtered list of dropbox files
    :rtype: list
    '''
    filter_df = pd.DataFrame(data=files_list, columns=["file"])

    stokes_only_sum = sum(
        [stokesI_only, stokesV_only, stokesQ_only, stokesU_only]
    )

    if stokes_only_sum > 1:
        logger.critical(
            "Multiple 'Stokes only' arguments have been set:"
            "\nStokes I only: %s"
            "\nStokes V only: %s"
            "\nStokes Q only: %s"
            "\nStokes U only: %s",
            stokesI_only,
            stokesV_only,
            stokesQ_only,
            stokesU_only
        )
        logger.critical("Please check your settings and run again!")
        sys.exit()
    elif stokes_only_sum == 1:
        check_nos = False
    else:
        check_nos = True

    if combined_only is True and tile_only is True:
        logger.warning(
            "Combined only and tiles only are both set to True.")
        logger.warning("Ignoring.")
        combined_only = False
        tiles_only = False

    if fields is not None:
        fields = [re.escape(i) for i in fields]
        field_pattern = "|".join(fields)
        logger.debug("Filtering fields for %s", field_pattern)
        filter_df = filter_df[filter_df.file.str.contains(field_pattern)]
        filter_df.reset_index(drop=True, inplace=True)

    if check_nos:
        if no_stokesI:
            logger.debug("Filtering out Stokes I products")
            filter_df = filter_df[~filter_df.file.str.contains("STOKESI")]
            filter_df.reset_index(drop=True, inplace=True)

        if no_stokesV:
            logger.debug("Filtering out Stokes V products")
            filter_df = filter_df[~filter_df.file.str.contains("STOKESV")]
            filter_df.reset_index(drop=True, inplace=True)

        if no_stokesQ:
            logger.debug("Filtering out Stokes Q products")
            filter_df = filter_df[~filter_df.file.str.contains("STOKESQ")]
            filter_df.reset_index(drop=True, inplace=True)

        if no_stokesU:
            logger.debug("Filtering out Stokes U products")
            filter_df = filter_df[~filter_df.file.str.contains("STOKESU")]
            filter_df.reset_index(drop=True, inplace=True)
    else:
        if stokesI_only:
            logger.debug("Filtering to Stokes I only")
            filter_df = filter_df[filter_df.file.str.contains("STOKESI")]
            filter_df.reset_index(drop=True, inplace=True)

        if stokesV_only:
            logger.debug("Filtering to Stokes V only")
            filter_df = filter_df[filter_df.file.str.contains("STOKESV")]
            filter_df.reset_index(drop=True, inplace=True)

        if stokesQ_only:
            logger.debug("Filtering to Stokes Q only")
            filter_df = filter_df[filter_df.file.str.contains("STOKESQ")]
            filter_df.reset_index(drop=True, inplace=True)

        if stokesU_only:
            logger.debug("Filtering to Stokes U only")
            filter_df = filter_df[filter_df.file.str.contains("STOKESU")]
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

    if selected_epochs is not None:
        logger.debug("Filtering epochs.")
        pattern_strings = []
        for i in selected_epochs.split(","):
            if i.startswith("0"):
                i = i[1:]
            if i not in RELEASED_EPOCHS:
                logger.warning(
                    "Epoch '{}' is unknown or not released yet!"
                    " No files will be found for this selection."
                )
            else:
                epoch_dbx_format = "/EPOCH{}/".format(
                    RELEASED_EPOCHS[i])
                pattern_strings.append(epoch_dbx_format)
        pattern = "|".join(pattern_strings)
        logger.debug("Filtering to %s only.", pattern
        )
        filter_df = filter_df[filter_df.file.str.contains(
            pattern)]
        filter_df.reset_index(drop=True, inplace=True)

    final_list = filter_df.file.tolist()

    return final_list


user_friendly_epochs = {v: k for k, v in RELEASED_EPOCHS.items()}
user_friendly_epochs = [
    user_friendly_epochs[i] for i in sorted(user_friendly_epochs.keys())
]

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

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
    type=str,
    help=(
        "Input an already generated list of available files for the script to"
        " use. If required, this file can be generated by the"
        " '--get-available-files' option."
    ),
    default=None)

parser.add_argument(
    '--get-available-files',
    action="store_true",
    help=(
        'Generate the list of available files on the shared folder.'
        ' The list will be saved to a file.'
    ))

parser.add_argument(
    '--download',
    action="store_true",
    help=('Download data according to the filter options entered'),
    default=None)

parser.add_argument(
    '--find-fields-input',
    type=str,
    help="Input of fields to fetch (can be obtained from 'find_sources.py').",
    default=None)

parser.add_argument(
    '--user-files-list',
    type=str,
    help='Input of files to fetch.',
    default=None)

parser.add_argument(
    '--only-epochs',
    type=str,
    help=("Only download files from the selected epochs."
          " Enter as a list with no spaces, e.g. '1,2,4x'."
          " If nothing is entered then all epochs are fetched."
          " The current epochs are: {}.".format(
              ", ".join(user_friendly_epochs)
          )),
    default=None)

parser.add_argument(
    '--only-fields',
    type=str,
    help=("Only download files from the selected fields."
          " Enter as a list with no spaces,"
          " e.g. 'VAST_0012+00A,VAST_0012-06A'."
          " If nothing is entered then all fields are fetched."
    ),
    default=None)

parser.add_argument(
    '--stokesI-only',
    action="store_true",
    help="Only download STOKES I products.")

parser.add_argument(
    '--stokesV-only',
    action="store_true",
    help="Only download STOKES V products.")

parser.add_argument(
    '--stokesQ-only',
    action="store_true",
    help="Only download STOKES Q products.")

parser.add_argument(
    '--stokesU-only',
    action="store_true",
    help="Only download STOKES U products.")

parser.add_argument(
    '--no-stokesI',
    action="store_true",
    help="Do not download Stokes I products.")

parser.add_argument(
    '--no-stokesV',
    action="store_true",
    help="Do not download Stokes V products.")

parser.add_argument(
    '--no-stokesQ',
    action="store_true",
    help="Do not download Stokes Q products.")

parser.add_argument(
    '--no-stokesU',
    action="store_true",
    help="Do not download Stokes U products.")

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

parser.add_argument(
    '--overwrite',
    action="store_true",
    help=(
        "Overwrite any files that already exist in the output directory."
        " If overwrite is not selected, integrity checking will still be"
        " performed on the existing files and if the check fails, the file"
        " will be re-downloaded."))

parser.add_argument(
    '--debug',
    action="store_true",
    help='Set logging level to debug.')

parser.add_argument(
    '--write-template-dropbox-config',
    action="store_true",
    help='Create a template dropbox config file.')

parser.add_argument(
    '--include-legacy',
    action="store_true",
    help=(
        "Include the 'LEGACY' directory when searching through files. "
        "Only valid when using the '--get-available-files' option."
    )
)

parser.add_argument(
    '--max-retries',
    type=int,
    help='How many times to attempt to retry a failed download',
    default=2)


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
        "Writen an example dropbox config file to '%s'.", config_file)
    sys.exit()

if not os.path.isfile(args.dropbox_config):
    logger.critical(
        "Cannot find dropbox config file '%s'!", args.dropbox_config
    )
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

logger.debug("Shared URL: %s", shared_url)
logger.debug("Password: %s", password != "")

output_dir = args.output

# check dir
if not args.available_epochs and not args.get_available_files:
    if os.path.isdir(output_dir):
        logger.warning(
            "Output directory '%s' already exists!", output_dir
        )
        logger.warning("Files may get overwritten!")
    else:
        os.mkdir(output_dir)


dbx = dropbox.Dropbox(access_token)

shared_link = dropbox.files.SharedLink(url=shared_url, password=password)

base_file_list = dbx.files_list_folder("", shared_link=shared_link)

vast_dropbox = Dropbox(dbx, shared_link)

if args.available_files is not None:
    if not os.path.isfile(args.available_files):
        logger.error(
            "Available files list file %s "
            "not found!", args.available_files
        )
        logger.error(
            "Check file or remove the"
            " '--available-files' option.")
        sys.exit()
    with open(args.available_files, 'r') as f:
        lines = f.readlines()
    files_list = [i.strip() for i in lines]
else:
    files_list = None

if args.available_epochs:
    logger.info("The following epochs are available:")
    for i in base_file_list.entries:
        if isinstance(i, dropbox.files.FolderMetadata) and "EPOCH" in i.name:
            logger.info(i.name)

elif args.get_available_files:
    logger.info(
        "Gathering a list of files - this will take "
        "approximately 4 minutes per epoch."
    )

    files_list, folders_list = vast_dropbox.recursive_build_files(
        base_file_list,
        legacy=args.include_legacy)
    logger.info("Found %s files.", len(files_list))
    vast_list_file_name = "vast_dbx_file_list_{}.txt".format(now_str)

    with open(vast_list_file_name, "w") as f:
        f.write("# File list on VAST Pilot survey dropbox as of {}\n".format(
            now))
        [f.write(i + "\n") for i in files_list]

    logger.info("All available files written to %s", vast_list_file_name)

elif args.download:
    if files_list is None:
        logger.info(
            "No list of available files provided, will generate."
        )
        logger.info(
            "Gathering a list of files - this will take "
            "approximately 4 minutes per epoch."
        )

        files_list, folders_list = vast_dropbox.recursive_build_files(
            base_file_list,
            legacy=args.include_legacy)

    if args.find_fields_input is not None:
        if not os.path.isfile(args.find_fields_input):
            logger.error(
                "Supplied file '%s' not found!", args.find_fields_input
            )
            sys.exit()
        fields_df = pd.read_csv(args.find_fields_input, comment="#")
        fields_to_fetch = fields_df.field_name.unique()
        logger.info("Will download data products of the following fields:")
        for f in fields_to_fetch:
            logger.info(f)

        # files_to_download = filter_files_list(
        #     files_list,
        #     fields=fields_to_fetch,
        #     stokesI_only=args.stokesI_only,
        #     stokesV_only=args.stokesV_only,
        #     stokesQ_only=args.stokesQ_only,
        #     stokesU_only=args.stokesU_only,
        #     no_stokesI=args.no_stokesI,
        #     no_stokesV=args.no_stokesV,
        #     no_stokesQ=args.no_stokesQ,
        #     no_stokesU=args.no_stokesU,
        #     skip_xml=args.skip_xml,
        #     skip_qc=args.skip_qc,
        #     skip_islands=args.skip_islands,
        #     skip_field_images=args.skip_field_images,
        #     skip_bkg_images=args.skip_bkg_images,
        #     skip_rms_images=args.skip_rms_images,
        #     skip_all_images=args.skip_all_images,
        #     combined_only=args.combined_only,
        #     tile_only=args.tile_only,
        #     selected_epochs=args.only_epochs
        # )

    elif args.user_files_list is not None:
        if not os.path.isfile(args.user_files_list):
            logger.error("Supplied file '%s' not found!", args.user_files_list)
            sys.exit()
        with open(args.user_files_list, 'r') as f:
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

    else:
        if args.only_fields is not None:
            fields_to_fetch = args.only_fields.split(",")
        else:
            fields_to_fetch = args.only_fields

    files_to_download = filter_files_list(
        files_list,
        fields=fields_to_fetch,
        stokesI_only=args.stokesI_only,
        stokesV_only=args.stokesV_only,
        stokesQ_only=args.stokesQ_only,
        stokesU_only=args.stokesU_only,
        no_stokesI=args.no_stokesI,
        no_stokesV=args.no_stokesV,
        no_stokesQ=args.no_stokesQ,
        no_stokesU=args.no_stokesU,
        skip_xml=args.skip_xml,
        skip_qc=args.skip_qc,
        skip_islands=args.skip_islands,
        skip_field_images=args.skip_field_images,
        skip_bkg_images=args.skip_bkg_images,
        skip_rms_images=args.skip_rms_images,
        skip_all_images=args.skip_all_images,
        combined_only=args.combined_only,
        tile_only=args.tile_only,
        selected_epochs=args.only_epochs
    )

    dirs_to_create = np.unique(
        ["/".join(i.strip().split("/")[1:-1]) for i in files_to_download])

    for i in dirs_to_create:
        if i == "":
            continue
        os.makedirs(os.path.join(output_dir, i), exist_ok=True)

    logger.info(
        "Downloading %s files from '%s'...",
        len(files_to_download),
        args.user_files_list
    )
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

logger.info("Log file written to %s.", logfilename)

logger.info("All done!")
