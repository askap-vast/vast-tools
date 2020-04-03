#!/usr/bin/env python

import argparse
import dropbox
import os
import sys
import datetime
import configparser
import numpy as np
import pandas as pd
import numexpr
import functools
from multiprocessing import Pool
import signal

from vasttools.survey import RELEASED_EPOCHS
from vasttools.survey import DROPBOX_FILE
from vasttools.survey import Dropbox
from vasttools.utils import get_logger
from vasttools.utils import check_file

import logging
import logging.handlers
import logging.config

try:
    import colorlog
    use_colorlog = True
except ImportError:
    use_colorlog = False

runstart = datetime.datetime.now()


def write_config_template():
    config_file = "dropbox.cfg"
    with open(config_file, "w") as f:
        f.write("[dropbox]\n")
        f.write("shared_url = ENTER_URL\n")
        f.write("password = ENTER_PASSWORD\n")
        f.write("access_token = ENTER_ACCESS_TOKEN\n")

    logger.info(
        "Writen an example dropbox config file to '%s'.", config_file
    )


def read_dbx_config(config_file):
    logger.debug("Reading config file.")
    config = configparser.ConfigParser()
    config.read(config_file)

    shared_url = config["dropbox"]["shared_url"]
    password = config["dropbox"]["password"]
    access_token = config["dropbox"]["access_token"]

    logger.debug("Shared URL: %s", shared_url)
    logger.debug("Password: %s", password != "")

    dbx_config = {
        "shared_url": shared_url,
        "password": password,
        "access_token": access_token
    }

    return dbx_config


def setup_outdir(output, args):
    # check dir
    if not args.available_epochs and not args.get_available_files:
        if os.path.isdir(output_dir):
            logger.warning(
                "Output directory '%s' already exists!", output_dir
            )
            logger.warning("Files may get overwritten!")
        else:
            if args.dry_run:
                logger.info(
                    "Dry run selected: will not create"
                    " output directory."
                )
            else:
                os.mkdir(output_dir)


def setup_dropbox(dbx_config):
    dbx = dropbox.Dropbox(dbx_config["access_token"])
    shared_link = dropbox.files.SharedLink(
        url=dbx_config["shared_url"],
        password=dbx_config["password"]
    )
    base_file_list = dbx.files_list_folder(
        "", shared_link=shared_link
    )
    vast_dropbox = Dropbox(dbx, shared_link)

    return vast_dropbox, base_file_list


def run_get_dropbox(
        vast_dropbox,
        base_file_list,
        dbx_config,
        output_dir,
        args):
    if args.available_files is not None:
        if not os.path.isfile(args.available_files):
            logger.error(
                "Available files list file %s "
                "not found!", args.available_files
            )
            logger.error(
                "Check file or remove the"
                " '--available-files' option.")
            return
    else:
        args.available_files = DROPBOX_FILE

    with open(args.available_files, 'r') as f:
        lines = f.readlines()
    files_list = [i.strip() for i in lines]

    if args.available_epochs:
        logger.info("The following epochs are available:")
        for i in base_file_list.entries:
            if isinstance(
                i, dropbox.files.FolderMetadata
            ) and "EPOCH" in i.name:
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
            f.write(
                "# File list on VAST Pilot survey dropbox as of {}\n".format(
                    now
                )
            )
            [f.write(i + "\n") for i in files_list]

        logger.info("All available files written to %s", vast_list_file_name)

        return

    elif args.download:

        fields_to_fetch = None

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

        elif args.user_files_list is not None:
            if not os.path.isfile(args.user_files_list):
                logger.error(
                    "Supplied file '%s' not found!", args.user_files_list
                )
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

        if fields_to_fetch is None and args.only_fields is not None:
                fields_to_fetch = args.only_fields.split(",")

        files_to_download = vast_dropbox.filter_files_list(
            files_list,
            fields=fields_to_fetch,
            stokes=args.stokes,
            skip_xml=args.skip_xml,
            skip_txt=args.skip_txt,
            skip_qc=args.skip_qc,
            skip_components=args.skip_components,
            skip_islands=args.skip_islands,
            skip_field_images=args.skip_field_images,
            skip_bkg_images=args.skip_bkg_images,
            skip_rms_images=args.skip_rms_images,
            skip_all_images=args.skip_all_images,
            combined_only=args.combined_only,
            tile_only=args.tile_only,
            selected_epochs=args.only_epochs,
            legacy_download=args.legacy_download
        )

        if len(files_to_download) != 0:
            dirs_to_create = np.unique(
                [
                    "/".join(i.strip().split("/")[1:-1])
                    for i in files_to_download
                ]
            )

            if not args.dry_run:
                for i in dirs_to_create:
                    if i == "":
                        continue
                    os.makedirs(os.path.join(output_dir, i), exist_ok=True)

                logger.info(
                    "Downloading %s files...",
                    len(files_to_download)
                )
                if args.download_threads > 1:
                    logger.info(
                        "Will use %s workers to download data.",
                        args.download_threads
                    )
                    logger.warning(
                        'There will be no logging apart from'
                        ' warning level messages printed to the terminal.'
                    )
                    files_to_download = [[j] for j in files_to_download]
                    multi_download = functools.partial(
                        vast_dropbox.download_files,
                        output_dir=output_dir,
                        shared_url=dbx_config["shared_url"],
                        password=dbx_config["password"],
                        max_retries=args.max_retries,
                        main_overwrite=args.overwrite,
                        checksum_check=True
                    )
                    original_sigint_handler = signal.signal(
                        signal.SIGINT, signal.SIG_IGN
                    )
                    p = Pool(args.download_threads)
                    signal.signal(signal.SIGINT, original_sigint_handler)
                    try:
                        complete_failures = p.map_async(
                            multi_download,
                            files_to_download
                        ).get()
                    except KeyboardInterrupt:
                        logger.error(
                            "Caught KeyboardInterrupt, terminating workers."
                        )
                        p.terminate()
                        sys.exit()
                    else:
                        logger.info("Normal termination")
                        p.close()
                        p.join()

                    complete_failures = [
                        j for i in complete_failures for j in i
                    ]
                else:
                    complete_failures = vast_dropbox.download_files(
                        files_to_download,
                        output_dir,
                        dbx_config["shared_url"],
                        dbx_config["password"],
                        args.max_retries,
                        args.overwrite)

                if len(complete_failures) > 0:
                    logger.warning(
                        "The following files failed to download correctly:"
                    )
                    for fail in complete_failures:
                        logger.warning(fail)
                    logger.warning("These files may be corrupted!")
            else:
                logger.info(
                    "Dry run selected. Would download the follwing"
                    " %s files:", len(files_to_download)
                )
                for i in files_to_download:
                    logger.info(i)

                logger.info("Total: %s files.", len(files_to_download))
        else:
            logger.info("No files to download with selected filters.")

    else:
        logger.info("Nothing to be done!")


def parse_args():
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
            "Provide a file containing the files available on Dropbox"
            " to download. Only use this option only if you wish"
            " to override the built-in list of Dropbox files"
            " that is already provided."
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
        help=(
            "Input of fields to fetch (can be"
            " obtained from 'find_sources.py')."
        ),
        default=None)

    parser.add_argument(
        '--user-files-list',
        type=str,
        help='Input of files to fetch.',
        default=None)

    parser.add_argument(
        '--only-epochs',
        type=str,
        help=(
            "Only download files from the selected epochs."
            " Enter as a list with no spaces, e.g. '1,2,4x'."
            " If nothing is entered then all epochs are fetched."
            " The current epochs are: {}.".format(
                ", ".join(user_friendly_epochs)
                )
        ),
        default=None)

    parser.add_argument(
        '--only-fields',
        type=str,
        help=(
            "Only download files from the selected fields."
            " Enter as a list with no spaces,"
            " e.g. 'VAST_0012+00A,VAST_0012-06A'."
            " If nothing is entered then all fields are fetched."
        ),
        default=None)

    parser.add_argument(
        '--stokes',
        type=str,
        help=(
            "Select which Stokes data products are to be downloaded"
            " Enter as a list separated by a comma with no space, e.g."
            " 'I,V'"
        )
    )

    parser.add_argument(
        '--skip-xml',
        action="store_true",
        help="Do not download XML files.")

    parser.add_argument(
        '--skip-txt',
        action="store_true",
        help="Do not download txt files.")

    parser.add_argument(
        '--skip-qc',
        action="store_true",
        help="Do not download the QC plots.")

    parser.add_argument(
        '--skip-components',
        action="store_true",
        help="Do not download components selavy files.")

    parser.add_argument(
        '--skip-islands',
        action="store_true",
        help="Do not download island selavy files.")

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
            " If overwrite is not selected, integrity checking"
            " will still be performed on the existing files and if"
            " the check fails, the file will be re-downloaded."))

    parser.add_argument(
        '--dry-run',
        action="store_true",
        help=(
            "Only print files that will be downloaded,"
            " without downloading them."
        ))

    parser.add_argument(
        '--debug',
        action="store_true",
        help='Set logging level to debug.')

    parser.add_argument(
        '--write-template-dropbox-config',
        action="store_true",
        help='Create a template dropbox config file.')

    parser.add_argument(
        '--legacy-download',
        type=str,
        help=(
            "Select the legacy version to download from. "
            "Enter with the included 'v', e.g. 'v0.6'."
            " Using this option will only download the legacy"
            " data, no other data shall be downloaded."
        ),
        default=None)

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

    parser.add_argument(
        '--download-threads',
        type=int,
        help=(
            'How many parallel downloads to attempt.'
            ' EXPERIMENTAL! See the VASTDROPBOX.md file'
            ' for full information.'
        ),
        default=1)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    numexpr.set_num_threads(2)
    args = parse_args()

    logfile = "get_vast_pilot_dbx_{}.log".format(
        runstart.strftime("%Y%m%d_%H:%M:%S"))

    logger = get_logger(args.debug, False, logfile=logfile)

    # Set the logging level of the dropbox module
    logging.getLogger("dropbox").setLevel(logging.WARNING)

    if args.write_template_dropbox_config:
        write_config_template()
        sys.exit()

    if not check_file(args.dropbox_config):
        logger.info(
            "A template dropbox file can be generated using "
            "python get_vast_pilot_dbx.py '--write-template-dropbox-config'"
        )
        sys.exit()

    dropbox_config = read_dbx_config(args.dropbox_config)

    output_dir = args.output
    setup_outdir(output_dir, args)

    vast_dropbox, base_file_list = setup_dropbox(dropbox_config)

    run_get_dropbox(
        vast_dropbox, base_file_list, dropbox_config, output_dir, args
    )

    runend = datetime.datetime.now()

    runtime = runend - runstart

    logger.info("Ran for {:.1f} minutes.".format(runtime.seconds / 60.))

    logger.info("Log file written to %s.", logfile)

    logger.info("All done!")
