#!/usr/bin/env python

import argparse
import dropbox
import os
import sys
import datetime
import configparser
import numpy as np
import itertools

import logging
import logging.handlers
import logging.config

try:
    import colorlog
    use_colorlog = True
except ImportError:
    use_colorlog = False


def recursive_build_files(base_file_list, dbx, preappend="", legacy=False):
    '''
    Very annoyingling recursive file lists do not work on shared folders.
    This function is to fetch every single file available by iterating over
    all folders found to build up a unique file list.

    :param base_file_list: a list of files in the root dropbox folder
    :type base_file_list:
    :param dbx: the dropbpx connection
    :type dbx: A dropbox.Dropbox object.
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
                these_files = dbx.files_list_folder(
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
        dbx,
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
    :param dbx:
    :type dbx:
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
        dbx.sharing_get_shared_link_file_to_file(
            download_path, shared_url, path=dropbox_path,
            link_password=password)


def check_dir(directory):
    '''
    Wrapper for os.path.isdir()

    :param directory: path to directory we're checking the existence of
    :type directory: str

    :returns: True if the directory exists, False otherwise
    :rtype: bool
    '''

    return os.path.isdir(directory)


def check_file(file_to_check):
    '''
    Wrapper for os.path.isfile()

    :param file_to_check: path to file we're checking the existence of
    :type file_to_check: str

    :returns: True if the specified path is an existing file, False otherwise
    :rtype: bool
    '''

    return os.path.isfile(file_to_check)


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
    type=int,
    help='Select to download an entire Epoch directory. Enter as an integer.',
    default=0)

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

if not check_file(args.dropbox_config):
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
    if check_dir(output_dir):
        logger.warning(
            "Output directory '{}' already exists!".format(output_dir))
        logger.warning("Files may get overwritten!")
    else:
        os.mkdir(output_dir)


dbx = dropbox.Dropbox(access_token)

shared_link = dropbox.files.SharedLink(url=shared_url, password=password)

base_file_list = dbx.files_list_folder("", shared_link=shared_link)

spinner = itertools.cycle(['-', '/', '|', '\\'])

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

    files_list, folders_list = recursive_build_files(
        base_file_list,
        dbx,
        legacy=args.include_legacy)
    logger.info("Found {} files.".format(len(files_list)))
    vast_list_file_name = "vast_dbx_file_list_{}.txt".format(now_str)

    with open(vast_list_file_name, "w") as f:
        f.write("# File list on VAST Pilot survey dropbox as of {}\n".format(
            now))
        [f.write(i + "\n") for i in files_list]

    logger.info("All available files written to {}".format(
        vast_list_file_name))

elif args.download_epoch != 0:
    epochs = []
    for i in base_file_list.entries:
        if isinstance(i, dropbox.files.FolderMetadata) and "EPOCH" in i.name:
            epochs.append(int(i.name.split('EPOCH')[-1]))
    if args.download_epoch not in epochs:
        logger.error(
            "EPOCH{:02d} has not yet been released!".format(
                args.download_epoch))
        sys.exit()
    else:
        epoch_string = "EPOCH{:02d}".format(args.download_epoch)
        epoch_file_list = dbx.files_list_folder(
            "/{}".format(epoch_string), shared_link=shared_link)
        logger.info(
            "Gathering {} files to download...".format(epoch_string))
        files_list, folders_list = recursive_build_files(
            epoch_file_list, dbx, preappend=epoch_string)
        logger.info("{} files to download".format(len(files_list)))

        for folder in folders_list:
            os.makedirs(os.path.join(output_dir, folder[1:]), exist_ok=True)
        logger.info("Downloading files for {}...".format(epoch_string))
        download_files(
            files_list,
            os.getcwd(),
            output_dir,
            dbx,
            shared_url,
            password)

elif args.files_list is not None:
    if not check_file(args.files_list):
        logger.error("Supplied file '{}' not found!".format(args.files_list))
        sys.exit()
    with open(args.files_list, 'r') as f:
        userlines = f.readlines()

    # check files start with / and ignore #
    files_to_download = []

    for i in userlines:
        if i.startswith("#"):
            continue
        else:
            if i.startswith("/"):
                files_to_download.append(i.strip())
            else:
                files_to_download.append("/{}".format(i.strip()))

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
    download_files(
        files_to_download,
        os.getcwd(),
        output_dir,
        dbx,
        shared_url,
        password,
        overwrite=args.overwrite)

else:
    logger.info("Nothing to be done!")

end = datetime.datetime.now()

runtime = end - now

logger.info("Ran for {:.1f} minutes.".format(runtime.seconds / 60.))

logger.info("Log file written to {}".format(logfilename))

logger.info("All done!")
