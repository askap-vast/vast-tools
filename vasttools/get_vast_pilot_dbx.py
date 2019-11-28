#!/usr/bin/env python

import argparse
import dropbox
import os
import sys
import datetime
import configparser
import numpy as np

import logging
import logging.handlers
import logging.config

try:
    import colorlog
    use_colorlog=True
except ImportError:
    use_colorlog=False

def recursive_build_files(base_file_list, dbx, preappend=""):
    #Very annoyingling recursive file lists do not work on shared folders.
    #Hence this function is to fetch every single file available.
    folders=[]
    searched_folders=[]
    files=[]
    for i in base_file_list.entries:
        if type(i) == dropbox.files.FolderMetadata:
            if preappend=="":
                folders.append("/{}".format(i.name))
            else:
                folders.append("/{}/{}".format(preappend, i.name))
        else:
            if preappend=="":
                files.append("/{}".format(i.name))
            else:
                files.append("/{}/{}".format(preappend, i.name))

    while folders != searched_folders:
        for i in folders:
            if i not in searched_folders:
                these_files = dbx.files_list_folder("/{}".format(i), shared_link=shared_link)
                for j in these_files.entries:
                    if type(j) == dropbox.files.FolderMetadata:
                        if preappend=="" or i.startswith("/{}".format(preappend)):
                            folders.append("{}/{}".format(i, j.name))
                        else:
                            folders.append("/{}/{}/{}".format(preappend, i, j.name))
                    else:
                        if preappend=="" or i.startswith("/{}".format(preappend)):
                            files.append("{}/{}".format(i, j.name))
                        else:
                            files.append("/{}/{}/{}".format(preappend, i, j.name))
                searched_folders.append(i)
                logger.info("Searched {}".format(i))
                logger.debug("Folders: {}".format(folders))
                logger.debug("Searched Folders: {}".format(searched_folders))
    return files, folders

def download_files(files_list, pwd, output_dir, dbx, shared_url, password):
    for vast_file in files_list:
        # pwd = os.getcwd()
        download_path = os.path.join(pwd, output_dir, vast_file[1:])
        dropbox_path = "{}".format(vast_file)
        logger.debug("Download path: {}".format(download_path))
        logger.info("Downloading {}...".format(dropbox_path))
        dbx.sharing_get_shared_link_file_to_file(download_path, shared_url, path=dropbox_path, link_password=password)
    

def check_dir(directory):
    return os.path.isdir(directory)
    
def check_file(file_to_check):
    return os.path.isfile(file_to_check)

parser=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('dropbox_config_file', type=str, help='Dropbox config file to be read in containing the shared url, password and access token.')

parser.add_argument('--output', type=str, help='Name of the local output directory where files will be saved', default="vast_dropbox")
parser.add_argument('--available-epochs', action="store_true", help='Print out what Epochs are available.', default="")
parser.add_argument('--available-files', action="store_true", help='Print out a list of available files on the shared folder.', default="")
parser.add_argument('--download-epoch', type=int, help='Select to download an entire Epoch directory', default=0)
parser.add_argument('--files-list', type=str, help='Input of files to fetch.', default="")
parser.add_argument('--debug', action="store_true", help='Set logging level to debug.', default="")
parser.add_argument('--write-config', action="store_true", help='Create a template dropbox config file.', default="")
# parser.add_argument('--combined-only', action="store_true", help='Only return combined products.', default="")
# parser.add_argument('--tiles-only', action="store_true", help='Only return tiles products.', default="")

args=parser.parse_args()

logger = logging.getLogger()
s = logging.StreamHandler()
logformat='[%(asctime)s] - %(levelname)s - %(message)s'

if use_colorlog:
    formatter = colorlog.ColoredFormatter(
        # "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
        "%(log_color)s[%(asctime)s] - %(levelname)s - %(blue)s%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        reset=True,
        log_colors={
        'DEBUG':    'cyan',
        'INFO':     'green',
        'WARNING':  'yellow',
        'ERROR':    'red',
        'CRITICAL': 'red,bg_white',
        },
        secondary_log_colors={},
        style='%'
    )
else:
    formatter = logging.Formatter(logformat, datefmt="%Y-%m-%d %H:%M:%S")

s.setFormatter(formatter)
logger.addHandler(s)

if args.debug:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

if args.write_config:
    config_file = "dropbox.cfg"
    with open(config_file, "w") as f:
        f.write("""[dropbox]
shared_url = ENTER_URL
password = ENTER_PASSWORD
access_token = ENTER_ACCESS_TOKEN
""")
    logger.info("Writen an example dropbox config file to '{}'.".format(config_file))
    sys.exit()

if not check_file(args.dropbox_config_file):
    logger.critical("Cannot find dropbox config file '{}!".format(args.dropbox_config_file))
    sys.exit()

config = configparser.ConfigParser()
config.read(args.dropbox_config_file)

shared_url = config["dropbox"]["shared_url"]
password = config["dropbox"]["password"]
access_token = config["dropbox"]["access_token"]

logger.debug(shared_url)
logger.debug(password)
logger.debug(access_token)

output_dir = args.output

#check dir
if not args.available_epochs:
    if check_dir(output_dir):
        logger.warning("Output directory '{}' already exists!".format(output_dir))
        logger.warning("Will not overwrite.")
        sys.exit()
    else:
        os.mkdir(output_dir)
    

dbx = dropbox.Dropbox(access_token)

shared_link = dropbox.files.SharedLink(url=shared_url, password=password)

base_file_list = dbx.files_list_folder("", shared_link=shared_link)

if args.available_epochs:
    logger.info("The following epochs are available:")
    for i in base_file_list.entries:
        if type(i) == dropbox.files.FolderMetadata and "EPOCH" in i.name:
            logger.info(i.name)

elif args.available_files:
    logger.info("Gathering a list of files - this will take a moment...")
    files_list, folders_list = recursive_build_files(base_file_list, dbx)
    logger.info("Found {} files.".format(len(files_list)))
    with open(os.path.join(output_dir, "vast_dbx_file_list.txt"), "w") as f:
        f.write("# File list on VAST Pilot survey dropbox as of {}\n".format(datetime.datetime.now()))
        [f.write(i+"\n") for i in files_list]
    logger.info("All available files written to {}".format(os.path.join(output_dir, "vast_dbx_file_list.txt")))
    
elif args.download_epoch != 0:
    epochs = []
    for i in base_file_list.entries:
        if type(i) == dropbox.files.FolderMetadata and "EPOCH" in i.name:
            epochs.append(int(i.name.split('EPOCH')[-1]))
    if args.download_epoch not in epochs:
        logger.error("EPOCH{:02d} has not yet been released!".format(args.download_epoch))
        sys.exit()
    else:
        epoch_string = "EPOCH{:02d}".format(args.download_epoch)
        epoch_file_list = dbx.files_list_folder("/{}".format(epoch_string), shared_link=shared_link)
        logger.info("Gathering {} files to download, please wait...".format(epoch_string))
        files_list, folders_list = recursive_build_files(epoch_file_list, dbx, preappend=epoch_string)
        logger.info("{} files to download".format(len(files_list)))
        #Mimic the directory structure locally
        # epoch_output_dir = os.path.join(output_dir, epoch_string)
        # os.mkdir(epoch_output_dir)
        for folder in folders_list:
            os.makedirs(os.path.join(output_dir, folder[1:]))
        logger.info("Downloading files for {}...".format(epoch_string))
        download_files(files_list, os.getcwd(), output_dir, dbx, shared_url, password)

elif args.files_list!="":
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
        
    dirs_to_create = np.unique(["/".join(i.strip().split("/")[1:-1]) for i in files_to_download])
    
    for i in dirs_to_create:
        if i=="":
            continue
        os.makedirs(os.path.join(output_dir, i))
    
    logger.info("Downloading {} files from '{}'...".format(len(files_to_download), args.files_list))
    download_files(files_to_download, os.getcwd(), output_dir, dbx, shared_url, password)

else:
    logger.info("Nothing to be done!")
    
logger.info("All done!")
    
            

    


