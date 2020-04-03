# get\_vast\_pilot\_dbx.py

This script allows for simple downloading of the VAST Pilot survey from Dropbox.

Features:
* State which Epochs are available.
* Flexible download of data to users wants.

## Prerequisites

**This script requires you to have a Dropbox App 'access token'.** You do this by making an 'app' on your Dropbox account and then generating an OAuth token for that app.

This tutorial shows you how to obtain one: http://99rabbits.com/get-dropbox-access-token/. Make sure you select the `Full Dropbox` option in the access section.

Otherwise the requirements installed from the main repo will cover all the python needs.

You also need to know the shared Dropbox URL of the Pilot survey and the password.

## Usage
```
usage: get_vast_pilot_dbx.py [-h] [--dropbox-config DROPBOX_CONFIG]
                             [--output OUTPUT] [--available-epochs]
                             [--available-files AVAILABLE_FILES]
                             [--get-available-files] [--download]
                             [--find-fields-input FIND_FIELDS_INPUT]
                             [--user-files-list USER_FILES_LIST]
                             [--only-epochs ONLY_EPOCHS]
                             [--only-fields ONLY_FIELDS] [--stokes STOKES]
                             [--skip-xml] [--skip-txt] [--skip-qc]
                             [--skip-components] [--skip-islands]
                             [--skip-field-images] [--skip-bkg-images]
                             [--skip-rms-images] [--skip-all-images]
                             [--combined-only] [--tile-only] [--overwrite]
                             [--dry-run] [--debug]
                             [--write-template-dropbox-config]
                             [--legacy-download LEGACY_DOWNLOAD] [--include-legacy]
                             [--max-retries MAX_RETRIES]
                             [--download-threads DOWNLOAD_THREADS]

optional arguments:
  -h, --help            show this help message and exit
  --dropbox-config DROPBOX_CONFIG
                        Dropbox config file to be read in containing the shared
                        url, password and access token. A template can be generated
                        using '--write-template-dropbox-config'. (default:
                        dropbox.cfg)
  --output OUTPUT       Name of the local output directory where files will be
                        saved (default: vast_dropbox)
  --available-epochs    Print out what Epochs are available. (default: False)
  --available-files AVAILABLE_FILES
                        Provide a file containing the files available on Dropbox to
                        download. Only use this option only if you wish to override
                        the built-in list of Dropbox files that is already
                        provided. (default: None)
  --get-available-files
                        Generate the list of available files on the shared folder.
                        The list will be saved to a file. (default: False)
  --download            Download data according to the filter options entered
                        (default: None)
  --find-fields-input FIND_FIELDS_INPUT
                        Input of fields to fetch (can be obtained from
                        'find_sources.py'). (default: None)
  --user-files-list USER_FILES_LIST
                        Input of files to fetch. (default: None)
  --only-epochs ONLY_EPOCHS
                        Only download files from the selected epochs. Enter as a
                        list with no spaces, e.g. '1,2,4x'. If nothing is entered
                        then all epochs are fetched. The current epochs are: 1, 2,
                        3x, 4x, 5x, 6x, 7x, 8, 9, 10x, 11x. (default: None)
  --only-fields ONLY_FIELDS
                        Only download files from the selected fields. Enter as a
                        list with no spaces, e.g. 'VAST_0012+00A,VAST_0012-06A'. If
                        nothing is entered then all fields are fetched. (default:
                        None)
  --stokes STOKES       Select which Stokes data products are to be downloaded
                        Enter as a list separated by a comma with no space, e.g.
                        'I,V' (default: None)
  --skip-xml            Do not download XML files. (default: False)
  --skip-txt            Do not download txt files. (default: False)
  --skip-qc             Do not download the QC plots. (default: False)
  --skip-components     Do not download components selavy files. (default: False)
  --skip-islands        Do not download island selavy files. (default: False)
  --skip-field-images   Do not download field images. (default: False)
  --skip-bkg-images     Do not download background images. (default: False)
  --skip-rms-images     Do not download background images. (default: False)
  --skip-all-images     Only download non-image data products. (default: False)
  --combined-only       Only download the combined products. (default: False)
  --tile-only           Only download the combined products. (default: False)
  --overwrite           Overwrite any files that already exist in the output
                        directory. If overwrite is not selected, integrity checking
                        will still be performed on the existing files and if the
                        check fails, the file will be re-downloaded. (default:
                        False)
  --dry-run             Only print files that will be downloaded, without
                        downloading them. (default: False)
  --debug               Set logging level to debug. (default: False)
  --write-template-dropbox-config
                        Create a template dropbox config file. (default: False)
  --legacy-download LEGACY_DOWNLOAD
                        Select the legacy version to download from. Enter with the
                        included 'v', e.g. 'v0.6'. Using this option will only
                        download the legacy data, no other data shall be
                        downloaded. (default: None)
  --include-legacy      Include the 'LEGACY' directory when searching through
                        files. Only valid when using the '--get-available-files'
                        option. (default: False)
  --max-retries MAX_RETRIES
                        How many times to attempt to retry a failed download
                        (default: 2)
  --download-threads DOWNLOAD_THREADS
                        How many parallel downloads to attempt. EXPERIMENTAL! See
                        the VASTDROPBOX.md file for full information. (default: 1)
```

To run the script needs a Dropbox configuration file, which by default is assumed to be named 'dropbox.cfg'. Create a text file in the following format and enter the respective values:
```
[dropbox]
shared_url = ENTER_URL
password = ENTER_PASSWORD
access_token = ENTER_ACCESS_TOKEN
```
**Double check that the password is correct before running!** Because of the many calls, a wrong password can lead to the link being locked for a period of time.

There is no need to put quotes around the strings. A template can be generated by using:
```
get_vast_pilot_dbx.py --write-template-dropbox-config
```
Use the option `--dropbox-config` if your config file is named something different than the default.

A log file will be saved for every run of the script.

### Modes

There are 2 main ways in which the script is intended to be used:

1. **Download according to filter flags** - Use the options such as `--only-epochs`, ` --only-fields` and other filter options to download the data you want.
2. **Easy downloading of required fields using `--find-fields-input`** - Directly uses the output from `find_sources.py --find-fields` to auto fill the `--only-fields` option. Other flags also apply.

Data will only download when the `--download` option is provided. This can be used in combination with `--dry-run` to see exactly what files will be downloaded before starting the download process.

**Note** As of vast-tools v1.2.0, the module comes with a packaged list of Dropbox files of the latest release, so users are no longer required to either fetch, or let the script fetch, a list of available files or provide one.

There are a few other options that present but are now mostly considered legacy as they should not be required often, if at all:

* `--available-epochs` will only display the currently released epochs. Nothing will be downloaded.
* `--get-available-files` will generate a complete list of all the files avaialble in the Dropbox folder. This is a legacy option at this point which shouldn't be needed. The module has an inbuilt file list that is kept up to date and the flexibility in downloading means users no longer need to build their own list of files.
* `--user-files-list` defines a text file that contains the files you wish to download. Usually used in combination with the previous option.

Take note of the `--overwrite` option. By default this is set to `False` such that it will skip files already present in the output directory. Using this option will download all files and overwrite any exisiting files if they are already present.

### File integrity and retries
The script will check the downloaded file checksum against the correct checksum stored in it's own data file. It should also catch exceptions when downloads timeout or there are network issues. In each case if there is a problem the file will be remembered as failed, and when the main download loop has finished, it will attempt again to download any failed files. You can set how many times it retries using the `--max-retries` option.

**Check the log output for any files that failed even after retries!**

### Filtering the data requested
Note the following options in the Dropbox script:
```
  --only-epochs         Only download data of the requested epochs. (default: all epochs)
  --only-fieds          Only download data of the requested fields. (default: all fields)
  --stokes              Only download selected Stokes. (default: all stokes)
  --skip-xml            Do not download XML files. (default: False)
  --skip-txt            Do not download txt files. (default: False)
  --skip-qc             Do not download the QC plots. (default: False)
  --skip-components     Do not downlaod selavy component files. (default: False)
  --skip-islands        Do not downlaod selavy island files. (default: False)
  --skip-field-images   Do not download field images. (default: False)
  --skip-bkg-images     Do not download background images. (default: False)
  --skip-rms-images     Do not download background images. (default: False)
  --skip-all-images     Only download non-image data products. (default: False)
  --combined-only       Only download the combined products. (default: False)
  --tile-only           Only download the combined products. (default: False)
```
You can use these flags to only obtain the bits of the data you need (see [examples](#examples) below).

**Note** The filtering does not quite work on the quality control plots due to the slight different naming scheme. This is hoped to be addressed in a future update, but if you wish to view the plots then the suggested method is to filter out everything apart from the QC and download this.

### User Files List
**Note** It's now easier to perform custom queries using the flags as presented above, so this method should not be needed often, if at all.

When supplying a list of files it needs to follow the directory structure of the Dropbox. It also needs to explictly state the files - i.e. you **cannot use wildcards** (sorry it's the limitations of using Dropbox this way).

For example if I wanted to download a set of STOKES I COMBINED images from EPOCH01, the file would be:
```
/EPOCH01/COMBINED/STOKESI_IMAGES/VAST_0918+00A.EPOCH01.I.fits
/EPOCH01/COMBINED/STOKESI_IMAGES/VAST_1739-25A.EPOCH01.I.fits
/EPOCH01/COMBINED/STOKESI_IMAGES/VAST_1753-18A.EPOCH01.I.fits
/EPOCH01/COMBINED/STOKESI_IMAGES/VAST_0943+00A.EPOCH01.I.fits
/EPOCH01/COMBINED/STOKESI_IMAGES/VAST_0216+00A.EPOCH01.I.fits
/EPOCH01/COMBINED/STOKESI_IMAGES/VAST_2143-06A.EPOCH01.I.fits
/EPOCH01/COMBINED/STOKESI_IMAGES/VAST_2208-06A.EPOCH01.I.fits

```
Note the leading `/` which is also needed.

I recommend you either run `get_vast_pilot_dbx.py --get-available-files` or grab the file list directly from the dropbox repository, and use this output to build your request (**warning** `--get-available-fields` will take a while to run, up to 30 mins with legacy).

### Parallel Dropbox Downloads (EXPERIMENTAL)
There is an experimental option of `--download-threads`. This allows multiple Dropbox download commands to be launched in parallel to speed up the download of large requests. However this mode is considered experimental and logging is not set up to use with this mode. Warning level messages will be printed to the terminal but you will not be receiving any feedback on the download until it completes (see issue [#141](https://github.com/askap-vast/vast-tools/issues/141) on Github). Integrity checking is still performed.

Do not use a high number of parallel downloads, we suggest no more than 6. And if only downloading a small amount of files it is recommended to not use this mode.

Example
```
get_vast_pilot_dbx.py --download --only-epochs 1 --output VAST_DOWNLOAD --download-threads 4
```

### Legacy Data Downloading
To download data from a specific legacy version you can use the `--legacy-download` option, which takes an argument of the legacy version you wish to download from in the form of the directory name on Dropbox (we suggest browsing the Dropbox folder via a browser to check the versions). For example, to limit the download to `v0.6` legacy data, the input would be:
```
get_vast_pilot_dbx.py --download --output VAST_DOWNLOAD --legacy-download v0.6 <use normal filter flags here>
```
The above would limit the download request to only use the data that is present in the `/LEGACY/v0.6/` directory.

### Examples
Below are examples of how to download the data with different scenarios in mind. Remember you can add `--dry-run` to your command to see exactly what will be downloaded without actually downloading.

#### Downloading everything
To download the entire release structure:
```
get_vast_pilot_dbx.py --download --output VAST_DOWNLOAD
```
This will place all the files in the directory `VAST_DOWNLOAD`. As of data releast v1.0, the total size stands at 8.0 TB. To clarify this will **not** download the legacy directory.

#### Downloading an entire epoch
Using epoch 01 as an example:
```
get_vast_pilot_dbx.py --download --only-epochs 1 --output VAST_DOWNLOAD
```
This will place the EPOCH01 directory in `VAST_DOWNLOAD`.

#### Making a selection
Scenario:

* Download fields VAST_0918+00A and VAST_1739-25A.
* From epochs 1, 2, 8 and 9.
* Combined data products only.
* Stokes I and V.
* Field images and selavy components only (txt files).
* No quality control plots.

```
get_vast_pilot_dbx.py --download --output VAST_DOWNLOAD --only-epochs 1,2,8,9 --only-fields VAST_0918+00A,VAST_1739-25A --combined-only --stokes I,V --skip-bkg-images --skip-rms-images --skip-islands --skip-xml --skip-qc
```
This will place the relevant files in the directory `VAST_DOWNLOAD`.

#### Downloading fields required

1. Run `find_sources.py` for your sources and you will obtain an output like so:
    ```
    ra,dec,name,sbid,field_name
    321.749583333333,-44.2686111111111,Q 2123-4429B,9673,VAST_2112-43A
    348.945,-59.0544444444444,ESO 148-IG02,9673,VAST_2256-56A
    ```
    **Tip**: The input here is only looking for the `field_name` column. So it's also possible to pass a CSV file with just that column header and the fields you want.

2. Pass this output to `get_vast_pilot_dbx.py` to download only these fields. In addition, in this example we assume that we only want the combined Stokes I field, and rms images and the selavy component files (txt format only).

    The command for this becomes:
    ```
    get_vast_pilot_dbx.py --find-fields-input find-fields-ouput.csv --output VAST_DOWNLOAD --stokes I --skip-xml --skip-bkg-images --skip-qc --skip-islands --combined-only
    ```

#### Downloading a user selected set of files

1. Create the text file containing the files, e.g. `to_download.txt`:
```
/EPOCH01/COMBINED/STOKESI_IMAGES/VAST_0918+00A.EPOCH01.I.fits
/EPOCH01/COMBINED/STOKESI_IMAGES/VAST_1739-25A.EPOCH01.I.fits
/EPOCH01/COMBINED/STOKESI_IMAGES/VAST_1753-18A.EPOCH01.I.fits
/EPOCH01/COMBINED/STOKESI_IMAGES/VAST_0943+00A.EPOCH01.I.fits
/EPOCH01/COMBINED/STOKESI_IMAGES/VAST_0216+00A.EPOCH01.I.fits
/EPOCH01/COMBINED/STOKESI_IMAGES/VAST_2143-06A.EPOCH01.I.fits
/EPOCH01/COMBINED/STOKESI_IMAGES/VAST_2208-06A.EPOCH01.I.fits

```
2. Then run with:
```
get_vast_pilot_dbx.py --files-list to_download.txt --output VAST_DOWNLOAD
```
This will place these files in `VAST_DOWNLOAD`. The directory structure will be mimiced. You can still apply flags to this method if you want to filter your own list.
