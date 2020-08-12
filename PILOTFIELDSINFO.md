# pilot\_fields\_info.py

This script allows you to obtain basic information on fields in the VAST Pilot Survey.

The script will print out a table containing the following information for a searched field(s):
* EPOCH
* FIELD_NAME
* SBID 
* RA_HMS (Beam 0)
* DEC_DMS (Beam 0)
* DATEOBS
* DATEEND

In addition you can request the script return the largest or calculated common psf of the field per epoch, as well as printing all the information of the individual ASKAP beams. 

# Running the script

There are two methods in which to define the fields you wish to query:

1. Command line: enter your field(s) as so:
```
pilot_fields_info.py VAST_0532-50A VAST_1212+00A VAST_2257-06A
```

2. Or use an input csv file, which must have the column `field_name`:
```
ra,dec,name,sbid,field_name
321.749583333333,-44.2686111111111,Q 2123-4429B,9673,VAST_2112-43A
348.945,-59.0544444444444,ESO 148-IG02,9673,VAST_2256-56A
```
```
pilot_fields_info.py my_fields.csv
```

If you wish to save the output then using the `--save` flag will write the results for each field to their own individual file in the current working directory.

See the section below for details on the other options available when running.

## Usage

Most options should be self explanatory. The lightcurve plots and csv files are saved in the same directory as the input

```
usage: pilot_fields_info.py [-h] [--psf] [--largest-psf] [--common-psf] [--all-psf] [--save] [--quiet]
                            [--debug] [--nice NICE]
                            fields [fields ...]

positional arguments:
  fields         Fields to query (or csv file containing fields).

optional arguments:
  -h, --help     show this help message and exit
  --psf          Include the used PSF of the 36 beams that make up the field. Usually set from beam
                 00. (default: False)
  --largest-psf  Include the largest PSF of the 36 beams that make up the field. (default: False)
  --common-psf   Include the common PSF of the 36 beams that make up the field. (default: False)
  --all-psf      Include all the PSF information for the field. (default: False)
  --save         Save the resulting information. Files will be saved to the current working directory
                 in the form of 'VAST_XXXX+/-XXA_field_info.csv'. (default: False)
  --quiet        Turn off non-essential terminal output. (default: False)
  --debug        Turn on debug output. (default: False)
  --nice NICE    Set nice level. (default: 5)
```

## Example

```
❯ pilot_fields_info.py VAST_0532-50A VAST_1212+00A
[2020-02-27 16:00:18] - INFO - Will find information for the following fields:
[2020-02-27 16:00:18] - INFO - VAST_0532-50A
[2020-02-27 16:00:18] - INFO - VAST_1212+00A
[2020-02-27 16:00:18] - INFO - VAST_0532-50A information:
EPOCH    FIELD_NAME       SBID  RA_HMS (Beam 0)    DEC_DMS (Beam 0)    DATEOBS                  DATEEND
-------  -------------  ------  -----------------  ------------------  -----------------------  -----------------------
01       VAST_0532-50A    9668  05:29:03.722       -49:34:04.84        2019-08-27 21:16:19.910  2019-08-27 21:28:26.499
02       VAST_0532-50A   10343  05:29:03.722       -49:34:04.84        2019-10-30 15:26:18.379  2019-10-30 15:39:14.734
03x      VAST_0532-50A   10336  05:29:03.722       -49:34:04.84        2019-10-29 16:33:44.940  2019-10-29 16:46:11.436
04x      VAST_0532-50A   10901  05:29:03.722       -49:34:04.84        2019-12-19 15:49:51.855  2019-12-19 16:01:48.491
05x      VAST_0532-50A   11120  05:29:03.722       -49:34:04.84        2020-01-10 14:26:48.248  2020-01-10 14:38:44.884
07x      VAST_0532-50A   11341  05:29:03.722       -49:34:04.84        2020-01-16 14:00:36.072  2020-01-16 14:12:22.755
08       VAST_0532-50A   11544  05:29:03.722       -49:34:04.84        2020-01-24 12:40:29.080  2020-01-24 12:52:25.716
09       VAST_0532-50A   11568  05:29:03.722       -49:34:04.84        2020-01-25 12:36:34.625  2020-01-25 12:48:31.261
10x      VAST_0532-50A   11391  05:29:03.722       -49:34:04.84        2020-01-17 13:45:35.521  2020-01-17 13:57:32.157
11x      VAST_0532-50A   11463  05:29:03.722       -49:34:04.84        2020-01-18 13:46:39.665  2020-01-18 13:58:36.301
[2020-02-27 16:00:18] - INFO - VAST_1212+00A information:
EPOCH    FIELD_NAME       SBID  RA_HMS (Beam 0)    DEC_DMS (Beam 0)    DATEOBS                  DATEEND
-------  -------------  ------  -----------------  ------------------  -----------------------  -----------------------
01       VAST_1212+00A    9669  12:10:19.001       +00:31:29.93        2019-08-28 02:26:12.637  2019-08-28 02:38:19.226
02       VAST_1212+00A   10380  12:10:19.001       +00:31:29.93        2019-10-31 22:14:59.611  2019-10-31 22:27:06.200
05x      VAST_1212+00A   11140  12:10:19.001       +00:31:29.93        2020-01-10 20:16:49.668  2020-01-10 20:28:36.351
07x      VAST_1212+00A   11357  12:10:19.001       +00:31:29.93        2020-01-16 19:45:28.941  2020-01-16 19:57:25.577
08       VAST_1212+00A   11408  12:10:19.001       +00:31:29.93        2020-01-17 19:45:44.092  2020-01-17 19:57:40.728
09       VAST_1212+00A   11480  12:10:19.001       +00:31:29.93        2020-01-18 19:41:49.637  2020-01-18 19:53:36.320
[2020-02-27 16:00:18] - INFO - Processing took 0.0 minutes.

```
