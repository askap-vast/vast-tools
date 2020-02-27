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

See the section below for details on the options available when running.

## Usage

Most options should be self explanatory. The lightcurve plots and csv files are saved in the same directory as the input

```
usage: pilot_fields_info.py [-h] [--psf] [--common-psf] [--all-psf] [--save] [--quiet] [--debug]
                            [--nice NICE]
                            fields [fields ...]

positional arguments:
  fields        Fields to query (or csv file containing fields).

optional arguments:
  -h, --help    show this help message and exit
  --psf         Include the largest PSF of the 36 beams that make up the field. (default: False)
  --common-psf  Include the common PSF of the 36 beams that make up the field. (default: False)
  --all-psf     Include all the PSF information for the field. (default: False)
  --save        Save the resulting information. (default: False)
  --quiet       Turn off non-essential terminal output. (default: False)
  --debug       Turn on debug output. (default: False)
  --nice NICE   Set nice level. (default: 5)
```
