# find\_sources.py

This script allows you to quickly query data from the RACS and VAST Pilot surveys on provided coordinates, either through the command line or using a csv list.

This includes Stokes I and Stokes V (negative sources included).

The outputs are/can be:
* Postage stamp fits images of each location.
* DS9 region file or kvis annotation file of selavy sources withing postage stamp.
* Figure png plots of postage stamp, with overlaid selavy sources.
* Crossmatch output file containing information on the nearest matched component.

# Running on your own machine
By default the script is set up for use on the ada machine. **No local data is required to run the `--find-fields` option which will find which fields that contain your sources of interest.** To create postage FITS files or PNG images, a copy of the survey data is required locally.

You must tell the script where your data is by using the following options:
```
  --img-folder IMG_FOLDER
                        Path to folder where images are stored (default: None)
  --rms-folder RMS_FOLDER
                        Path to folder where image RMS estimates are stored (default: None)
  --cat-folder CAT_FOLDER
                        Path to folder where selavy catalogues are stored (default: None)
```
For example:
```
--img-folder /Users/askap/my-pilot-data/EPOCH01/COMBINED/STOKESI_IMAGES
```

The script assumes that VAST Pilot data are in the same directory structure as that used in the Dropbox folder. If you are running `find_sources.py` on your own machine we recommend first using the `--find-fields` flag, downloading the relevant fields to an appropriate base directory and then re-running the script as normal.

## Warning!
* Currently the crossmatching matches against **components**. Check your results to see if the component is part of an island (`has_sibling` column will = 1) and query the island catalogue, using the `island_id` if you need further information on your source.

* If you are running this script on ada you also have access to RACS data to search for sources. Remember that RACS is not a VAST data product and you must have explicit permission to obtain and use RACS data. The find-fields option will work for RACS regardless (use `--vast-pilot 0`). If you have previously been given access to RACS data we can help set up the data structure to use with this script.

## Usage

Most options should be self explanatory. See examples below on how to run the script.

All output is placed in an output directory of which the name can be set with the option `--out-folder`.

Can be run in either Stokes I or Stokes V, not both at once.
```
usage: find_sources.py [-h] [--vast-pilot VAST_PILOT] [--imsize IMSIZE] [--maxsep MAXSEP]
                       [--out-folder OUT_FOLDER] [--source-names SOURCE_NAMES]
                       [--crossmatch-radius CROSSMATCH_RADIUS] [--crossmatch-radius-overlay]
                       [--use-tiles] [--img-folder IMG_FOLDER] [--rms-folder RMS_FOLDER]
                       [--cat-folder CAT_FOLDER] [--create-png] [--png-selavy-overlay]
                       [--png-linear-percentile PNG_LINEAR_PERCENTILE] [--png-use-zscale]
                       [--png-zscale-contrast PNG_ZSCALE_CONTRAST] [--png-no-island-labels]
                       [--png-ellipse-pa-corr PNG_ELLIPSE_PA_CORR] [--png-no-colorbar]
                       [--ann] [--reg] [--stokesv] [--quiet] [--crossmatch-only]
                       [--selavy-simple] [--process-matches] [--debug] [--no-background-rms]
                       [--find-fields] [--clobber]
                       "HH:MM:SS [+/-]DD:MM:SS" OR input.csv

positional arguments:
  "HH:MM:SS [+/-]DD:MM:SS" OR input.csv
                        Right Ascension and Declination in format "HH:MM:SS [+/-]DD:MM:SS",
                        in quotes. E.g. "12:00:00 -20:00:00". Degrees is also acceptable,
                        e.g. "12.123 -20.123". Multiple coordinates are supported by
                        separating with a comma (no space) e.g. "12.231 -56.56,123.4 +21.3".
                        Finally you can also enter coordinates using a .csv file. See example
                        file for format.

optional arguments:
  -h, --help            show this help message and exit
  --vast-pilot VAST_PILOT
                        Select the VAST Pilot Epoch to query. Epoch 0 is RACS. (default: 1)
  --imsize IMSIZE       Edge size of the postagestamp in arcmin (default: 30.0)
  --maxsep MAXSEP       Maximum separation of source from beam centre in degrees. (default:
                        1.0)
  --out-folder OUT_FOLDER
                        Name of the output directory to place all results in. (default:
                        find_sources_output_YYYYMMDD_HH:MM:SS)
  --source-names SOURCE_NAMES
                        Only for use when entering coordaintes via the command line. State
                        the name of the source being searched. Use quote marks for names that
                        contain a space. For multiple sources separate with a comma with no
                        space, e.g. "SN 1994N,SN 2003D,SN 2019A". (default: )
  --crossmatch-radius CROSSMATCH_RADIUS
                        Crossmatch radius in arcseconds (default: 15.0)
  --crossmatch-radius-overlay
                        A circle is placed on all PNG and region/annotation files to
                        represent the crossmatch radius. (default: False)
  --use-tiles           Use the individual tiles instead of combined mosaics. (default:
                        False)
  --img-folder IMG_FOLDER
                        Path to folder where images are stored (default: None)
  --rms-folder RMS_FOLDER
                        Path to folder where image RMS estimates are stored (default: None)
  --cat-folder CAT_FOLDER
                        Path to folder where selavy catalogues are stored (default: None)
  --create-png          Create a png of the fits cutout. (default: False)
  --png-selavy-overlay  Overlay selavy components onto the png image. (default: False)
  --png-linear-percentile PNG_LINEAR_PERCENTILE
                        Choose the percentile level for the png normalisation. (default:
                        99.9)
  --png-use-zscale      Select ZScale normalisation (default is 'linear'). (default: False)
  --png-zscale-contrast PNG_ZSCALE_CONTRAST
                        Select contrast to use for zscale. (default: 0.1)
  --png-no-island-labels
                        Disable island lables on the png. (default: False)
  --png-ellipse-pa-corr PNG_ELLIPSE_PA_CORR
                        Correction to apply to ellipse position angle if needed (in deg).
                        Angle is from x-axis from left to right. (default: 0.0)
  --png-no-colorbar     Do not show the colorbar on the png. (default: False)
  --ann                 Create a kvis annotation file of the components. (default: False)
  --reg                 Create a DS9 region file of the components. (default: False)
  --stokesv             Use Stokes V images and catalogues if available. (default: False)
  --quiet               Turn off non-essential terminal output. (default: False)
  --crossmatch-only     Only run crossmatch, do not generate any fits or png files. (default:
                        False)
  --selavy-simple       Only include flux density and uncertainty in returned table.
                        (default: False)
  --process-matches     Only produce data products for sources that have a selavy match.
                        (default: False)
  --debug               Turn on debug output. (default: False)
  --no-background-rms   Do not estimate the background RMS around each source. (default:
                        False)
  --find-fields         Only return the associated field for each source. (default: False)
  --clobber             Overwrite the output directory if it already exists. (default: False)
```

## Inputs

The scipt take one main input which is coordinates, either direct in the command line or using an input csv file.

### Command line: Single Coordinate
Here the format can be either in Hours or decimal degrees: 
* `"HH:MM:SS.ss +/-DD:MM:SS.ss"`
* `"DDD.ddd +/-DD.ddd"`
Note the space between the coodinates and the quotation marks.

E.g.
```
find_sources.py "22:37:5.6000 +34:24:31.90"
```
```
find_sources.py "339.2733333 34.4088611"
```

It's recommended to provide a source name using the option `--source-names`, e.g.
```
find_sources.py "22:37:5.6000 +34:24:31.90" --source-names "SN 2014C"
```


### Command line: Multiple Coordinates
Same format as above but now separate coodinates with `,`: 
* `"HH:MM:SS.ss +/-DD:MM:SS.ss,HH:MM:SS.ss +/-DD:MM:SS.ss,HH:MM:SS.ss +/-DD:MM:SS.ss"`
* `"DDD.ddd +/-DD.ddd,DDD.ddd +/-DD.ddd,DDD.ddd +/-DD.ddd"`

Note there is no space between the commas.

E.g. 
```
find_sources.py "22:37:5.6000 +34:24:31.90,22:37:5.6000 -34:24:31.90,13:37:5.6000 -84:24:31.90"
```
```
find_sources.py "339.2733333 34.4088611,154.2733333 -34.4088611,20.2733333 -54.4088611"
```

Source names can still be defined using the option `--source-names` with the same comma notation e.g.

```
find_sources.py "22:37:5.6000 +34:24:31.90,22:37:5.6000 -34:24:31.90,13:37:5.6000 -84:24:31.90" --source-names "SN 2014C,SN 2012C,SN2019B"
```

### Input CSV file
To crossmatch many coordinates it's recommended to use a csv. Instead of entering coordaintes enter the name of the csv. The `--source-names` options is not used with CSV files.

E.g. 
```
find_sources.py my_coords.csv
```

The columns `ra` and `dec` are required and can be in either of the formats shown in the command line options. `name` is also accepted and is recommended. E.g.
```
ra,dec,name
123.45,-67.89,source name
```

See `input_example.csv`.

## Examples

Search for a match to one source and create a FITS postage stamp of 5 arcminutes across. Will place the output in `example_source`.

```
find_sources.py "22:37:5.6000 +34:24:31.90" --imsize 5.0 --source-names "SN 2014C" --out-folder example_source
```

To include a png output with selavy overlay:

```
find_sources.py "22:37:5.6000 +34:24:31.90" --imsize 5.0 --source-names "SN 2014C" --out-folder example_source --create-png --png-selavy-overlay
```
Now search in Stokes V to a different directory and also include a kvis annotation file and an extra coodinate:
```
find_sources.py "22:37:5.6000 +34:24:31.90,22:37:5.6000 +44:24:31.90" --imsize 5.0 --source-names "SN 2014C,SN 2019I" --out-folder example_source_stokesv_ --create-png --png-selavy-overlay --stokesv --ann
```
Search through a csv of coordinates, make pngs, use zscale with a contrast of 0.2, create annotation and region files.:
```
find_sources.py my_coords.csv --imsize 5.0  --out-folder example_source --create-png --png-selavy-overlay --png-use-zscale --png-zscale-contrast 0.2 --ann --reg
```
