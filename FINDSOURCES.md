# find\_sources

This script allows you to quickly query data from the RACS and VAST Pilot surveys on provided coordinates, either through the command line or using a csv list.

This includes Stokes I and Stokes V (except RACS, negative sources included).

The outputs are/can be:
* Postage stamp fits images of each location.
* DS9 region file or kvis annotation file of selavy sources withing postage stamp.
* Figure png plots of postage stamp, with overlaid selavy sources and synthesised beam.
* Per source measurments CSV file containing all the matched measurements, and forced extractions if selected.

# Running on your own machine
**No local data is required to run the `--find-fields` option which will find which fields that contain your sources of interest.** To create postage FITS files or PNG images, a copy of the survey data is required locally.

You must tell the script where your data is and as of v2.0 this data must follow the same directory structure and naming scheme of the VAST Pilot release. The easiest method is to set the environment variable `VAST_DATA_DIR` to the data path which will be read by the module, for example on bash:
```
export VAST_DATA_DIR=/path/to/my-pilot-data/
```
If this variable is not set you can define the path to this data using the following option when running the script:
```
--base-folder /path/to/my-pilot-data/
```

If you are running `find_sources` on your own machine we recommend first using the `--find-fields` flag, downloading the relevant fields and then re-running the script as normal.

## Running on Nimbus
You can access running the script on the vast-data instance by selecting the terminal in your Jupyter Hub session.

See [this wiki page](https://github.com/askap-vast/vast-project/wiki/Nimbus:-SSH,-Downloads,-Pipeline-&-Jupyter-Hub) for information on Nimbus.

## Warning!
* The default crossmatching uses **components**. Check your results to see if the component is part of an island (`has_sibling` column will = 1) and query the island catalogue, using the `island_id` if you need further information on your source.

## RACS
If you are running this script on ada or Nimbus you also have access to RACS data to search for sources. Remember that RACS is not a VAST data product and you *MUST* let Tara or David know that you intend to use RACS. Also keep in mind that this is not final quality RACS data, there are some issues with the data so please be aware when interpreting results. Currently only Stokes I data is available.

RACS can be included by specifying `--epochs all`, or you can only select RACS using `--epochs 0`. If you would like to query all VAST epochs (and exclude RACS) then use `--epochs all-vast`.

If you are using your own installation RACS is considered to be `EPOCH00` in the directory structure and follows the same naming structure as the VAST files but with `RACS` as the survey name in place of `VAST`. Please check the data structure on Nimbus if you are unsure.

## Usage

Most options should be self explanatory. See examples below on how to run the script.

All output is placed in an output directory of which the name can be set with the option `--out-folder`.

Can be run on any Stokes parameter, but only one at a time.
```
usage: find_sources [-h] [--coords COORDS] [--source-names SOURCE_NAMES] [--ncpu NCPU] [--epochs EPOCHS]
                       [--imsize IMSIZE] [--maxsep MAXSEP] [--out-folder OUT_FOLDER]
                       [--crossmatch-radius CROSSMATCH_RADIUS] [--use-tiles] [--islands] [--base-folder BASE_FOLDER]
                       [--stokes {I,Q,U,V}] [--quiet] [--forced-fits]
                       [--forced-cluster-threshold FORCED_CLUSTER_THRESHOLD] [--forced-allow-nan]
                       [--crossmatch-only] [--selavy-simple] [--process-matches] [--debug] [--no-background-rms]
                       [--planets PLANETS] [--find-fields] [--search-around-coordinates] [--clobber] [--sort-output]
                       [--nice NICE] [--crossmatch-radius-overlay] [--no-fits] [--plot-dpi PLOT_DPI] [--create-png]
                       [--png-selavy-overlay] [--png-linear-percentile PNG_LINEAR_PERCENTILE] [--png-use-zscale]
                       [--png-zscale-contrast PNG_ZSCALE_CONTRAST] [--png-hide-beam] [--png-no-island-labels]
                       [--png-no-colorbar] [--png-disable-autoscaling] [--ann] [--reg] [--lightcurves]
                       [--lc-use-int-flux] [--lc-no-plotting] [--lc-min-points LC_MIN_POINTS]
                       [--lc-min-detections LC_MIN_DETECTIONS] [--lc-mjd] [--lc-start-date LC_START_DATE]
                       [--lc-grid] [--lc-yaxis-start {auto,0}] [--lc-use-forced-for-limits]
                       [--lc-use-forced-for-all] [--lc-hide-legend]

optional arguments:
  -h, --help            show this help message and exit
  --coords COORDS       Right Ascension and Declination in quotes. Can be formatted as "HH:MM:SS [+/-]DD:MM:SS"
                        (e.g. "12:00:00 -20:00:00") or decimal degrees (e.g. "12.123 -20.123"). Multiple coordinates
                        are supported by separating with a comma (no space) e.g. "12.231 -56.56,123.4 +21.3".
                        Finally you can also enter coordinates using a .csv file. See example file for format.
                        (default: None)
  --source-names SOURCE_NAMES
                        Only for use when entering coordaintes via the command line. State the name of the source
                        being searched. Use quote marks for names that contain a space. For multiple sources
                        separate with a comma with no space, e.g. "SN 1994N,SN 2003D,SN 2019A". (default: )
  --ncpu NCPU           Number of cpus to use in queries (default: 2)
  --epochs EPOCHS       Select the VAST Pilot Epoch to query. Epoch 0 is RACS. All available epochs can be queried
                        using 'all' or all VAST Epochs using 'all-vast'. Otherwise enter as a comma separated list
                        with no spaces, e.g. '1,2,3x,4x'. (default: 1)
  --imsize IMSIZE       Edge size of the postagestamp in arcmin (default: 30.0)
  --maxsep MAXSEP       Maximum separation of source from beam centre in degrees. (default: 1.5)
  --out-folder OUT_FOLDER
                        Name of the output directory to place all results in. (default:
                        find_sources_output_20200904_23:59:27)
  --crossmatch-radius CROSSMATCH_RADIUS
                        Crossmatch radius in arcseconds (default: 15.0)
  --use-tiles           Use the individual tiles instead of combined mosaics. (default: False)
  --islands             Search islands instead of components. (default: False)
  --base-folder BASE_FOLDER
                        Path to base folder if using default directory structure. Not required if the
                        `VAST_DATA_DIR` environment variable has been set. (default: None)
  --stokes {I,Q,U,V}    Select the Stokes parameter. (default: I)
  --quiet               Turn off non-essential terminal output. (default: False)
  --forced-fits         Perform forced fits at the locations requested. (default: False)
  --forced-cluster-threshold FORCED_CLUSTER_THRESHOLD
                        Multiple of `major_axes` to use for identifying clusters, when performing forced fits.
                        (default: 1.5)
  --forced-allow-nan    When used, forced fits are attempted even when NaN values are present. (default: False)
  --crossmatch-only     Only run crossmatch, do not generate any fits or png files. (default: False)
  --selavy-simple       Only include flux density and uncertainty in returned table. (default: False)
  --process-matches     Only produce data products for sources with a selavy match. (default: False)
  --debug               Turn on debug output. (default: False)
  --no-background-rms   Do not estimate the background RMS around each source. (default: False)
  --planets PLANETS     Also search for solar system objects. Enter as a comma separated list, e.g.
                        'jupiter,venus,moon'. Allowed choices are: ['mercury', 'venus', 'mars', 'jupiter', 'saturn',
                        'uranus', 'neptune', 'sun', 'moon'] (default: [])
  --find-fields         Only return the associated field for each source. (default: False)
  --search-around-coordinates
                        Return all crossmatches within the queried crossmatch radius.Plotting options will be
                        unavailable. (default: False)
  --clobber             Overwrite the output directory if it already exists. (default: False)
  --sort-output         Place results into individual source directories within the main output directory. (default:
                        False)
  --nice NICE           Set nice level. (default: 5)
  --crossmatch-radius-overlay
                        A circle is placed on all PNG and region/annotation files to represent the crossmatch
                        radius. (default: False)
  --no-fits             Do not save the FITS cutouts. (default: False)
  --plot-dpi PLOT_DPI   Specify the DPI of all saved figures. (default: 150)
  --create-png          Create a png of the fits cutout. (default: False)
  --png-selavy-overlay  Overlay selavy components onto the png image. (default: False)
  --png-linear-percentile PNG_LINEAR_PERCENTILE
                        Choose the percentile level for the png normalisation. (default: 99.9)
  --png-use-zscale      Select ZScale normalisation (default is 'linear'). (default: False)
  --png-zscale-contrast PNG_ZSCALE_CONTRAST
                        Select contrast to use for zscale. (default: 0.1)
  --png-hide-beam       Select to not show the image synthesised beam on the plot. (default: False)
  --png-no-island-labels
                        Disable island lables on the png. (default: False)
  --png-no-colorbar     Do not show the colorbar on the png. (default: False)
  --png-disable-autoscaling
                        Do not use the auto normalisation and instead apply scale settings to each epoch
                        individually. (default: False)
  --ann                 Create a kvis annotation file of the components. (default: False)
  --reg                 Create a DS9 region file of the components. (default: False)
  --lightcurves         Create lightcurve plots. (default: False)
  --lc-use-int-flux     Use the integrated flux, rather than peak flux (default: False)
  --lc-no-plotting      Write lightcurves to file without plotting (default: False)
  --lc-min-points LC_MIN_POINTS
                        Minimum number of epochs a source must be covered by (default: 2)
  --lc-min-detections LC_MIN_DETECTIONS
                        Minimum number of times a source must be detected (default: 0)
  --lc-mjd              Plot lightcurve in MJD rather than datetime. (default: False)
  --lc-start-date LC_START_DATE
                        Plot lightcurve in days from some start date, formatted as YYYY-MM-DD HH:MM:SS or any other
                        form that is accepted by pd.to_datetime() (default: None)
  --lc-grid             Turn on the 'grid' in the lightcurve plot. (default: False)
  --lc-yaxis-start {auto,0}
                        Define where the y axis on the lightcurve plot starts from. 'auto' will let matplotlib
                        decide the best range and '0' will start from 0. (default: 0)
  --lc-use-forced-for-limits
                        Use the forced fits values instead of upper limits. (default: False)
  --lc-use-forced-for-all
                        Use the forced fits for all datapoints. (default: False)
  --lc-hide-legend      Don't show the legend on the final lightcurve plot. (default: False)
```

## Inputs

To run the script needs at least some coordinates, or a planet to search for. Coordinates are entered using the `--coords` parameter as demonstrated below, while planets can be specified using the `--planet` parameter.

### Command line: Single Coordinate
Here the format can be either in Hours or decimal degrees: 
* `"HH:MM:SS.ss +/-DD:MM:SS.ss"`
* `"DDD.ddd +/-DD.ddd"`
Note the space between the coodinates and the quotation marks.

E.g.
```
find_sources --coords "22:37:5.6000 +34:24:31.90"
```
```
find_sources --coords "339.2733333 34.4088611"
```

It's recommended to provide a source name using the option `--source-names`, e.g.
```
find_sources --coords "22:37:5.6000 +34:24:31.90" --source-names "SN 2014C"
```


### Command line: Multiple Coordinates
Same format as above but now separate coodinates with `,`: 
* `"HH:MM:SS.ss +/-DD:MM:SS.ss,HH:MM:SS.ss +/-DD:MM:SS.ss,HH:MM:SS.ss +/-DD:MM:SS.ss"`
* `"DDD.ddd +/-DD.ddd,DDD.ddd +/-DD.ddd,DDD.ddd +/-DD.ddd"`

Note there is no space between the commas.

E.g. 
```
find_sources --coords "22:37:5.6000 +34:24:31.90,22:37:5.6000 -34:24:31.90,13:37:5.6000 -84:24:31.90"
```
```
find_sources --coords "339.2733333 34.4088611,154.2733333 -34.4088611,20.2733333 -54.4088611"
```

Source names can still be defined using the option `--source-names` with the same comma notation e.g.

```
find_sources --coords "22:37:5.6000 +34:24:31.90,22:37:5.6000 -34:24:31.90,13:37:5.6000 -84:24:31.90" --source-names "SN 2014C,SN 2012C,SN2019B"
```

### Input CSV file
To crossmatch many coordinates it's recommended to use a csv. Instead of entering coordinates, enter the name of the csv. The `--source-names` option is not used with CSV files.

E.g. 
```
find_sources --coords my_coords.csv
```

The columns `ra` and `dec` are required and can be in either of the formats shown in the command line options. `name` is also accepted and is recommended. E.g.
```
ra,dec,name
123.45,-67.89,source name
```

## Outputs
The following files are or can be produced (for tiles the `combined` will be replaced with `tile`):

* `{source_name}_measurements.csv` - csv file containing the crossmatch results per source. For detections, the source information included is taken direct from the selavy catalogues (see the [selavy documentation](https://www.atnf.csiro.au/computing/software/askapsoft/sdp/docs/current/analysis/postprocessing.html#component-catalogue)). All fluxes are in mJy. Upper limits and forced fits (if both/either are selected) are also included here.
* `{source_name}.EPOCH{NN}.{STOKES}_combined.fits` - Cutout FITS file of the source referenced in the name (if requested).
* `{source_name}.EPOCH{NN}.{STOKES}_combined.png` - Matplotlib png figure of the above cutout of the source referenced in the name (if requested).
* `{source_name}.EPOCH{NN}.{STOKES}_combined.ann` - Kvis annotation file for use with the FITS file (if requested).
* `{source_name}.EPOCH{NN}.{STOKES}_combined.reg` - DS9 region file for use with the FITS file (if requested).
* `{source_name}_lc.png` - Lightcurve png plot (if requested).
* `find_fields_result.csv` - Output of the find fields option containing the input sources and the matched VAST Pilot Survey field (find-fields only).

`X` refers to the out-folder name.

## Examples

Search for a match to one source and create a FITS postage stamp of 5 arcminutes across. Will place the output in `example_source`.

```
find_sources "22:37:5.6000 +34:24:31.90" --imsize 5.0 --source-names "SN 2014C" --out-folder example_source
```

To include a png output with selavy overlay:

```
find_sources "22:37:5.6000 +34:24:31.90" --imsize 5.0 --source-names "SN 2014C" --out-folder example_source --create-png --png-selavy-overlay
```
Now search in Stokes V to a different directory and also include a kvis annotation file and an extra coodinate:
```
find_sources "22:37:5.6000 +34:24:31.90,22:37:5.6000 +44:24:31.90" --imsize 5.0 --source-names "SN 2014C,SN 2019I" --out-folder example_source_stokesv_ --create-png --png-selavy-overlay --stokes="V" --ann
```
Search through a csv of coordinates, make pngs, use zscale with a contrast of 0.2, create annotation and region files.:
```
find_sources my_coords.csv --imsize 5.0  --out-folder example_source --create-png --png-selavy-overlay --png-use-zscale --png-zscale-contrast 0.2 --ann --reg
```
