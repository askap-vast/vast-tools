# build\_lightcurves

This script allows you to quickly build lightcurves of sources you have extracted using `find_sources.py`. As of v2.0.0 `find_sources.py` can now output lightcurve plots directly, this script can be helpful to run the lightcurve plotting again on a complete `find_sources.py` output.

Peak fluxes are used by default, integrated fluxes can be used by using the `--use-int-flux` flag.

## Running the script
Prior to running this script you should query the survey data using `find_sources.py` with the `--vast-pilot` flag set to your desired epochs. Then run `build_lightcurves FOLDER` where `FOLDER` is the output folder of the previous query.

## Usage

Most options should be self explanatory. The lightcurve plots are saved in the same directory as the input.

```
usage: build_lightcurves [-h] [--use-int-flux] [--quiet] [--debug] [--min-points MIN_POINTS]
                            [--min-detections MIN_DETECTIONS] [--mjd] [--grid]
                            [--yaxis-start {auto,0}] [--use-forced-for-limits] [--use-forced-for-all]
                            [--hide-legend] [--plot-dpi PLOT_DPI] [--nice NICE]
                            folder

positional arguments:
  folder

optional arguments:
  -h, --help            show this help message and exit
  --use-int-flux        Use the integrated flux, rather than peak flux (default: False)
  --quiet               Turn off non-essential terminal output. (default: False)
  --debug               Turn on debug output. (default: False)
  --min-points MIN_POINTS
                        Minimum number of epochs a source must be covered by (default: 2)
  --min-detections MIN_DETECTIONS
                        Minimum number of times a source must be detected (default: 1)
  --mjd                 Plot lightcurve in MJD rather than datetime. (default: False)
  --grid                Turn on the 'grid' in the lightcurve plot. (default: False)
  --yaxis-start {auto,0}
                        Define where the y axis on the lightcurve plot starts from. 'auto' will let
                        matplotlib decide the best range and '0' will start from 0. (default: 0)
  --use-forced-for-limits
                        Use the forced fits values instead of upper limits. (default: False)
  --use-forced-for-all  Use the forced fits for all datapoints. (default: False)
  --hide-legend         Don't show the legend on the final plot. (default: False)
  --plot-dpi PLOT_DPI   Specify the DPI of all saved figures. (default: 150)
  --nice NICE           Set nice level. (default: 5)
```
