# build\_lightcurves.py

This script allows you to quickly build lightcurves of sources you have extracted using `find_sources.py`.

The script will output a csv file containing:
* The start and end datetimes of the observation
* Integrated flux density and associated uncertainty
* The image noise in the local region
* A flag stating whether the measurement is a detection or an upper limit.

By default the script will also plot the lightcurve, although this can be disabled using the `--no-plotting` flag

# Running the script
Prior to running this script you should query the survey data using `find_sources.py` with the `--vast-pilot` flag set to `all`. Then run `build_lightcurves.py FOLDER` where `FOLDER` is the output folder of the previous query.

## Usage

Most options should be self explanatory. The lightcurve plots and csv files are saved in the same directory as the input

```
usage: build_lightcurves.py [-h] [--no-plotting] [--quiet] [--debug]
                            [--min-points MIN_POINTS]
                            [--min-detections MIN_DETECTIONS] [--mjd] [--grid] [--nice NICE]
                            folder

positional arguments:
  folder

optional arguments:
  -h, --help            show this help message and exit
  --no-plotting         Write lightcurves to file without plotting (default: False)
  --quiet               Turn off non-essential terminal output. (default: False)
  --debug               Turn on debug output. (default: False)
  --min-points MIN_POINTS
                        Minimum number of epochs a source must be covered by (default:
                        2)
  --min-detections MIN_DETECTIONS
                        Minimum number of times a source must be detected (default: 1)
  --mjd                 Plot lightcurve in MJD rather than datetime. (default: False)
  --grid                Turn on the 'grid' in the lightcurve plot. (default: False)
  --nice NICE           Set nice level. (default: 5)
```