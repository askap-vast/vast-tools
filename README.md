# vast-tools

A Python module to interact with and obtain the VAST Pilot Survey data.

## Prerequisites

Python 3 (3.8.0 tested).

Recommended to use an environment created with conda or virtualenv.

## Installation

To install the module you can run
```
git clone https://github.com/askap-vast/vast-tools.git
pip install ./vast-tools
```

There is a requirements.txt included in the repository that you can use to install the dependancies separately using
```
pip install -r requirements.txt
````

## Notebook Usage

As of version v2.0 the module has been made 'notebook friendly' and can be used interactively. See the `notebook-examples` directory in this repository for examples on how to use the module in a notebook environment. Note that for large queries it is better to use the `find_sources.py` script as pre version v2.0.

## Current Scripts
As part of the installation the following scripts are made available in your path:

* **pilot\_fields\_info.py** - A script to get basic information of VAST Pilot survey fields.
    - See [PILOTFIELDSINFO.md](PILOTFIELDSINFO.md) for full instructions.
* **find\_sources.py** - A tool to swiftly search VAST Pilot data at chosen coordinates (also supports RACS if available).
    - See [FINDSOURCES.md](FINDSOURCES.md) for full instructions and options.
* **build\_lightcurves.py** - A script to allow easy creation of source lightcurves on output of `find_sources.py`.
    - As of v2.0 `find_sources.py` can also output lightcurves. This script can be useful if you'd like to regenerate them using different settings.
    - See [BUILDLIGHTCURVES.md](BUILDLIGHTCURVES.md) for full instructions.
