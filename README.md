# vast-tools

A Python module to interact with and obtain the VAST Pilot Survey data.

## Prerequisites

Python 3.8+.

Recommended to use an environment created with your favourite manager (conda, virtualenv, pyenv, etc).

## Installation

To install the module you can run
```
git clone https://github.com/askap-vast/vast-tools.git
pip install ./vast-tools
```

### Development Install

`vast-tools` uses [poetry](https://python-poetry.org/docs/) to manage the dependancies. 
To install the full dependancies required for development (including the documentation) please install `poetry` as described in the linked documentation, and then perform:
```
cd vast-tools
poetry install
```

Included in the development dependancies is `jupyterlab` such that vast-tools can be easily tested in a notebook environment.
Also included is `[jupyterlab-system-monitor](https://github.com/jtpio/jupyterlab-system-monitor)` which allows for the memory and cpu usage to be monitored in the Jupyter Lab environment, along with `[jupyterlab-execute-time](https://github.com/deshaw/jupyterlab-execute-time)` which allows for cell timings to be displayed.
Please refer to the documentation in the linked repositories for configuration of these add-ons.

## Notebook Usage

As of version v2.0 the module has been made 'notebook friendly' and can be used interactively. See the `notebook-examples` directory in this repository for examples on how to use the module in a notebook environment. Note that for large queries it is better to use the `find_sources.py` script as pre version v2.0.

**Note**: Jupyter is not included in the requirements, hence please install the required packages to the environment if you wish to use a notebook interface.

## System Variables

To save specifying the data directories in every call to VAST tools there are two system variables you can set that will be read by the module:

* `VAST_DATA_DIR`: The path to the VAST Pilot data, i.e. the path which contains the `EPOCHXX` folders.
* `PIPELINE_WORKING_DIR`: The path to the VAST Pipeline directory containing the pipeline runs.

These can be overridden by specifying a `base_folder` when initialising the `Query` class, and a `project_dir` when initialising the `Pipeline` class.

## Current Scripts
As part of the installation the following scripts are made available in your path:

* **pilot\_fields\_info** - A script to get basic information of VAST Pilot survey fields.
    - See [PILOTFIELDSINFO.md](PILOTFIELDSINFO.md) for full instructions.
* **find\_sources** - A tool to swiftly search VAST Pilot data at chosen coordinates (also supports RACS if available).
    - See [FINDSOURCES.md](FINDSOURCES.md) for full instructions and options.
* **build\_lightcurves** - A script to allow easy creation of source lightcurves on output of `find_sources.py`.
    - As of v2.0 `find_sources.py` can also output lightcurves. This script can be useful if you'd like to regenerate them using different settings.
    - See [BUILDLIGHTCURVES.md](BUILDLIGHTCURVES.md) for full instructions.
