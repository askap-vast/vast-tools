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

## Current Scripts
As part of the installation the following scripts are made available in your path:

* **find\_sources.py** - A tool to swiftly search VAST Pilot data at chosen coordaintes (also supports RACS if available).
    - See [FINDSOURCES.md](FINDSOURCES.md) for full instructions.
* **get\_vast\_pilot\_dbx.py** - A script to allow simpler downloading of the VAST Pilot survey from Dropbox.
    - See [VASTDROPBOX.md](VASTDROPBOX.md) for full instructions.
    
## Workflow Example

This is an example of a basic workflow to query three sources of interest. The steps are:

1. Determine which VAST Pilot fields contain the sources of interest. Do this using the `--find-fields ` mode in `find_sources.py` (see [FINDSOURCES.md](FINDSOURCES.md)).
2. Download the required fields by passing the above output to `get_vast_pilot_dbx.py` (see [VASTDROPBOX.md](VASTDROPBOX.md)).
3. Run `find_sources.py` for a second time, making sure to point to the newly downloaded data above, to gather information and cut outs of the sources and their crossmatches (see [FINDSOURCES.md](FINDSOURCES.md)).