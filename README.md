# vast-tools

A Python module to interact and obtain the VAST Pilot Survey data.

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

* **find\_sources.py** - A tool to swiftly search RACS and VAST Pilot data at chosen coordaintes.
    - See [FINDSOURCES.md](FINDSOURCES.md) for full instructions.
* **get\_vast\_pilot\_dbx.py** - A script to allow simpler downloading of the VAST Pilot survey from Dropbox.
    - See [VASTDROPBOX.md](VASTDROPBOX.md) for full instructions.
