# vast-tools

A space to share your hacky scripts that others may find useful.

Currently these are scripts that can just be copied and run, not a module installation yet (soon).

## Requirements
* Python 3
* pandas
* numpy
* astropy
* matplotlib
* scipy
* dropbox
* colorlog (optional)

Latest versions of above recommended.

There is a requirements.txt included in the repository that you can use to install the dependancies using
```
pip install -r requirements.txt
````

## Current Scripts
The current avaialble scripts are:

* **find\_sources.py** - A tool to swiftly search RACS and VAST Pilot data at chosen coordaintes.
    - See [FINDSOURCES.md](FINDSOURCES.md) for full instructions.
* **get\_vast\_pilot\_dbx.py** - A script to allow simpler downloading of the VAST Pilot survey from Dropbox.
    - See [VASTDROPBOX.md](VASTDROPBOX.md) for full instructions.
