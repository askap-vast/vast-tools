# vast-tools

A space to share your hacky scripts that others may find useful.

Currently these are scripts that can just be copied and run, not a module installation yet (soon).

## Requirements
* Python 3 (3.8.0)
* pandas (0.25.3)
* numpy (1.17.4)
* astropy (<4.0)
* matplotlib (3.1.2)
* scipy (1.4.0)
* dropbox (9.4.0)
* colorlog (4.0.2 ;optional)

Recommended versions, which have been tested against, are stated in brackets.

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
