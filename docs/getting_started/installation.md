# Installation

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
Also included is `[jupyterlab-system-monitor](https://github.com/jtpio/jupyterlab-system-monitor)` which allows for the memory and cpu usage to be monitored in the Jupyter Lab environment.

