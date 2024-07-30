# Installation

!!!warning "Warning: Data Access"
    For full functionality, VAST Tools requires access to the VAST Pilot data in addition to the 
    VAST Pipeline output and any related data directories, i.e. any images and catalogues not part of the VAST Pilot data previously mentioned.
    Be mindful of this when selecting where to install VAST Tools. 
    Further details can be found in the [Configuration and Data Access](configuration.md) page.
    

It is strongly recommend installing VAST Tools in an isolated virtual environment 
(e.g. using [Miniconda](https://docs.conda.io/en/latest/miniconda.html){:target="_blank"}, 
[Virtualenv](https://virtualenv.pypa.io/en/latest/){:target="_blank"}, or 
[venv](https://docs.python.org/3/library/venv.html){:target="_blank"}). 
This will keep the dependencies separated from the system-wide Python installation.

A minimum version of Python 3.8 is also required.

For example, with Miniconda, a virtual environment would be created by the following commands:

```terminal
conda create --name <environment-name> python=3.8
conda activate <environment-name>
```

With the python environment activated, VAST Tools can then be installed.

!!!warning "Warning: Select the Release"
    The default branch of VAST Tools is the development branch, which may not be stable.
    When installing the package please use the release tag for the version that is desired to be installed.
    This is done by placing `@<tag>` at the end of the URL. For example, the URL to use for version 3.0.0 would be:
    ```terminal
    https://github.com/askap-vast/vast-tools.git@v3.0.0
    ```
VAST Tools can be installed using `pip` directly from GitHub by using the command below, being sure to specify the version.

```terminal
pip install git+https://github.com/askap-vast/vast-tools.git@v3.0.0
```

VAST Tools will now be installed in the virtual environment.
The next step is to refer to the [Configuration and Data Access](configuration.md) page for details on what data is required by VAST Tools and how to configure it.

!!!note "Note: Jupyter"
    Jupyter products such as lab or notebook are not included in the dependencies.
    Please install these to the environment if you wish to use them.
    Instructions can be found [here](https://jupyter.org/install){:target="_blank"}.

## Development Install

`vast-tools` uses [poetry](https://python-poetry.org/docs/) to manage the dependancies. 
To install the full dependancies required for development (including the documentation) please install `poetry` as described in the linked documentation, 
and then clone the VAST Tools repository from GitHub and install using poetry:

```terminal
git clone https://github.com/askap-vast/vast-tools.git
cd vast-tools
poetry install
```

As the development branch `dev` is the default, this will be the branch installed, ready to begin any development.

Included in the development dependancies is [`jupyterlab`](https://jupyter.org){:target="_blank"} such that vast-tools can be easily tested in a notebook environment.
Also included is [`jupyterlab-system-monitor`](https://github.com/jtpio/jupyterlab-system-monitor){:target="_blank"} which allows for the memory and cpu usage to be monitored in the Jupyter Lab environment.
