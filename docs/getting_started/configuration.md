# Configuration & Data Access

This page details the configuration options after installation.

## Environment Variables

VAST Tools can read two defined system environment variables that are designed to remove the need to state the directories every time a component of VAST Tools is used.
These are defined below.

**`VAST_DATA_DIR`**  
This is the system path to the VAST Pilot data, i.e. the path which contains the EPOCHXX folders as released by the VAST team. 
The images and catalogues in this directory is used by the `Query` functionality and is not used by the pipeline.

**`PIPELINE_WORKING_DIR`**  
The path to the VAST Pipeline directory containing the pipeline runs.

## Data Access

VAST Tools requires access to the observational data to fully function. 
The sections below describe precisely what data is required for which area of VAST Tools.

### Query: VAST Pilot Data

The `Query` component of the VAST Tools, that queries the VAST Pilot survey directly, requires the data as released by the VAST team.
The data is required to be in the same directory structure as the release as shown below.
Full details of the VAST Pilot survey data release can be found on [this wiki page](https://github.com/askap-vast/vast-project/wiki/Pilot-Survey-Status-&-Data){:target="_blank"} (askap-vast GitHub membership required to access).
It is this directory that is defined by the `VAST_DATA_DIR` environment variable.

```bash
VAST_PILOT_DATA/
|-- EPOCHXX/  
|-- |-- COMBINED/  
|-- |   |-- STOKESI_IMAGES/  
|-- |   |-- STOKESI_SELAVY/  
|-- |   |-- STOKESI_RMSMAPS/  
|-- |   |-- STOKESV_IMAGES/  
|-- |   |-- STOKESV_SELAVY/  
|-- |   |-- STOKESV_RMSMAPS/  
|-- |   |-- STOKESQ_SELAVY/  
|-- |   |-- STOKESU_SELAVY/   
|-- |-- TILES/  
|-- |   |-- STOKESI_IMAGES/  
|-- |   |-- STOKESI_RMSMAPS/  
|-- |   |-- STOKESI_SELAVY/  
|-- |   |-- STOKESV_IMAGES/  
```

!!! note "Note: RACS Data"
    VAST Tools supports data from the [`Rapid ASKAP Continuum Survey (RACS)`](https://research.csiro.au/racs/){:target="_blank"}, including observations that have not been publicly released. We strongly recommend using official RACS data products where possible, but currently VAST tools supports accessing data from unreleased RACS observations (based on publicly available information from the RACS survey database), if the user has downloaded that data themselves.

#### Minimum Data Requirement

The 'find fields' option of requires requires no external data. 
Beyond this, the source catalogues are the minimum data required to perform other queries where source matching is requested.
Images (including noise and background images) are used for postage stamps and for measuring upper limits or performing forced extractions.
Hence, while source matching queries can be performed without the image, the functionality mentioned won't function without them.

### Pipeline Data

The pipeline component of VAST Tools requires access to: 

  * the directory where the output of pipeline runs is stored, which is described on the VAST Pipeline documentation [here](https://vast-survey.org/vast-pipeline/outputs/outputs/), and
  * any data directories where the observational image data is stored that the pipeline run used.

The pipeline output directory is defined by the `PIPELINE_WORKING_DIR` environment variable.

#### Minimum Data Requirement

The pipeline output directory is the minimum requirement to use the pipeline exploration.
This will allow for the parquets to be read in, and data analysis is able to be performed on these files alone.
Without access to the images no postage stamps or or image analysis can be produced by the pipeline routines.

## Containers

There is an assumption with the pipeline interaction that VAST Tools is installed and running on the same system that the VAST Pipeline is installed.
However, there are times when installing the various components in containers is preferable, which would mean they are separated.
As long as the directories are mounted and reachable between the containers there should be no problems with the functioning of VAST Tools.
No actual installed components are shared between the systems. 

!!!warning "Warning: Container Pipeline Data Paths"
    Note that system paths to the data in pipeline runs may need to be adjusted.
    This is because the paths are written to the pipeline produced parquet files and will point to the data from the perspective of the pipeline installation.
