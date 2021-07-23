# Introduction

VAST Tools is a python package written to streamline and enable the exploration of the VAST data products, including those produced by the VAST Pipeline.

As of the current version, VAST Tools provides the following:

  * The ability to interactively explore results from the VAST Pipeline, which includes:
      - Viewing light curves and postage stamps.
      - Performing transient analysis.
      - Crossmatching to external surveys.
      - Allowing for custom analyses to be easily performed.
  * Explore the VAST Pilot Survey footprints and check source coverage.
  * Query the VAST Pilot Survey data directly.
  * Perform forced extractions in the Pilot Survey data.
  * Search for the Sun, Moon and planets in the data, including pipeline outputs.

The package is also written such that exploration using Jupyter Notebooks is fully supported.

## Why was VAST Tools created?

VAST Tools was initially created to provide a short-term solution to allow for the exploration of the VAST Pilot Survey Phase I data, while the VAST Pipeline was being developed.
Primarily, the task was to provide a method to efficiently search the pilot data for sources of interest and to generate postage stamps.
Once the VAST Pipeline was completed, VAST Tools became an encompassing software package for everything related to the VAST Pilot Survey, 
and most importantly, providing a method to interactively explore results from the VAST Pipeline.

!!!note "Relationship to the VAST Pipeline"
    The VAST Pipeline has superseded a lot of the initial features of VAST Tools, such as querying the VAST Pilot data directly for source matches (known as a 'Query').
    While these methods still exist in the package, it is always highly recommended to use the VAST Pipeline data products as the base for any searches.

## VAST Tools Structure

There are five main components to the package which users interact with. 
These are described in greater detail in the next sections of this documentation, along with examples of their use. 
However, below is a brief summary of their function:

  * **moc**: The package includes ready made MOCs (Multi-Order Coverage maps) of the VAST Pilot surveys. 
    This component allows easy access to access MOCs at the field, epoch or pilot survey footprint level.
  * **pipeline**: Provides interaction with results from the VAST Pipeline.
  * **query**: Query the VAST Pilot survey data directly (superseded by the VAST Pipeline).
  * **source**: A representation of a VAST 'source' (astronomical object).
  * **survey**: A representation of the survey itself, from field properties to epochs.

## Scripts

From version 2.0, VAST Tools was primarily designed for usage via Notebooks, 
however some packaged scripts still remain for performing large scale queries that would be impractical inside of a notebook.
Full details can be found in the [Scripts](../scripts/build_lightcurves.md) section of the documentation.
