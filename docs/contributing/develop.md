# Developing VAST Tools

VAST Tools is hosted on the GitHub platform.
Please refer to [this section](github.md) on how to interact with the GitHub repository including how to open issues, the expected workflow, and how to create a release as a maintainer.

This page gives details on some specific details of VAST Tools when it comes to maintenance and development.

!!! note "Note: PEP 8"
    The VAST Tools codebase follows the python [PEP 8 style guide](https://pep8.org){:target="_blank"}.

## Development Checklist

When adding features to the project it is important to document and test as much of the new code as possible.
While it can be cumbersome, it really helps to maintain both the code quality and the documentation for any future developers.
Below is a checklist to refer to when adding new features or fixes to vast tools.

  - Make sure docstrings and type hints are present on any new functions and classes.
  - Write tests where possible for new code (refer to the [Tests](../tests) section).
  - If applicable, update, or create, all relevant documentation pages.
  - Update the [`CHANGELOG.md`](../../changelog) file with the new changes.

## Adding a New Epoch

VAST Tools contains packaged observing information and MOC files for each epoch of the VAST Pilot Survey that is available.
When a new epoch is ready to be added to VAST Tools the following must be completed.

### Future Epochs Warning

!!! warning 
    Thus far during development the VAST Pilot Survey Phase I has been conducted at one frequency.
    This meant that the individual tile observations from which the epochs were constructed were standardised and had the same footprint.
    For example, the field `VAST_0216-06A` had the same footprint in every epoch.
    As the pilot survey progresses it is possible that a new frequency will be used that will mean the footprints and/or standard field centres will change.
    In this event VAST Tools must be updated to:
      
      * Be able to determine which footprint of the tile is requried if the tiles have the same central name and pointing at a different frequency.
      * Update the `vasttools/data/mocs/COMBINED` and `vasttools/data/mocs/TILES` with MOCS of the new footprints.
      * Add new 'field' MOC files in `vasttools/data/mocs/` to provide new VAST Pilot Survey defined fields if necessary.
      * Update the file `vasttools/data/csv/vast_field_centres.csv` if required.
      
    In addition, users should be aware that the `max_sep` parameter in the [`Query`](../../components/query) component may need tweaking.

### Epoch Addition Steps

1. A new csv file must be placed in `vasttools/data/csvs/` in the form of `vast_epochXX_info.csv`, replacing `XX` with the two digit zero padded epoch number.
    These files contain the observing information of each individual beam that makes up the epoch.
    The example below shows the expected format.
    
    !!! example "vast_epoch09_info.csv"
        ```text
        SBID,FIELD_NAME,BEAM,RA_HMS,DEC_DMS,DATEOBS,DATEEND,NINT,BMAJ,BMIN,BPA
        11248,VAST_1724-31A,0,17:22:15.223,-30:51:39.82,2020-01-12 02:52:19.947,2020-01-12 03:04:16.583,73,15.970375,12.021243,-42.804366
        11248,VAST_1724-31A,1,17:27:08.777,-30:51:39.82,2020-01-12 02:52:19.947,2020-01-12 03:04:16.583,73,15.916086,11.722993,-41.766406
        11248,VAST_1724-31A,2,17:22:13.572,-31:54:39.6,2020-01-12 02:52:19.947,2020-01-12 03:04:16.583,73,15.960169,12.041986,-42.239253
        11248,VAST_1724-31A,3,17:27:10.428,-31:54:39.6,2020-01-12 02:52:19.947,2020-01-12 03:04:16.583,73,16.000031,12.015099,-42.539542
        ```
        The description of these columns can be found [here](../../components/survey/#fields-attributes).
    
    !!! tip "Tip: Creating the Files"
        These files were previously created manually during the processing of the VAST Pilot Survey.
        This may no longer be the case, in which case it is advised to create from the [`ASKAP_SURVEYS`](https://bitbucket.csiro.au/projects/ASKAP_SURVEYS/repos/vast/browse){:target="_blank"} repository.
        In future it is hoped VAST Tools will be directly compatible with `ASKAP_SURVEYS`.

2. Generate new MOC files and updated existing STMOC file:
    
    * A MOC of the entire epoch should be placed in `vastools/data/mocs/` with a name in the format of `VAST_PILOT_EPOCHXX.moc.fits`, replacing `XX` with the two digit zero padded epoch number.
        This is created by creating a MOC for each individual tile image in the epoch and then combining them into one MOC file.

        !!! tip "Tip: Creating a MOC from an FITS file"
            Creating a MOC from a FITS file using the default process found in [`mocpy`](https://cds-astro.github.io/mocpy/){:target="_blank"} can be slow.
            Refer to the method `vasttools.pipeline.PipeRun._create_moc_from_fits` for a slightly faster way to generate a MOC from a large FITS file.

    * Update the `VAST_PILOT.stmoc.fits` which the tiles from the new epoch.
        An STMOC should be made for each tile and then added to the existing `VAST_PILOT.stmoc.fits`.

3. Add the new epoch to the `RELEASED_EPOCHS` variable found in [`vasttools.survey`](../../reference/survey).

4. Add the new epoch to the `FIELD_FILES` variable found in [`vasttools.survey`](../../reference/survey).

5. Make sure the new epoch data is present in the standard release format if the instance of VAST Tools has access to the survey data.

## Developer Notes

This section includes specific notes from previous developers.

### Pipeline Component & vaex

As stated in the [`Pipeline`](../../components/pipeline) component section, reading the measurements of a large pipeline run is sometimes done by using `vaex`.
In particular, `vaex` is used if the VAST Pipeline has produced a `measurements.arrow` file that contains all the measurements of a run compiled into one `arrow` file.
This is instead of VAST Tools reading in the measurements from the individual `measurements.parquet` files for every image, which can be very memory and time consuming using `pandas`.
Version 4.0 of `vaex` introduced the ability to open `parquet` files in an out-of-core context, however testing proved that the resulting dataframe was very slow to query compared to the compiled `.arrow` file.
See [this issue](https://github.com/askap-vast/vast-tools/issues/225){:target="_blank"} for more information.

Hence, as of the current VAST Tools version the decision was made to keep the `.arrow` functionality of the pipeline, but this should be revisited in the future to see if the `parquet` files could be used directly as originally intended.