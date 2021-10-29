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


## Developer Notes

This section includes specific notes from previous developers.

### Pipeline Component & vaex

As stated in the [`Pipeline`](../../components/pipeline) component section, reading the measurements of a large pipeline run is sometimes done by using `vaex`.
In particular, `vaex` is used if the VAST Pipeline has produced a `measurements.arrow` file that contains all the measurements of a run compiled into one `arrow` file.
This is instead of VAST Tools reading in the measurements from the individual `measurements.parquet` files for every image, which can be very memory and time consuming using `pandas`.
Version 4.0 of `vaex` introduced the ability to open `parquet` files in an out-of-core context, however testing proved that the resulting dataframe was very slow to query compared to the compiled `.arrow` file.
See [this issue](https://github.com/askap-vast/vast-tools/issues/225){:target="_blank"} for more information.

Hence, as of the current VAST Tools version the decision was made to keep the `.arrow` functionality of the pipeline, but this should be revisited in the future to see if the `parquet` files could be used directly as originally intended.

### Adding new epochs

VAST Tools relies on not only having access to ASKAP images and catalogues external to the package, but also a number of metadata files stored within the package. Please refer to [this section](../newepoch) for a step-by-step guide on adding access to new data.
