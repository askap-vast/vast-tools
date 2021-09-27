# Tests

VAST Tools uses the [`pytest`](https://pytest.org){:target="_blank"} framework to perform unit tests on the codebase.

The [`pytest-mock`](https://github.com/pytest-dev/pytest-mock/){:target="_blank"} plugin is also used to streamline any required mocks.

!!! note "Automatic Tests"
    The test suit is run automatically on every new pull request opened.

!!! warning "Complex Testing"
    Some aspects of the codebase are challenging to test comprehensively due to the complexity of the tasks.
    Many perform dataframe operations or require specific data to test.
    The testing introduced in version 2.0.0 tests makes sure the main components but some 'smoke tests' are present and could be improved.

## Tests Structure

The tests can be found in the `tests` directory, which contains the following files:

```terminal
tests
├── data
│   ├── psr-j2129-04-pipe-meas.csv
│   ├── psr-j2129-04-query-meas.csv
│   ├── test_images.csv
│   ├── test_measurement_pairs.csv
│   ├── test_measurements.csv
│   ├── test_measurements_vaex.csv
│   ├── test_pairs_df_result.csv
│   └── test_sources.csv
├── test_moc.py
├── test_pipeline.py
├── test_query.py
├── test_source.py
├── test_survey.py
└── test_utils.py
```

As shown above, the tests for each component are written in their own respective file.
For example all the tests for `vasttools/source.py` are located in `tests/test_source.py`.

## Test Data

!!! note "No External Data"
    The tests do not need to fetch any external data to run the tests.
    All test data is locally packaged or defined.

To be tested comprehensively, some components require representative dummy data.
While some of this is small enough to be defined within the test files, some dataframes are large enough that they are stored as CSV files in the `tests/data` directory.
These are detailed in the table below.

 
| data file                    | description                                                                               |      used in  |
|:-----------------------------|:------------------------------------------------------------------------------------------|:--------------|
|  psr-j2129-04-pipe-meas.csv  | A measurements dataframe attached to a source object created from a pipeline run.         | test_source   | 
|  psr-j2129-04-query-meas.csv | A measurements dataframe attached to a source object created from a vast tools query.     | test_source   | 
|  test_images.csv             | Images dataframe used by the dummy pipeline run.                                          | test_pipeline | 
|  test_measurement_pairs.csv  | Measurement pairs dataframe used by the dummy pipeline run.                               | test_pipeline | 
|  test_measurements.csv       | Measurements dataframe used by the dummy pipeline run.                                    | test_pipeline | 
|  test_measurements_vaex.csv  | Measurements dataframe used by the dummy pipeline run, written by vaex instead of pandas. | test_pipeline | 
|  test_pairs_df_result.csv    | Pairs dataframe used by the dummy pipeline run.                                           | test_pipeline | 
|  test_sources.csv            | Sources dataframe used by the dummy pipeline run.                                         | test_pipeline | 
 
### Dummy Pipeline Run

To test the pipeline component a dummy pipeline run was constructed.
This was done by using the test data for the VAST Pipeline, run extracting three sources and all accompanying data.
One of these sources was made sure to be PSR J2129-04.

Small dataframes that represent the data are defined directly in the `tests/test_pipeline.py` where as the larger dataframes are contained in the files detailed above.

!!! warning "Warning: CSV Storage"
    For legibility the dataframes are stored as CSV files.
    However this means that some column data types are not interpreted correctly by pandas upon reading.
    Be wary of this if any new data is added.

If in the future the VAST Pipeline is updated and new data products are added, then the test data here will need to be updated to reflect the changes.

## Running the Tests

After [installing the development dependancies](../../getting_started/installation#development-install) to the python environment, the full test suite can be run by using the following command:

```terminal
pytest -vv
```

The `-vv` is optional but recommended as it provides a more detailed verbose output.

Running the tests for a single component can be done by specifying the test file.
For example, the command to run the source tests would be:

```terminal
pytest -vv tests/test_source.py
```

A single test can be run by specifying it in the command, for example:

```terminal
pytest -vv tests/test_source.py::TestSource::test_plot_lightcurve
```

## Writing Tests

Tips on writing tests:

* Follow the style already present if you are unsure.
* Make use of [pytest fixtures](https://docs.pytest.org/en/latest/how-to/fixtures.html){:target="_blank"}.
* Use the pytest-mock plugin framework to mock required parts of the tests. For example, try this [this guide](https://medium.com/analytics-vidhya/mocking-in-python-with-pytest-mock-part-i-6203c8ad3606){:target="_blank"}.
* Write docstrings for your tests so that it is clear what is being tested.

## Tests Coverage

The [pytest-cov](https://pytest-cov.readthedocs.io/en/latest/){:target="_blank"} package is included in the dependancies and the current coverage of the tests can be seen by running the command:

```terminal
pytest --cov=vasttools tests/
```