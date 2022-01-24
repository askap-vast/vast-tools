# Query

!!! warning "Version 2.0.0 Epoch 12 Update"
    In v2.0.0 of vast tools, what was defined as `EPOCH12` has now been redefined as `EPOCH13`.
    `EPOCH12` is now the epoch that was observed between `11x` and `13` that was processed after these observations.
    Please refer to the [VAST wiki page](https://github.com/askap-vast/vast-project/wiki/Pilot-Survey-Status-&-Data){:target="_blank"} for the full details (VAST GitHub organization membership required).

The `Query` component of VAST Tools is used to query the VAST Pilot Survey data directly.

!!!warning "Deprecation Warning"
    The completion of the [VAST Pipeline](https://vast-survey.org/vast-pipeline/){:target="_blank"} means that some functionality of this component has been superseded.
    To perform tasks such as:
      
      * searching for sources in the VAST Pilot Survey,
      * crossmatching the VAST Pilot Survey to other catalogues,
      * obtaining lightcurves and postage stamps of sources,
      * and performing transient searches,
    
    output from the VAST Pipeline should always be used.
    
    Users may find features such as the forced fitting and stokes V search options still useful.

The base function of a `Query` is to search the VAST Pilot catalogues for source matches to the queried positions.
If no source is found in an epoch then the upper limit or forced fit measurement is returned.
In full, the `Query` component as the ability to:

  * determine whether provided coordinates are in the VAST Pilot survey, and if so, which field(s) the source present in (no data required),
  * find source matches to provided coordinates,
  * perform forced fit and upper limit measurements
  * attempt to construct a lightcurve for the source,
  * create postage stamps for sources, with overlays and save as fits or png files,
  * search stokes V data,
  * search for planets and the Sun and Moon.

!!!warning "Warning: Data Access"
    It is assumed that the machine that is running VAST Tools has access to the VAST Pilot Survey release output.
    Refer to the [Configuration & Data Access](../../getting_started/configuration/) page for more information.

!!!tip "Tip: `find_sources` script"
    It is not ideal to perform large queries in a notebook environment.
    For these situations there is the [`find_sources`](../../scripts/find_sources/) script that can be used
    to perform a large search in a non-interactive manner.

## Using the Query Component

!!! info "Info: Query Notebook Example"
     An example notebook of using the Query component can be found in the example notebooks section [here](../../notebook-examples/source-search-example/).

:fontawesome-regular-file-alt: [Code reference](../../reference/query/#vasttools.query.Query).

A `Query` instance can be imported from `vasttools.query`:

!!!example
    ```python
    from vasttools.query import Query
    ```

!!!warning "Warning: Running a Query inside a script"
    The main `Query` functions use multiprocessing to speed up large queries. 
    Users who attempt to call a query from their own scripts may encounter a Dask RuntimeError which, due to the nature of the error, cannot be nicely caught with the `vasttools` module. 
    The solution is to ensure that all calls to `Query` functions are protected within a `if __name__ == '__main__'` statement, i.e.
    ```python
    from vasttools.query import Query
    
    if __name__ == '__main__':
        my_query = Query(source_names=["PSR J2129-04"])
        my_query.find_sources()
    ```
        

### Constructing a Query

The `Query` is defined on the initialisation of the instance. 
In other words, when the `Query` is first defined all the desired `Query` arguments must be entered so the `Query` is initialised correctly.

!!! info "Info: Query Initialisation Arguments"
    Please refer to the code reference [here](../../reference/query/#vasttools.query.Query.__init__) for details on all the arguments that can be passed to the query.

For example, the code below demonstrates how to create a Query object that will use the RA and Dec coordinate `21:29:45.29 -04:29:11.9` to perform a search.

!!! example
    Note that when entering coordinates to query these need to be represented by an astropy 
    [`SkyCoord`](https://docs.astropy.org/en/stable/api/astropy.coordinates.SkyCoord.html){:target="_blank"} object.
    Creating the `SkyCoord` instance is also performed in the example.
    ```python
    from vasttools.query import Query
    from astropy.coordinates import SkyCoord
    
    my_coord = SkyCoord("21h29m45.29s -04d29m11.9s")
    
    my_query = Query(coords=my_coord)
    ```

See the [following section](#entering-the-coordinates-to-query) for details on how to enter coordinates or how to instead use object names as the coordinate input.

!!! note "Note: Query Execution"
    Initialising the `Query` does not automatically execute the search.
    Refer to the [Running a Query](#running-a-query) section on this document page for full details on running a `Query`.

#### Entering the Coordinates to Query

Coordinates passed to the `coords` parameter in the `Query` object must be entered as an astropy [`SkyCoord`](https://docs.astropy.org/en/stable/api/astropy.coordinates.SkyCoord.html){:target="_blank"} object.
The SkyCoord object can be a singular coordinate or contain multiple coordinates.

!!! example "Example: Multiple Coordinates"
    ```python
    from vasttools.query import Query
    from astropy.coordinates import SkyCoord
    
    coords_list = ["21h29m45.29s -04d29m11.9s", "21h18m50.70s -57d38m42.5s"]
    
    my_coord = SkyCoord(coords_list)
    
    my_query = Query(coords=my_coord)
    ```

The `source_names` parameter can be used to enter names of the sources alongside the coordinates.
The future results of the query will then be accessible using these source names.
If not source names are provided then they will be automatically named using the degrees representation of the coordinates in the format `ra_dec`.

!!! example "Example: Multiple Coordinates"
    ```python
    from vasttools.query import Query
    from astropy.coordinates import SkyCoord
    
    coords_list = ["21h29m45.29s -04d29m11.9s", "21h18m50.70s -57d38m42.5s"]
    my_source_names = ["PSR J2129-04", "SN 2012dy"]
    
    my_coord = SkyCoord(coords_list)
    
    my_query = Query(coords=my_coord, source_names=my_source_names)
    ```

It is also possible to only define the `source_names` parameter.
If this is done then the Query will use the [`CDS SIMBAD`](https://simbad.u-strasbg.fr/simbad/){:target="_blank"} service to attempt to fetch the coordinates for the requested sources.

!!! example "Example: Source Names Only"
    ```python
    from vasttools.query import Query
    from astropy.coordinates import SkyCoord
    
    my_source_names = ["PSR J2129-04", "SN 2012dy"]
    
    my_query = Query(source_names=my_source_names)
    ```

!!! warning "Warning: Do Not Mix Methods"
    If `source_names` are entered then all names must have a matching coordinate. 
    I.e., if an extra source name is entered this will not be searched for using `SIMBAD` and will return an error.

#### Crossmatching Options

There are two options that control how the crossmatching to the VAST catalogues is performed.

**`crossmatch_radius`**  
This is the radius, in arcsec, to which a VAST Pilot source is considered a match to the search coordinate.

**`max_sep`**  
The maximum distance, in degrees, from the centre of an individual beam (36 beams make up an ASKAP tile) to determine if a match is possible.

!!!note "Note: Field Selection"
    When the query is made, if the source is found in two separate overlapping fields, all fields will be noted as a match but field used will be that which has
    the coordinate location closest to the centre of its respective field.

**`search_around_coordinates`**
All matches within the `crossmatch_radius` are returned instead of just the closest matched component or island.

#### Data Selection Options

These `Query` keyword arguments control the data that is used to perform the query or the data returned.

**`epochs`**  
A comma-separated list of epochs (entered as a string) to search.
Do not use zero padded values when entering the epochs.
The values `all` and `all-vast` are also valid, selecting Epoch 0 + VAST data and just VAST data, respectively.

!!! example
    ```python
    my_query = Query(..., epochs="1,2,6x,8,9")
    ```

**`stokes`**  
Select which Stokes parameter data to search in.
Valid entries are `I`, `V`, `Q` and `U`.

**`use_tiles`**  
Switch to using the catalogues and images of the `TILE` type instead of the `COMBINED` (default).
Refer to [this wiki page](https://github.com/askap-vast/vast-project/wiki/Pilot-Survey-Status-&-Data){:target="_blank"} for an explanation on the two types (askap-vast GitHub membership required to access).

**`use_islands`**  
Query the island catalogues produced by the [`selavy`](https://www.atnf.csiro.au/computing/software/askapsoft/sdp/docs/current/analysis/selavy.html){:target="_blank"} source finder instead of the component catalogues (default).

**`matches_only`**  
Only return results that have a source match. 
I.e. no forced fit or upper limit measurements will be returned.

#### Planet Search

Planets are included in the `Query` search by using the `planets` keyword argument.

**`planets`**
A python list of planets to include in the `Query` search.
Valid entries are: `mercury`, `venus`, `mars`, `jupiter`, `saturn`, `uranus`, `neptune`, `pluto`, `sun` and `moon`.

!!! example "Example: Including Planets"
    Including Jupiter and the Moon in the search.
    ```python
    my_query = Query(..., planets=["jupiter", "moon"])
    ```

#### RMS & Forced Fits

These `Query` arguments control the RMS estimation and forced fitting.

!!! info "Info: Forced Fitting"
    The forced fitting is performed using the package [`forced_phot`](https://github.com/askap-vast/forced_phot){:target="_blank"}.

**`no_rms`**
Set to `True` to not make background `rms` measurements for each requested position per epoch.

**`forced_fits`**
When set to `True` a forced fit will be produced at each queried location in each requested epoch where possible.

!!! warning "Warning: Forced Fits Runtime"
    Be patient with the execution of a `Query` when using the `forced_fits` option as it can take a little while to complete.

!!! info "Info: Forced Fits Results"
    See the [Forced Fits Results](#forced-fits-results) section for details on how the results of the forced fitting are added to the source result.

**`forced_cluster_threshold`**
Passed to the `cluster_threshold` parameter in [`forced_phot`](https://github.com/askap-vast/forced_phot){:target="_blank"}.

**`forced_allow_nan`**
Passed to the `allow_nan` parameter in [`forced_phot`](https://github.com/askap-vast/forced_phot){:target="_blank"}.

#### Checking Settings

Once the `Query` has been defined the settings can be checked by viewing the `Query.settings` attribute as demonstrated below.

!!! example "Example: Check Settings"
    ```python
    my_coords = SkyCoord([206.058966, 319.711250, 322.438710], [0.278384, -57.645140, -4.486640], unit=(u.deg, u.deg))
    my_source_names = ["1AXG J134412+0016", "SN 2012dy", "PSR J2129-0429"]

    my_query = Query(
        coords=coords_to_query,
        source_names=source_names,
        matches_only=True, 
        epochs="all-vast", 
        crossmatch_radius=10., 
        output_dir='source-search-example-output', 
        base_folder='/import/ada1/askap/PILOT/release'
    )

    my_query.settings

    {'epochs': ['1', '2', '3x', '4x', '5x', '6x', '7x', '8', '9', '10x', '11x'],
     'stokes': 'I',
     'crossmatch_radius': <Angle 10. arcsec>,
     'max_sep': 1.0,
     'islands': False,
     'tiles': False,
     'no_rms': False,
     'matches_only': True,
     'search_around': False,
     'sort_output': False,
     'forced_fits': False,
     'output_dir': 'source-search-example-output'}
    ```

### Running a Query

The query does not automatically run once initialised.
There are two methods available to launch the query that are detailed below.
See the [following section](#using-the-query-results) for details on how to use the results from each method.

!!! note "Note: No Arguments or Returns"
    All the settings for the `Query` have already been stored in the object, hence these methods have no possible arguments.
    In addition, when the queries are launched all the results are saved to the `Query` object itself.
    Hence nothing is returned when the the run methods are used.

#### find_fields

This method will only locate the fields that the contains the queried sources/coordinates.
The results will be saved to the attribute `fields_df` in the form of a pandas dataframe.

!!! example
    ```python
    my_query.find_fields()
    ```

#### find_sources

This method will build upon `find_fields` and perform the source crossmatching from the `selavy` catalogues.
The results will be stored in the attribute `.results`, in the format of a pandas series containing `vasttools.source.Souce` objects with the source name as the key.

!!! example
    ```python
    my_query.find_sources()
    ```

### Using the Query Results

#### find_fields Results

Each row in the results dataframe, saved to `Query.fields_df`, represents the result of the queried source for a particular epoch.
For example, if the source `PSR J2129-04` should be present in 5 epochs there will be 5 row.
The columns of the dataframe are:

  * **`name`** The name of the source/coordinate queried.
  * **`ra`** The right ascension of the source/coordinate in degrees.
  * **`dec`** The declination of the source/coordinate in degrees.
  * **`skycoord`** The `SkyCoord` representation of the coordinate.
  * **`stokes`** The stokes parameter of this field.
  * **`fields`** The complete list of fields the source can be found in.
  * **`primary_field`** Which VAST Pilot field is considered the main field of the source. Defined by being the field where the source is the closest to the respective fields centre.
  * **`epoch`** The epoch this row is for.
  * **`field`** The best field for this epoch (may not be the primary field if that field is not present in the epoch).
  * **`sbid`** The ASKAP SBID of the field observation.
  * **`dateobs`** The date and time of the observation.
  * **`planet`** Set to `True` if the source/coordinate is a planet that has been requested to search for.

These results can be written to a file by using the `Query.write_find_fields()` method.

#### find_sources Results

The `find_sources` method creates VAST Tools `Source` objects for each queried coordinate/source and saves these to a pandas `Series` object that is accessible as `Query.results`.
Below shows an example of what a results series looks like when `find_sources` has completed.

!!! example "Example: Contents of `my_query.results`"
    ```python
    name
    1AXG J134412+0016    <vasttools.source.Source object at 0x7f181dea4...>
    SN 2012dy            <vasttools.source.Source object at 0x7f182938d...>
    PSR J2129-0429       <vasttools.source.Source object at 0x7f182938d...>
    Name: name, dtype: object
    ```
    _Note that the `name` values are just strings_.

To access a specific source of the results, the source name is used as a key as demonstrated below.

!!! example "Example: Accessing a Specific Source"
    Accessing the source object for SN 2012dy.
    ```python
    my_query.results['SN 2012dy']
    ```

Please now refer to the [Source page](source.md) of this documentation to learn more about `vasttools.source.Source` objects, which details how to produce lightcurves, postage stamps and more.

#### Forced Fits Results

If forced fits has been used in the query these results will be placed in the `measurements` attribute of the `Source` object (refer to [here](source.md#sourcemeasurements) for more details).
The following columns will be added:

  * `f_island_id` An id to the forced extraction island given by [`forced_phot`](https://github.com/askap-vast/forced_phot){:target="_blank"}.
  * `f_component_id` An id to the forced extraction component given by `forced_phot`.
  * `f_ra_deg_cont` The right ascension coordinate of the extraction in degrees.
  * `f_dec_deg_cont` The declination coordinate of the extraction in degrees.
  * `f_flux_peak` The measured peak flux of the extraction in mJy/beam.
  * `f_flux_peak_err` The error of the measured peak flux.
  * `f_flux_int` The measured integrated flux of the extraction in mJy.
    Note this should be equal to the peak flux.
  * `f_flux_int_err` The error of the integrated flux.
  * `f_chi_squared_fit` The $\chi^{2}$ value of the fit.
  * `f_rms_image` The rms at the measured position as calculated by `forced_phot` in mJy.
  * `f_maj_axis` The major axis of the Gaussian used in arcsec.
  * `f_min_axis` The minor axis of the Gaussian used in arcsec.
  * `f_pos_ang` The position angle of the Gaussian used in degrees.

## `find_sources` script

VAST Tools provides the script `find_sources` that can be accessed from the command line.
It is a script that allows for the use of the `Query` functionality but in a command line environment.
This is most useful for large queries that would be non-ideal to perform in a notebook environment.
See the [`find_sources`](../../scripts/find_sources/) documentation page for full details.
