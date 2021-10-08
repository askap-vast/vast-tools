# Tools

The `Tools` sub-package is a mishmash of useful VAST-related functions that do not have a home elsewhere. It can be used for:

  * Filtering VAST data using gravitational wave skymaps and MOCs
  
## Using the Tools Component

:fontawesome-regular-file-alt: [Code reference](../../reference/tools).  

### Tools Component Functions

#### skymap2moc
:fontawesome-regular-file-alt: [Code reference](../../reference/tools/#vasttools.tools.skymap2moc).

This function returns a MOC of the credible level of a provided skymap.
The arguments to this function are the path to the skymap file and the requested credible region.

!!! example "Example: Converting a gravitational wave skymap to MOC"
    Build a MOC of the 90% credible region of GW170817
    ```python
    from vasttools.tools import skymap2moc
    GW170817_moc = skymap2moc('gw170817.fits.gz', 0.9)
    ```

#### find_in_moc
:fontawesome-regular-file-alt: [Code reference](../../reference/tools/#vasttools.tools.find_in_moc).

This function the indices of sources that are contained within a given MOC
The arguments to this function are the MOC, the source DataFrame and an optional bool flagging whether the dataframe is from the pipeline.
The value returned is a numpy array.

!!! note "Note: Coordinate Columns Names"
    The function assumes that the coordinate columns are defined as `ra` and `dec`.
    If the `pipe` flag is set to `True`, the function will assume the columns to be named `wavg_ra` and `wavg_dec`.

!!! example "Example: Finding which sources are contained in a MOC"
    Build a MOC of the 90% credible region of GW170817
    ```python
    from vasttools.tools import find_in_moc
    idx = find_in_moc(GW170817_moc, source_df, pipe=False)
    ```
    
#### add_credible_levels
:fontawesome-regular-file-alt: [Code reference](../../reference/tools/#vasttools.tools.add_credible_levels).

This function calculates the smallest credible region a source is contained in and adds it as a column, named `credible_level`, to the source dataframe in-place.
The arguments to this function are the path to the skymap file, the source DataFrame and an optional bool flagging whether the dataframe is from the pipeline.

!!! note "Note: Coordinate Columns Names"
    The function assumes that the coordinate columns are defined as `ra` and `dec`.
    If the `pipe` flag is set to `True`, the function will assume the columns to be named `wavg_ra` and `wavg_dec`.

!!! example "Example: Adding the credible level to sources dataframe"
    Build a MOC of the 90% credible region of GW170817
    ```python
    from vasttools.tools import add_credible_levels
    add_credible_levels('gw170817.fits.gz', source_df, pipe=False)
    ```
