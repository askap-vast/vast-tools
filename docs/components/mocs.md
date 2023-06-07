# MOCs

MOC stands for Multi-Order Coverage map and is a standard in astronomy for showing survey coverage (see [here](http://cds.unistra.fr/adass2018/){:target="_blank"} for a tutorial on both HiPS and MOCS).
MOCs of the VAST pilot survey are packaged into VAST Tools and are available to load for use within an analysis.

!!!info "Info: mocpy"
    All MOCs are loaded as `mocpy` objects. 
    Refer to the documentation of `mocpy` [here](https://cds-astro.github.io/mocpy/){:target="_blank"} to learn more about what can be achieved with the package, including how to plot a MOC onto a sky map.

The MOCs available to load are in VAST Tools are:

  * **`Tile`**  
    These are MOCs of the individual tiles that construct the overall pilot survey observation.
    
  * **`Field`**  
    These are groups of tiles that construct a unique field used in the pilot survey.
    For example, the VAST pilot survey phase 1 used a survey footprint of 6 distinct fields.
    
  * **`Epoch`**  
    The entire footprint observed for a single epoch of the pilot survey.
    
  * **`Full Pilot STMOC`**  
    The entire VAST Pilot survey contained in a MOC that contains Space and Time information - a STMOC.
    See [this document](https://www.ivoa.net/documents/stmoc/20190515/NOTE-stmoc-1.0-20190515.pdf){:target="_blank"} for information on STMOCs.
    
## Using the MOC Component

!!!info "Info: MOCs Example Notebook"
    A notebook example of using the MOCs component can be found in the example notebooks section [here](../../notebook-examples/using-vast-mocs-example/).
    
!!!info "Info: VAST Pilot Survey Details"
    Full details of the design of the VAST Pilot survey can be found these wiki pages (askap-vast GitHub membership required to access):
    
      * [Pilot Survey Planning](https://github.com/askap-vast/vast-project/wiki/Pilot-Survey-Planning){:target="_blank"}.
      * [Pilot Survey Phase II Planning](https://github.com/askap-vast/vast-project/wiki/Pilot-Survey-Phase-II-Planning){:target="_blank"}.
      * [Pilot Survey Status & Data](https://github.com/askap-vast/vast-project/wiki/Pilot-Survey-Status-&-Data){:target="_blank"}.

The MOC component is known as `VASTMOCS` in VAST Tools a `VASTMOCS` instance can be initialised with:

!!!example
    ```python
    from vasttools.moc import VASTMOCS

    vast_moc = VASTMOCS()
    ```

No arguments are required as all the MOCs can be loaded from this object.

### Available Methods

The following methods are available with the `VASTMOCS` class.

!!!info "Info: Code Reference"
    Each method below has a link to the Code Reference section which provides full details of the method, including the arguments.

#### load_pilot_tile_moc

:fontawesome-regular-file-alt: [Code reference](../../reference/moc/#vasttools.moc.VASTMOCS.load_pilot_tile_moc).

This loads the MOC of a single tile, returning a `mocpy.MOC` instance.

!!!example
    Loading the MOC for field VAST_0012-06A.
    ```python
    tile_moc = vast_moc.load_pilot_tile_moc('VAST_0012-06A')
    ```

#### load_pilot_field_moc

:fontawesome-regular-file-alt: [Code reference](../../reference/moc/#vasttools.moc.VASTMOCS.load_pilot_field_moc).

This loads the MOC of a VAST pilot survey field, returning a `mocpy.MOC` instance.

!!!example
    Loading the MOC for field 1:
    ```python
    field_moc = vast_moc.load_pilot_field_moc('1')
    ```

#### load_pilot_epoch_moc

:fontawesome-regular-file-alt: [Code reference](../../reference/moc/#vasttools.moc.VASTMOCS.load_pilot_epoch_moc).

This loads the MOC of a VAST pilot survey epoch, returning a `mocpy.MOC` instance.
Enter as string with no zero padding on the epoch.

!!!example
    Loading the MOC for epoch 7x:
    ```python
    epoch_moc = vast_moc.load_pilot_epoch_moc('7x')
    ```

#### load_pilot_stmoc

:fontawesome-regular-file-alt: [Code reference](../../reference/moc/#vasttools.moc.VASTMOCS.load_pilot_stmoc).

This loads the STMOC of all the VAST pilot observations, returning a `mocpy.MOC` instance.

!!!example
    Loading the STMOC:
    ```python
    stmoc = vast_moc.load_pilot_stmoc()
    ```

#### load_survey_footprint

:fontawesome-regular-file-alt: [Code reference](../../reference/moc/#vasttools.moc.VASTMOCS.load_survey_footprint).

This loads a MOC of the footprint of either the pilot or full VAST survey.

!!!example
    Loading the survey footprint MOC:
    ```python
    pilot_moc = vast_moc.load_survey_footprint('pilot')
    full_moc = vast_moc.load_survey_footprint('full')
    ```

#### query_vizier_vast_pilot

:fontawesome-regular-file-alt: [Code reference](../../reference/moc/#vasttools.moc.VASTMOCS.query_vizier_vast_pilot).

This searches the provided Vizier table for sources that are in the VAST pilot survey epoch 1 footprint, returning a `astropy.table.Table` instance.

!!!example
    To search for matches in the SUMSS catalogue (Vizier id: `VIII/81B`) the command would be:
    ```python
    vast_sumss_sources = vast_moc.query_vizier_vast_pilot('VIII/81B')
    ```
