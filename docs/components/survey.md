# Survey

!!! warning "Version 2.0.0 Epoch 12 Update"
    In v2.0.0 of vast tools, what was defined as `EPOCH12` has now been redefined as `EPOCH13`.
    `EPOCH12` is now the epoch that was observed between `11x` and `13` that was processed after these observations.
    Please refer to the [VAST wiki page](https://github.com/askap-vast/vast-project/wiki/Pilot-Survey-Status-&-Data){:target="_blank"} for the full details (VAST GitHub organization membership required).

The `Survey` component offers some convenient parameters and data on the VAST Pilot Survey or the ASKAP telescope.

These include:

  * A list of the supported pilot survey epochs in VAST Tools.
  * A list of what fields are contained in each pilot survey epoch.
  * The centres of each pilot survey tile field.
  * The ASKAP observing location.

The `Survey` component also contains two classes `Fields` and `Image` that may be useful to users but are mainly used for internal purposes in other components.

## Using the Survey Component

:fontawesome-regular-file-alt: [Code reference](../../reference/survey).

Unlike the other components, some tools and functions are directly imported from the `Survey` component.
These are detailed in the section below.

### Survey Component Attributes

#### RELEASED_EPOCHS

:fontawesome-regular-file-alt: [Code reference](../../reference/survey/#vasttools.survey).

`RELEASED_EPOCHS` is a python dictionary that contains the supported epochs as a string value as the keys and the zero-padded string epoch name as the respective value.

!!! example
    ```python
    from vasttools.survey import RELEASED_EPOCHS
    
    print(RELEASED_EPOCHS)
    # output
    {'0': '00', '1': '01', '2': '02', '3x': '03x', '4x': '04x', '5x': '05x', '6x': '06x', '7x': '07x', '8': '08', '9': '09', '10x': '10x', '11x': '11x', '12': '12'}
    ```

#### FIELD_CENTRES

:fontawesome-regular-file-alt: [Code reference](../../reference/survey/#vasttools.survey).

Returns a pandas dataframe containing three columns: 

  * `field` is the name of the field in question.
  * `centre-ra` is the centre right ascension coordinate of the respective field in degrees.
  * `centre-dec` is the centre declination coordinate of the respective field in degrees.

!!! note "Note: RACS Fields"
    The field centres of all the [`Rapid ASKAP Continuum Survey (RACS)`](https://research.csiro.au/racs/){:target="_blank"} fields (low frequency component) are also included.

!!! example
    ```python
    from vasttools.survey import FIELD_CENTRES
    
    FIELD_CENTRES
    ```
    Pandas dataframe output:
    
    |      | field         |     centre-ra |   centre-dec |
    |-----:|:--------------|--------------:|-------------:|
    |    0 | VAST_0012+00A |   3.10175     |   0.00377094 |
    |    1 | VAST_0012-06A |   3.10156     |  -6.29821    |
    |  ... | ...           |  ...          |  ...         |
    | 1014 | RACS_2359-25A |   0.000611873 | -25.1363     |
    | 1015 | RACS_2359-43A |   0.00902694  | -43.8906     |

### Survey Component Functions

#### get_askap_observing_location

:fontawesome-regular-file-alt: [Code reference](../../reference/survey/#vasttools.survey.get_askap_observing_location).

This function returns the location of the ASKAP telescope in the form of an [astropy EarthLocation instance](https://docs.astropy.org/en/stable/api/astropy.coordinates.EarthLocation.html){:target="_blank"}.
There are no arguments available to the function.
The values returned are the longitude, latitude and height.

!!! example
    ```python
    from vasttools.survey import get_askap_observing_location

    askap_loc = get_askap_observing_location()
    print(askap_loc)
    #output
    (-2556451.02811979, 5096903.35306894, -2848173.88980533) m
    ```

#### get_fields_per_epoch_info

:fontawesome-regular-file-alt: [Code reference](../../reference/survey/#vasttools.survey.get_fields_per_epoch_info).

This function returns a pandas dataframe containing the field that are present in each epoch along with the SBID value and the date of the observation.
There are no arguments available to the function.

!!! example
    ```python
    from vasttools.survey import get_fields_per_epoch_info

    field_info = get_fields_per_epoch_info()
    field_info
    ```
    Dataframe output example. 
    Note that the `EPOCH` and `FIELD_NAME` columns form the multiindex of the dataframe.
    
    |  **EPOCH** | **FIELD_NAME**    |   SBID | DATEOBS                 |
    |:-------|---------------|-------:|:------------------------|
    | 0 | RACS_0131+37A |   8537 | 2019-04-21 04:07:50.563 |
    |  | RACS_0202+37A |   8537 | 2019-04-21 04:26:45.254 |
    |  | RACS_0233+37A |   8537 | 2019-04-21 04:11:29.501 |
    |  | RACS_0249+31A |   8537 | 2019-04-21 04:57:26.611 |
    |  | RACS_0303+37A |   8537 | 2019-04-21 04:42:00.922 |
    |   ...    |      ...     |   ... | ... |
    | 12 | VAST_2146-43A |  15773 | 2020-08-30 14:44:40.327 |
    |    | VAST_2209-50A |  15774 | 2020-08-30 14:58:46.355 |
    |    | VAST_2131-62A |  15775 | 2020-08-30 15:12:42.431 |
    |    | VAST_2246-50A |  15776 | 2020-08-30 15:26:38.506 |
    |    | VAST_2220-62A |  15777 | 2020-08-30 15:40:34.582 |

    To select all fields in epoch 8:
    
    ```python
    epoch8_fields = field_info.loc['8']
    ```
    Note that epoch values are strings.


### Fields Class

!!! warning "Warning: `Fields` Custom Usage"
    The `Fields` class is not intended for custom user usage and is designed to help internal processes.
    However it is detailed here as it may prove useful in some situations and during development.

:fontawesome-regular-file-alt: [Code reference](../../reference/survey/#vasttools.survey.Fields).

The `Fields` class is designed to act as a representation of the fields contained in an epoch.
It is primarily used by the `Query` component to perform the act of finding matches to provided sky coordinates.
The class holds details of the fields at the individual beam level rather than the 'tile field' as a whole.

#### Initialising a Fields Instance

:fontawesome-regular-file-alt: [Code reference](../../reference/survey/#vasttools.survey.Fields.__init__).

A `Fields` instance is initialised by declaring the epoch that is wanted to be represented.
This is demonstrated below.

!!! example
    Initialise a `Fields` instance for Epoch 8.
    ```python
    from vasttools.survey import Fields

    epoch_8 = Fields('8')
    ```

#### Fields Attributes

:fontawesome-regular-file-alt: [Code reference](../../reference/survey/#vasttools.survey.Fields)

The following attributes are available on a `Fields` instance.

**`Fields.fields`**  
This is a pandas dataframe which contains the information of each individual beam that makes up each tile which in turn makes up the complete epoch.
The columns of the dataframe are:

  * `SBID` The SBID of the observation.
  * `FIELD_NAME` The field name of which the beam belongs to (there are 36 beams in each tile).
  * `BEAM` The designated beam number.
  * `RA_HMS` The right ascension coordinate of the centre of the beam pointing, in sexagesimal format.
  * `DEC_DMS` The declination coordinate of the centre of the beam pointing, in sexagesimal format.
  * `DATEOBS` The start date and time of the observation.
  * `DATEEND` The end date and time of the observation.
  * `NINT` The number of integrations.
  * `BMAJ` The size of the major axis of the restoring beam in arcsec.
  * `BMIN` The size of the minor axis of the restoring beam in arcsec.
  * `BPA` The position angle of the restoring beam in degrees.

!!! example "Example: `Fields.fields`"
    ```python
    epoch_8.fields
    ```
    Output dataframe:
    
    |    |   SBID | FIELD_NAME    |   BEAM | RA_HMS       | DEC_DMS      | DATEOBS                 | DATEEND                 |   NINT |    BMAJ |    BMIN |     BPA |
    |---:|-------:|:--------------|-------:|:-------------|:-------------|:------------------------|:------------------------|-------:|--------:|--------:|--------:|
    |  0 |  11160 | VAST_1724-31A |      0 | 17:22:15.223 | -30:51:39.82 | 2020-01-11 02:56:47.027 | 2020-01-11 03:08:43.663 |     73 | 13.0281 | 10.4522 | 69.3692 |
    |  1 |  11160 | VAST_1724-31A |      1 | 17:27:08.777 | -30:51:39.82 | 2020-01-11 02:56:47.027 | 2020-01-11 03:08:43.663 |     73 | 12.7945 | 10.3603 | 68.4207 |
    |  2 |  11160 | VAST_1724-31A |      2 | 17:22:13.572 | -31:54:39.6  | 2020-01-11 02:56:47.027 | 2020-01-11 03:08:43.663 |     73 | 12.8277 | 10.3425 | 68.373  |
    |  3 |  11160 | VAST_1724-31A |      3 | 17:27:10.428 | -31:54:39.6  | 2020-01-11 02:56:47.027 | 2020-01-11 03:08:43.663 |     73 | 13.0037 | 10.4943 | 68.4667 |
    |  4 |  11160 | VAST_1724-31A |      4 | 17:17:26.352 | -29:47:57.8  | 2020-01-11 02:56:47.027 | 2020-01-11 03:08:43.663 |     73 | 13.1051 | 10.4401 | 70.2277 |
    |  ... |  ... | ... |      ... | ... | ...  | ... | ... |    ... | ... | ... | ... |


**`Fields.direction`**  
An astropy [`SkyCoord`](https://docs.astropy.org/en/stable/api/astropy.coordinates.SkyCoord.html){:target="_blank"} instance containing the centres of all the beams in the epoch.

!!! example
    ```python
    epoch_8.direction
    ```
    Output:
    ```console
    <SkyCoord (ICRS): (ra, dec) in deg
        [(260.56342917, -30.86106111), (261.78657083, -30.86106111),
         (260.55655   , -31.911     ), ..., (329.39785   , -62.90618889),
         (329.59647083, -61.86028056), (329.78020833, -60.81385   )]>
    ```

### Image Class

!!! warning "Warning: Custom Usage"
    The `Image` class is not intended for custom user usage and is designed to help internal processes.
    However it is detailed here as it may prove useful in some situations and during development.

:fontawesome-regular-file-alt: [Code reference](../../reference/survey/#vasttools.survey.Image).

The `Image` class provides a template of representing an individual ASKAP image, allowing for easy access to properties and common tasks.

#### Initialising an Image Instance

:fontawesome-regular-file-alt: [Code reference](../../reference/survey/#vasttools.survey.Image.__init__).

!!! warning "Warning: Pilot Survey Assumption"
    By default the class was written to load images from the VAST Pilot Survey.
    Hence, the minimum initialisation options attempts to load the image from the directory defined by the `base_folder` argument, and it is assumed the VAST Pilot data is present in the standard release directory structure.

The initialisation of an `Image` instance requires four inputs:

  * `field` The field name of the image as a string.
  * `epoch` The epoch of the image as a string.
  * `stokes` The stokes value of the image.
  * `base_folder` The base directory of the VAST Pilot Survey.

See the code reference for further options.

!!! example "Example: Initialising a Pilot Survey Image"
    Load the stokes I image of field 'VAST_1724-31A' from epoch 8.
    ```python
    from vasttools.survey import Image
    
    my_image = Image('VAST_1724-31A', '8', 'I', '/path/to/VAST/release')
    ```

!!! tip "Tip: Loading any ASKAP image"
    The `Image` class can load any ASKAP image if the further optional arguments are used.
    This method is used in the `Pipeline` component to load pipeline images.
    The important option is the `path` option to point directory to the FITS file.
    When this is used the `base_folder` directory is ignored (though something still needs to be entered for the parameter).
    In addition, the parameters such as field and epoch can be set to any values as it is not required to be accurate for a non pilot survey image.
    
    For example:
    ```python
    my_custom_image = Image('field1', '1', 'I', 'None', path='/path/to/the/image.fits)
    ```
    It is also possible to provide the path of the corresponding rms image path at initialisation:
    ```python
    my_custom_image = Image(..., rmspath='/path/to/the/rmsimage.fits)
    ```

#### Image Attributes

Once initialised, an `Image` instance has access to a number of attributes, such as the image data, header, field and epoch.
Please refer to the [Code reference](../../reference/survey/#vasttools.survey.Image) section for a full list of attributes available.

#### Image Methods

There are two methods available with the `Image` class.

**`Image.get_rms_img`**  
!!! warning "Warning: Pilot Survey Images Only"
    This method will only work correctly with Pilot Survey images loaded from the expected release directory.
    If using with an external ASKAP image, use the `rmspath` parameter when initialising instead of this method.
    
This method checks for the presense of the rms image, and if it is available the rms image information, including the data and header are attached to the instance.

!!! example
    ```python
    my_image.get_rms_img()
    ```

**`Image.measure_coord_pixel_values`**  
A method to measure the pixel values of the image at the provided sky coordinates.
The coordinates should be provided in the form of an astropy `SkyCoord` instance.
By default the pixel values are taken from the image data.
This can be changed to the rms data by setting `rms=True`.
The values are returned in the form of a numpy array and will be in the units of the image (very commonly Jy).
Be aware that `NaN` values can be returned if the coordinate falls within the null area of the image.

!!! example
    Measuring the pixel values in the image data at the provided coordinates:
    ```python
    import astropy.units as u
    
    from astrpy.coordinates import SkyCoord
    
    my_coords = SkyCoord(
      [206.058966, 207.711250, 206.438710], [-57.278384, -57.645140, -57.486640], unit=(u.deg, u.deg)
    )
    
    pixel_values = my_image.measure_coord_pixel_values(my_coords)
    ```
    Measuring the pixel values from the rms image instead:
    ```python
    pixel_values = my_image.measure_coord_pixel_values(my_coords, rms=True)
    ```

!!! warning "Warning: Out of range coordinates"
    As of the current version this method is only used in conjuction with the `Query` component, and hence it is known that all provided coordinates are within the image boundary.
    Adjustments need to be made to the method in order to filter out-of-range coordinates if any are provided.
    Without adjusting erroneous results could be returned because of pixel wrapping.
    
    For example, if a coordinate is not contained in the image, the calculated pixel x,y position could be `[-500, 500]`.
    Hence, the x pixel will wrap backwards from 0 and will likely provide a value that is not `NaN`, which will be incorrect.
