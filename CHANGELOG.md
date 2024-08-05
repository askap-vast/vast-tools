# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), with an added `List of PRs` section and links to the relevant PRs on the individal updates. This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased](https://github.com/askap-vast/vast-pipeline/compare/v3.1.1...HEAD)

#### Added

- Support for Python 3.10 [#334](https://github.com/askap-vast/vast-tools/pull/334)
- Add access to epoch 63 [#563](https://github.com/askap-vast/vast-tools/pull/563)

#### Changed

- Minor changes to docstring formatting throughout based on updated mkdocs versions [#334](https://github.com/askap-vast/vast-tools/pull/334)
- Minor changes for matplotlib 3.7: add angle kwarg to Ellipse and change matplotlib.pyplot.cm.get_cmap to matplotlib.colormaps.get_cmap [#334](https://github.com/askap-vast/vast-tools/pull/334)
- Refreshed dependencies - major changes are python 3.10, mkdocs (and related packages), astropy v5 and matplotlib v3.7 [#334](https://github.com/askap-vast/vast-tools/pull/334)
- Update emoji names to reflect latest fontawesome naming scheme[#334](https://github.com/askap-vast/vast-tools/pull/334)
- Fixed minor typos in docs and docstrings [#334](https://github.com/askap-vast/vast-tools/pull/334)
- Updated old contact information [#334](https://github.com/askap-vast/vast-tools/pull/334)
- Updated github actions to latest versions [#562](https://github.com/askap-vast/vast-tools/pull/562)
- Updated 2023 workshop notebook - fixed Query options and general cleanup [#558](https://github.com/askap-vast/vast-tools/pull/558)

#### Fixed

- Directly compare stmoc times, avoiding conversion to JD, and replace equality requirement with `isclose` [#334](https://github.com/askap-vast/vast-tools/pull/334)

#### Removed

- Removed lightgallery functionality from docs [#334](https://github.com/askap-vast/vast-tools/pull/334)

#### List of PRs

- [#334](https://github.com/askap-vast/vast-tools/pull/334): fix, docs, feat: Dependency refresh including python 3.10 support and corresponding minor updates
- [#563](https://github.com/askap-vast/vast-tools/pull/563): feat: Add access to epoch 63
- [#562](https://github.com/askap-vast/vast-tools/pull/562): fix: Updated github actions to latest versions
- [#558](https://github.com/askap-vast/vast-tools/pull/558): docs: Updated 2023 workshop notebook - fixed Query options and general cleanup


## [3.1.1](https://github.com/askap-vast/vast-tools/releases/v3.1.1) (2024-07-30)

#### Added

- Added 2023 workshop notebook to docs [#552](https://github.com/askap-vast/vast-tools/pull/552)

#### Changed

- Updated author email addresses [#553](https://github.com/askap-vast/vast-tools/pull/553)
- Explicitly require setuptools in pyproject.toml [#553](https://github.com/askap-vast/vast-tools/pull/553)
- Pin mkdocs=1.3.1 [#553](https://github.com/askap-vast/vast-tools/pull/553)
- Force jinja2==3.0.3 [#551](https://github.com/askap-vast/vast-tools/pull/551)

#### Fixed

- Temporary fix for pydantic version breaking vaex [#553](https://github.com/askap-vast/vast-tools/pull/553)
- Downgraded jinja2 to fix bug in docs deployment [#551](https://github.com/askap-vast/vast-tools/pull/551)

#### Removed

#### List of PRs

- [#553](https://github.com/askap-vast/vast-tools/pull/553): fix, dep: Pin versions to fix dependency issues, update author email addresses
- [#551](https://github.com/askap-vast/vast-tools/pull/551): fix, dep: Downgrade jinja2 to fix bug in docs deployment
- [#552](https://github.com/askap-vast/vast-tools/pull/552): feat: Add 2023 workshop notebook to docs

## [3.1.0](https://github.com/askap-vast/vast-tools/releases/v3.1.0) (2024-07-26)

#### Added

- Add epoch 61 [#537](https://github.com/askap-vast/vast-tools/pull/537)
- Add epochs 58-60 [#535](https://github.com/askap-vast/vast-tools/pull/535)
- Added extra logging in Query to improve clarity of data selection [#515](https://github.com/askap-vast/vast-tools/pull/515)
- Added support for post-processed data [#515](https://github.com/askap-vast/vast-tools/pull/515)
- Added epoch 40 as a base epoch to allow query of full VAST extragalactic fields [#522](https://github.com/askap-vast/vast-tools/pull/522)
- Added support for epochs 42-51 [#522](https://github.com/askap-vast/vast-tools/pull/522)

#### Changed

- Convert all Source names to strings [#537](https://github.com/askap-vast/vast-tools/pull/537)
- Updated README [#535](https://github.com/askap-vast/vast-tools/pull/535)
- Changed default behaviour to using post-processed TILES data [#515](https://github.com/askap-vast/vast-tools/pull/515)

#### Fixed

- Fix errors when creating a Source with an int name [#537](https://github.com/askap-vast/vast-tools/pull/537)
- Fix incorrect warnings when using COMBINED data [#538](https://github.com/askap-vast/vast-tools/pull/538)
- Fix forced fitting of compressed fits files [#534](https://github.com/askap-vast/vast-tools/pull/534)
- Fix checking length of HDUl that should have been fixed by #526 [#528](https://github.com/askap-vast/vast-tools/pull/528)
- Disable auto-scaling for show_png_cutout to allow users to specify the scaling [#527](https://github.com/askap-vast/vast-tools/pull/527)
- Fix open_fits handling of unprocessed files [#526](https://github.com/askap-vast/vast-tools/pull/526)
- Update open_fits to check HDU list contents rather than the file extension [#524](https://github.com/askap-vast/vast-tools/pull/524)

#### Removed

#### List of PRs
- [#537](https://github.com/askap-vast/vast-tools/pull/537): fix: Convert all Source names to strings
- [#538](https://github.com/askap-vast/vast-tools/pull/538): fix: Fix incorrect warnings when using COMBINED data
- [#537](https://github.com/askap-vast/vast-tools/pull/537): feat: Add epoch 61
- [#535](https://github.com/askap-vast/vast-tools/pull/535): feat: Add epochs 58-60 and updated README
- [#534](https://github.com/askap-vast/vast-tools/pull/534): fix: Fix forced fitting of compressed fits files 
- [#515](https://github.com/askap-vast/vast-tools/pull/515): fix, docs, feat: Support post-procesed data, make it the default and improve clarity of what data has been selected in a query.
- [#528](https://github.com/askap-vast/vast-tools/pull/528): fix: Fix checking length of HDUl that should have been fixed by #526
- [#527](https://github.com/askap-vast/vast-tools/pull/527): fix: Disable auto-scaling for show_png_cutout to allow users to specify the scaling
- [#526](https://github.com/askap-vast/vast-tools/pull/526): fix: Fix open_fits handling of unprocessed files
- [#524](https://github.com/askap-vast/vast-tools/pull/524): fix: Update open_fits to check HDU list contents rather than the file extension
- [#522](https://github.com/askap-vast/vast-tools/pull/522): fix, feat: Add support for extrgalactic queries and epochs 42-51

## [3.0.1](https://github.com/askap-vast/vast-tools/releases/v3.0.1) (2023-09-24)

#### Changed

- Bumped `mkdocstrings` to version `0.17.0` to avoid `mkdocstrings` error [#496](https://github.com/askap-vast/vast-tools/pull/496).
- Pinned `jinja2` to version `3.0.3` to avoid `mkdocs_autoref` error [#496](https://github.com/askap-vast/vast-tools/pull/496).
- Update github workflows to latest versions [#496](https://github.com/askap-vast/vast-tools/pull/496)
- Exit nicely when invalid survey is requested for SkyView contour plot [#497](https://github.com/askap-vast/vast-tools/pull/497)

#### Fixed

- Fixed docs build process [#496](https://github.com/askap-vast/vast-tools/pull/496).

#### List of PRs

- [#496](https://github.com/askap-vast/vast-tools/pull/496): fix: Update github workflows to latest versions
- [#497](https://github.com/askap-vast/vast-tools/pull/497): feat: Exit nicely when invalid survey is requested for SkyView contour plot

## [3.0.0](https://github.com/askap-vast/vast-tools/releases/v3.0.0) (2023-09-02)

#### Added

- Raise exception when simbad name query returns more names than there are objects queried [#490](https://github.com/askap-vast/vast-tools/pull/490)
- Added open_fits function to correctly handle compressed image HDUs [#480](https://github.com/askap-vast/vast-tools/pull/480)
- Added epoch 41 [#470](https://github.com/askap-vast/vast-tools/pull/470)
- Added warning when querying corrected data [#465](https://github.com/askap-vast/vast-tools/pull/465)
- Added epochs 38, 39 and 40 [#466](https://github.com/askap-vast/vast-tools/pull/466)
- Added epoch 37 [#458](https://github.com/askap-vast/vast-tools/pull/458)
- Added epoch 36 [#455](https://github.com/askap-vast/vast-tools/pull/455)
- Added epochs 34 and 35 [#454](https://github.com/askap-vast/vast-tools/pull/454)
- Added automatic generation of resource (csvs/pickles) paths [#431](https://github.com/askap-vast/vast-tools/pull/431)
- Added `SkyCoord` pickle functionality (creation, loading and implementation) [#431](https://github.com/askap-vast/vast-tools/pull/431)
- Added access to epoch 33 [#434](https://github.com/askap-vast/vast-tools/pull/434)
- Added "search_all_fields" option to Query and find_sources, which returns all data available at the source location rather than just that associated with the closest/best field [#418](https://github.com/askap-vast/vast-tools/pull/418)
- Added option to specify which dask scheduler to use to Query, PipeRun, PipeAnalysis objects, and argument to find_sources.py [#430](https://github.com/askap-vast/vast-tools/pull/430)
- Added access to epoch 32 [#429](https://github.com/askap-vast/vast-tools/pull/429)
- Added access to epoch 31 [#427](https://github.com/askap-vast/vast-tools/pull/427)
- Added access to epoch 30 [#419](https://github.com/askap-vast/vast-tools/pull/419)
- Added access to epoch 27 [#414](https://github.com/askap-vast/vast-tools/pull/414)
- Added access to epochs 22, 23, 24, 25, 26, 28 (RACS-high), 29 (RACS-low-2) [#406](https://github.com/askap-vast/vast-tools/pull/406)
- Added `_validate files` to `query.Query` class to ensure that only accessible data is queried [#370](https://github.com/askap-vast/vast-tools/pull/370)
- Added logging to `pipeline.PipeRun` [#383](https://github.com/askap-vast/vast-tools/pull/383)
- Added `tools.wise_color_color_plot` to create WISE color-color plots [#379](https://github.com/askap-vast/vast-tools/pull/379)
- Added access to epoch 21 [#351](https://github.com/askap-vast/vast-tools/pull/351)
- Added check for existence of requested data in `Query` init [#346](https://github.com/askap-vast/vast-tools/pull/346/)
- Added pylint github action for pull requests and flake8 dev dependency [#338](https://github.com/askap-vast/vast-tools/pull/338).
- Added observing frequency to all fields csv files [#311](https://github.com/askap-vast/vast-tools/pull/311)
- Added OBSERVED_EPOCHS variable for epochs that have been observed but not yet released [#311](https://github.com/askap-vast/vast-tools/pull/311)
- Added access to epochs up 17, 18, 19 [#311](https://github.com/askap-vast/vast-tools/pull/311)
- Added BASE_EPOCHS variable [#311](https://github.com/askap-vast/vast-tools/pull/311)
- Added tools.offset_postagestamp_axes which allows users to display postagestamps with offset (rather than absolute) coordinates [#315](https://github.com/askap-vast/vast-tools/pull/315)
- Added Dask RuntimeError warning to query.md docs [#317](https://github.com/askap-vast/vast-tools/pull/317)
- Added tools for new epoch addition: add datetimes to fits images create fields csv files, create and update MOCs [#298](https://github.com/askap-vast/vast-tools/pull/298)

#### Changed

- Changed opening of all fits files to use open_fits function [#480](https://github.com/askap-vast/vast-tools/pull/480)
- Changed default behaviour of Image.measure_coord_pixel_values to have `img=False` [#476](https://github.com/askap-vast/vast-tools/pull/476)
- Changed `Fields.direction` to be loaded from pickle rather than generated at each call [#431](https://github.com/askap-vast/vast-tools/pull/431)
- Updated RACS dec limit to +50 [#438](https://github.com/askap-vast/vast-tools/pull/438)
- Updated Query._get_epochs to allow lists and ints [#421](https://github.com/askap-vast/vast-tools/pull/421)
- Bumped pytest and lint github workflow from ubuntu-18.04 -> ubuntu-20.04 [#425](https://github.com/askap-vast/vast-tools/pull/425)
- Changed path generation to allow for image/rms/background files to contain ".conv" [#410](https://github.com/askap-vast/vast-tools/pull/410)
- Changed handling of data product generation from epoch-based to index-based [#403](https://github.com/askap-vast/vast-tools/pull/403)
- Updated GitHub actions Gr1N/setup-poetry to v7 [#385](https://github.com/askap-vast/vast-tools/pull/385)
- Updated numpy to ~1.22.1 [#380](https://github.com/askap-vast/vast-tools/pull/380)
- Removed epoch 12 warning [#361](https://github.com/askap-vast/vast-tools/pull/361)
- Updated RELEASED_EPOCHS to include epochs 14 and 20 [#351](https://github.com/askap-vast/vast-tools/pull/351)
- Enabled access to full TILES data [#325](https://github.com/askap-vast/vast-tools/pull/325)
- Allow query epochs to be specified as list [#327](https://github.com/askap-vast/vast-tools/pull/327).
- Changed plot legend to show frequency rather than selavy/forced [#311](https://github.com/askap-vast/vast-tools/pull/311)
- General changes throughout to allow observing frequency to propagate through a query [#311](https://github.com/askap-vast/vast-tools/pull/311)
- Updated access to field centres csvs to distinguish between low/mid bands [#311](https://github.com/askap-vast/vast-tools/pull/311)
- Updated find_sources.py to automatically query all epochs [#311](https://github.com/askap-vast/vast-tools/pull/311)
- Updated handling of RACS epochs [#311](https://github.com/askap-vast/vast-tools/pull/311)
- Updated default values of query epochs and imsize [#311](https://github.com/askap-vast/vast-tools/pull/311)
- Allow users to specify whether to write generated MOCs, and to use arbitrary STMOC [#323](https://github.com/askap-vast/vast-tools/pull/323)
- Changed source.Source.make_png to create postagestamps using coordinate offsets [#315](https://github.com/askap-vast/vast-tools/pull/315)
- Minor documentation edits [#298](https://github.com/askap-vast/vast-tools/pull/298)

#### Fixed

- Exit nicely when users try to use measurement pairs but they do not exist [#478](https://github.com/askap-vast/vast-tools/pull/478)
- Ensure that full image data does not persist when creating FITS cutouts from compressed HDUs [#483](https://github.com/askap-vast/vast-tools/pull/483)
- Fixed missing RMS measurements for non-detections [#476](https://github.com/askap-vast/vast-tools/pull/476)
- Removed duplicates in field matching [#472](https://github.com/askap-vast/vast-tools/pull/472)
- Fix read_selavy to force all flux & flux_err values to be positive [#463](https://github.com/askap-vast/vast-tools/pull/463)
- Fix planet matching to correctly handle field names with/without 'A' suffix [#453](https://github.com/askap-vast/vast-tools/pull/453)
- Correctly handle empty df in Query.init_sources() [#453](https://github.com/askap-vast/vast-tools/pull/453)
- Updated VASTMOCS to correctly handle fields with/without 'A' suffix [#452](https://github.com/askap-vast/vast-tools/pull/452)
- Updated github pytest workflow to use Gr1N/setup-poetry@v8 [#449](https://github.com/askap-vast/vast-tools/pull/449)
- Updated default value of Query.max_sep to 1.5 degrees [#440](https://github.com/askap-vast/vast-tools/pull/440)
- Correctly handle show_all_png_cutouts(save=True) file naming [#437](https://github.com/askap-vast/vast-tools/pull/437)
- Ensure selected_fields is defined for all field matching cases [#435](https://github.com/askap-vast/vast-tools/pull/435)
- Updated Query._get_epochs to exit nicely when no epochs are available [#421](https://github.com/askap-vast/vast-tools/pull/421)
- Update match_planet_to_field to handle empty groups [#416](https://github.com/askap-vast/vast-tools/pull/416)
- Added `corrected_data` to `Source` class to ensure correct image paths are used [#412](https://github.com/askap-vast/vast-tools/pull/412)
- Fixed handling of Stokes Q/U/V selavy files [#410](https://github.com/askap-vast/vast-tools/pull/410)
- Fixed naming of fits cutout functions [#407](https://github.com/askap-vast/vast-tools/pull/407)
- Changed field name handling to correctly convert RACS -> VAST [#405](https://github.com/askap-vast/vast-tools/pull/405)
- Fixed variable names in query documentation example [#398](https://github.com/askap-vast/vast-tools/pull/398)
- Updated logging in query.py to correctly reference the file type it was dealing with [#398](https://github.com/askap-vast/vast-tools/pull/398)
- Fixed handling of empty list of planets in query [#390](https://github.com/askap-vast/vast-tools/pull/390)
- Fix bug where cutouts could not be created [#389](https://github.com/askap-vast/vast-tools/pull/389)
- Correctly plot integrated flux histograms in eta-V bokeh plots [#383](https://github.com/askap-vast/vast-tools/pull/383)
- Fixed handling of negative values in eta-V bokeh plots [#383](https://github.com/askap-vast/vast-tools/pull/383)
- Fixed incorrect selavy filenames for EPOCH00 tiles [#373](https://github.com/askap-vast/vast-tools/pull/373)
- Fixed `search_around_coordinates` option in query [#365](https://github.com/askap-vast/vast-tools/pull/365)
- Fixed handling of RACS-mid fields [#368](https://github.com/askap-vast/vast-tools/pull/368)
- Fixed handling of RACS filenames and epochs [#358](https://github.com/askap-vast/vast-tools/pull/358)
- Fixed issue with ncpu exceeding number of unique sources in Query._init_sources [#363](https://github.com/askap-vast/vast-tools/pull/363)
- Fixed plot_lightcurve legend creation [#345](https://github.com/askap-vast/vast-tools/pull/345)
- Fixed pipeline eta-v matplotlib plot [#340](https://github.com/askap-vast/vast-tools/pull/340)
- Fixed pandas dataframe and series append deprecation by using `pd.concat` [#337](https://github.com/askap-vast/vast-tools/pull/337).
- Fixed metadata for islands queries [#328](https://github.com/askap-vast/vast-tools/pull/328)
- Updated missing values in RACS beam info csv file [#311](https://github.com/askap-vast/vast-tools/pull/311)

#### Removed

- Removed '#' from get_components dataframe except when it is used in `search_around_coordinates` queries [#365](https://github.com/askap-vast/vast-tools/pull/365)

#### List of PRs

- [#490](https://github.com/askap-vast/vast-tools/pull/490): feat: Raise exception when simbad name query returns more names than there are objects queried
- [#478](https://github.com/askap-vast/vast-tools/pull/478): fix: Account for pipeline runs with no measurement pairs file
- [#483](https://github.com/askap-vast/vast-tools/pull/483): fix: Ensure that full image data does not persist when creating FITS cutouts from compressed HDUs
- [#480](https://github.com/askap-vast/vast-tools/pull/480): feat: Added open_fits function to handle compressed image HDUs
- [#476](https://github.com/askap-vast/vast-tools/pull/476): fix: Fixed missing RMS measurements for non-detections
- [#472](https://github.com/askap-vast/vast-tools/pull/472): fix: Remove duplicates in field matching
- [#470](https://github.com/askap-vast/vast-tools/pull/470): feat: Added epoch 41
- [#465](https://github.com/askap-vast/vast-tools/pull/465): feat: Added warning when querying corrected data
- [#466](https://github.com/askap-vast/vast-tools/pull/466): feat: Added epochs 38, 39 and 40
- [#463](https://github.com/askap-vast/vast-tools/pull/463): fix: Fix read_selavy to force all flux & flux_err values to be positive
- [#458](https://github.com/askap-vast/vast-tools/pull/458): feat: Added epoch 37
- [#453](https://github.com/askap-vast/vast-tools/pull/453): fix: Fix planet matching to correctly handle field names with/without 'A' suffix, and correctly handle empty df in init_sources
- [#455](https://github.com/askap-vast/vast-tools/pull/455): feat: Added epoch 36
- [#454](https://github.com/askap-vast/vast-tools/pull/454): feat: Added epochs 34 and 35
- [#452](https://github.com/askap-vast/vast-tools/pull/452): fix: Updated VASTMOCS to correctly handle fields with/without 'A' suffix
- [#431](https://github.com/askap-vast/vast-tools/pull/431): feat, docs, tests: Store beam centre SkyCoords as serialised objects rather than generating them on the fly
- [#449](https://github.com/askap-vast/vast-tools/pull/449): fix: Updated github pytest workflow to use Gr1N/setup-poetry@v8
- [#440](https://github.com/askap-vast/vast-tools/pull/440): fix: Updated default value of Query.max_sep to 1.5 degrees
- [#438](https://github.com/askap-vast/vast-tools/pull/438): feat: Updated RACS dec limit to +50
- [#437](https://github.com/askap-vast/vast-tools/pull/437): fix: Correctly handle show_all_png_cutouts(save=True) file naming
- [#435](https://github.com/askap-vast/vast-tools/pull/435): fix: Ensure selected_fields is defined for all field matching cases
- [#434](https://github.com/askap-vast/vast-tools/pull/434): feat: Added access to epoch 33
- [#418](https://github.com/askap-vast/vast-tools/pull/418): feat: Added option to return all data at the available position rather than one field per epoch.
- [#430](https://github.com/askap-vast/vast-tools/pull/430): feat: Allow users to specify which dask scheduler to use for multi-processing
- [#421](https://github.com/askap-vast/vast-tools/pull/421): feat, fix, docs: Updated Query._get_epochs to exit nicely when no epochs available & to allow lists and ints to be passed.
- [#429](https://github.com/askap-vast/vast-tools/pull/429): feat: Added access to epoch 32
- [#427](https://github.com/askap-vast/vast-tools/pull/427): feat: Added access to epoch 31
- [#425](https://github.com/askap-vast/vast-tools/pull/425): fix: Bumped pytest and lint github workflow from ubuntu-18.04 -> ubuntu-20.04
- [#419](https://github.com/askap-vast/vast-tools/pull/419): feat: Added access to epoch 30
- [#416](https://github.com/askap-vast/vast-tools/pull/416): fix: Update match_planet_to_field to handle empty groups
- [#414](https://github.com/askap-vast/vast-tools/pull/414): feat: Added access to epoch 27
- [#412](https://github.com/askap-vast/vast-tools/pull/412): fix: Added `corrected_data` to `Source` class to ensure correct image paths are used
- [#410](https://github.com/askap-vast/vast-tools/pull/410): fix: Fix handling of Stokes V products and image naming scheme
- [#406](https://github.com/askap-vast/vast-tools/pull/406): feat: Add epochs 22, 23, 24, 25, 26, 28 (RACS-high), 29 (RACS-low-2)
- [#407](https://github.com/askap-vast/vast-tools/pull/407): fix: Fixed naming of fits cutout functions
- [#403](https://github.com/askap-vast/vast-tools/pull/403): feat, docs: Changed data product generation from epoch-based to index-based
- [#405](https://github.com/askap-vast/vast-tools/pull/405): fix: Fix incorrect handling of RACS-named fields
- [#370](https://github.com/askap-vast/vast-tools/pull/370): feat, docs: Add file validation to check what data is available and remove anything that is not available from the query.
- [#398](https://github.com/askap-vast/vast-tools/pull/398): fix, docs: Fix incorrect logging in query.py and incorrect variable names in query documentation
- [#390](https://github.com/askap-vast/vast-tools/pull/390): fix: Update handling of planets to be consistent with regular sources.
- [#389](https://github.com/askap-vast/vast-tools/pull/389): fix: Fix cutout creation bug.
- [#385](https://github.com/askap-vast/vast-tools/pull/385): dep: Update Gr1N/setup-poetry to v7.
- [#380](https://github.com/askap-vast/vast-tools/pull/380): dep: Update numpy to ~1.22.1.
- [#379](https://github.com/askap-vast/vast-tools/pull/379): feat: Add function to create WISE color-color plots.
- [#373](https://github.com/askap-vast/vast-tools/pull/373): fix: Fixed incorrect selavy filenames for RACS tiles
- [#365](https://github.com/askap-vast/vast-tools/pull/365): fix, tests: Fix query `search_around_coordinates` option.
- [#368](https://github.com/askap-vast/vast-tools/pull/368): fix, docs: Ensure find_fields returns RACS fields.
- [#361](https://github.com/askap-vast/vast-tools/pull/361): feat, docs: remove epoch 12 warning.
- [#358](https://github.com/askap-vast/vast-tools/pull/358): fix: Fix RACS file paths and epoch handling.
- [#363](https://github.com/askap-vast/vast-tools/pull/363): fix: Update dask npartition specification in Query._init_sources call.
- [#351](https://github.com/askap-vast/vast-tools/pull/351): fix, feat: Add epoch 21.
- [#346](https://github.com/askap-vast/vast-tools/pull/346/): feat, fix, tests: Check if requested data exists in Query init.
- [#345](https://github.com/askap-vast/vast-tools/pull/345): fix: Fixed plot_lightcurve.
- [#338](https://github.com/askap-vast/vast-tools/pull/338): feat, dep: Added pylint workflow for pull requests.
- [#340](https://github.com/askap-vast/vast-tools/pull/340): fix: Fixed eta-v plot.
- [#337](https://github.com/askap-vast/vast-tools/pull/337): fix: Fixed pandas append deprecation.
- [#325](https://github.com/askap-vast/vast-tools/pull/325): tests, feat: Enabled access to full TILES data.
- [#328](https://github.com/askap-vast/vast-tools/pull/328): fix, docs: Fixed metadata for islands queries.
- [#327](https://github.com/askap-vast/vast-tools/pull/327): feat, docs: Allow query epochs to be specified as list.
- [#311](https://github.com/askap-vast/vast-tools/pull/311): tests, docs, feat: Added multi-frequency handling and small updates.
- [#323](https://github.com/askap-vast/vast-tools/pull/323): feat, tests: New options for MOC generation.
- [#315](https://github.com/askap-vast/vast-tools/pull/315): tests, docs, feat: Use offset, rather than absolute, coordinates for postagestamps.
- [#317](https://github.com/askap-vast/vast-tools/pull/317): docs: Added warning to docs.
- [#298](https://github.com/askap-vast/vast-tools/pull/298): tests, docs, feat: Added tools for new epoch addition.

## [2.0.0](https://github.com/askap-vast/vast-tools/releases/v2.0.0) (2021-10-09)

**Note** Changelog started from PR[#273](https://github.com/askap-vast/vast-tools/pull/273) hence full notes are not available from before this point.

#### Added

- Added tools sub-module consisting of skymap/MOC interface tools [#277](https://github.com/askap-vast/vast-tools/pull/277).
- Added `get_supported_epochs` function to survey component [#294](https://github.com/askap-vast/vast-tools/pull/294).
- Added support for epoch 13 (formerly epoch 12) [#293](https://github.com/askap-vast/vast-tools/pull/293).
- Added new data files for epoch 12 [#293](https://github.com/askap-vast/vast-tools/pull/293).
- Added epoch 12 warnings for query and moc components in code and docs [#293](https://github.com/askap-vast/vast-tools/pull/293).
- Added pep8speaks config file to project [#290](https://github.com/askap-vast/vast-tools/pull/290).
- Added a method to recalculate the measurement pairs dataframe (two epoch metrics) [#290](https://github.com/askap-vast/vast-tools/pull/290).
- Docs badge on readme [#288](https://github.com/askap-vast/vast-tools/pull/288).
- Added unit testing for the codebase [#285](https://github.com/askap-vast/vast-tools/pull/285).
- Added documentation on tests [#285](https://github.com/askap-vast/vast-tools/pull/285).
- Added github workflow to run pytest on new PRs [#285](https://github.com/askap-vast/vast-tools/pull/285).
- Added documentation using the mkdocs framework [#273](https://github.com/askap-vast/vast-tools/pull/273).

#### Changed

- Minor documentation edits [#289](https://github.com/askap-vast/vast-tools/pull/289).
- Updated survey docs page to reflect previous refactoring [#294](https://github.com/askap-vast/vast-tools/pull/294).
- Changed all former epoch 12 data files to epoch 13 [#293](https://github.com/askap-vast/vast-tools/pull/293).
- Updated vaex-core version to 4.5.0 [#290](https://github.com/askap-vast/vast-tools/pull/290).
- Updated docs dependencies versions to fix notebook pages [#288](https://github.com/askap-vast/vast-tools/pull/288).
- Refactored data loading to use importlib.resources [#285](https://github.com/askap-vast/vast-tools/pull/285).
- Refactored some data loading to be method base [#285](https://github.com/askap-vast/vast-tools/pull/285).
- Moved `RELEASED_EPOCHS` and `ALLOWED_PLANETS` to top level import [#285](https://github.com/askap-vast/vast-tools/pull/285).
- Changed docstrings to google format for docs compatibility [#273](https://github.com/askap-vast/vast-tools/pull/273).

#### Fixed

- Fixed docs announcement bar link [#289](https://github.com/askap-vast/vast-tools/pull/289).
- Fixed test_query.py::TestQuery::test_init_failure_base_folder test [#291](https://github.com/askap-vast/vast-tools/pull/291).
- Fixed jsmin dependancy install (2.2.2 -> 3.0.0) [#285](https://github.com/askap-vast/vast-tools/pull/285).
- Remerged docs branch [#287](https://github.com/askap-vast/vast-tools/pull/287).
- Docs branch reverted to avoid squashing to make merge of tests straightforward [#286](https://github.com/askap-vast/vast-tools/pull/286).

#### Removed

- Removed MANIFEST.in file [#289](https://github.com/askap-vast/vast-tools/pull/289).
- Removed deprecated information from the survey docs page [#294](https://github.com/askap-vast/vast-tools/pull/294).
- Removed `EpochInfo` class from query as no longer used [#285](https://github.com/askap-vast/vast-tools/pull/285).

#### List of PRs

- [#289](https://github.com/askap-vast/vast-tools/pull/289): docs: Minor documentation edits.
- [#277](https://github.com/askap-vast/vast-tools/pull/277): tests, docs, feat, dep: Add skymap utils
- [#294](https://github.com/askap-vast/vast-tools/pull/294): docs, feat: Survey code tidy and docs correction.
- [#293](https://github.com/askap-vast/vast-tools/pull/293): feat: Added EPOCH13 support.
- [#290](https://github.com/askap-vast/vast-tools/pull/290): docs, dep, feat: Added a recalc two epoch metrics method.
- [#291](https://github.com/askap-vast/vast-tools/pull/291): tests: Fixed TestQuery::test_init_failure_base_folder test.
- [#288](https://github.com/askap-vast/vast-tools/pull/288): docs, dep: Updated docs dependencies.
- [#285](https://github.com/askap-vast/vast-tools/pull/285): test, feat, dep: Added codebase unit testing.
- [#287](https://github.com/askap-vast/vast-tools/pull/287): docs, dep: Documentation using mkdocs
- [#286](https://github.com/askap-vast/vast-tools/pull/286): docs, dep: Revert "Documentation using mkdocs (#273)"
- [#273](https://github.com/askap-vast/vast-tools/pull/273): docs, dep: Documentation using mkdocs
- [#274](https://github.com/askap-vast/vast-tools/pull/274): fix: fixing typo in exception
- [#270](https://github.com/askap-vast/vast-tools/pull/270): dep: Changed over dependency management to poetry
- [#264](https://github.com/askap-vast/vast-tools/pull/264): feat: Add negative selavy catalogues
- [#258](https://github.com/askap-vast/vast-tools/pull/258): feat: v2.0.0-rc.6
- [#257](https://github.com/askap-vast/vast-tools/pull/257): fix: Fix lightcurve axist limits for forced fits
- [#255](https://github.com/askap-vast/vast-tools/pull/255): fix: Added if statement to catch os.getenv returning None
- [#253](https://github.com/askap-vast/vast-tools/pull/253): feat: Added filter pipeline run by MOC function
- [#251](https://github.com/askap-vast/vast-tools/pull/251): fix: Check self.pipeline inside write_measurements
- [#247](https://github.com/askap-vast/vast-tools/pull/247): feat, dep: Updated forced photometry to be a dependency
- [#246](https://github.com/askap-vast/vast-tools/pull/246): fix: Fixed vaex two epoch metric analysis
- [#244](https://github.com/askap-vast/vast-tools/pull/244): fix, dep: Fix mocpy dependancy
- [#243](https://github.com/askap-vast/vast-tools/pull/243): fix: Fixed dropping level on empty dataframe and selavy-simple
- [#242](https://github.com/askap-vast/vast-tools/pull/242): dep: Frozen dependancies versions
- [#238](https://github.com/askap-vast/vast-tools/pull/238): feat: Update the example notebooks with pipeline catalogue crossmatching and new features
- [#236](https://github.com/askap-vast/vast-tools/pull/236): feat: Added pipeline sources SkyCoord convenience function
- [#235](https://github.com/askap-vast/vast-tools/pull/235): feat: Added recalc_sources functionality to pipeline
- [#232](https://github.com/askap-vast/vast-tools/pull/232): feat: Give project_dir precedence over environment variable
- [#230](https://github.com/askap-vast/vast-tools/pull/230): feat: Improve plotting capabilities
- [#228](https://github.com/askap-vast/vast-tools/pull/228): fix: Fix fits loc issue
- [#224](https://github.com/askap-vast/vast-tools/pull/224): fix: Changed lightcurve y axis start at 0 by default
- [#223](https://github.com/askap-vast/vast-tools/pull/223): fix: Fixes query checks with scaler SkyCoord entry

## [v2.0.0-rc.5](https://github.com/askap-vast/vast-tools/releases/v2.0.0-rc.5) (2020-10-20)

The release candidate 5 of v2.0.0. This was the release before the changelog was implemented.

#### List of PRs

- [#221](https://github.com/askap-vast/vast-tools/pull/221) feat: Pipeline add 2 epoch Mooley style plotting
- [#220](https://github.com/askap-vast/vast-tools/pull/220) fix: Issue 219 fix source names
- [#218](https://github.com/askap-vast/vast-tools/pull/218) fix: Removed bad fields from epoch 12 csv
