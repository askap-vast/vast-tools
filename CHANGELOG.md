# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), with an added `List of PRs` section and links to the relevant PRs on the individal updates. This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased](https://github.com/askap-vast/vast-tools/compare/v2.0.0...HEAD)

#### Added

- Added observing frequency to all fields csv files [#311](https://github.com/askap-vast/vast-tools/pull/311)
- Added OBSERVED_EPOCHS variable for epochs that have been observed but not yet released [#311](https://github.com/askap-vast/vast-tools/pull/311)
- Added access to epochs up 17, 18, 19 [#311](https://github.com/askap-vast/vast-tools/pull/311)
- Added BASE_EPOCHS variable [#311](https://github.com/askap-vast/vast-tools/pull/311)
- Added tools.offset_postagestamp_axes which allows users to display postagestamps with offset (rather than absolute) coordinates [#315](https://github.com/askap-vast/vast-tools/pull/315)
- Added Dask RuntimeError warning to query.md docs [#317](https://github.com/askap-vast/vast-tools/pull/317)
- Added tools for new epoch addition: add datetimes to fits images create fields csv files, create and update MOCs [#298](https://github.com/askap-vast/vast-tools/pull/298)

#### Changed

- Enabled access to full TILES data [#325](https://github.com/askap-vast/vast-tools/pull/325)
- [#327](https://github.com/askap-vast/vast-tools/pull/327): feat, docs: Allow query epochs to be specified as list
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

- Fixed pandas dataframe and series append deprecation by using `pd.concat` [#337](https://github.com/askap-vast/vast-tools/pull/337).
- Fixed metadata for islands queries [#328](https://github.com/askap-vast/vast-tools/pull/328)
- Updated missing values in RACS beam info csv file [#311](https://github.com/askap-vast/vast-tools/pull/311)

#### Removed

#### List of PRs

- [#337](https://github.com/askap-vast/vast-tools/pull/337): fix: Fixed pandas append deprecation.
- [#325](https://github.com/askap-vast/vast-tools/pull/325): tests, feat: Enabled access to full TILES data
- [#328](https://github.com/askap-vast/vast-tools/pull/328): fix, docs: Fixed metadata for islands queries 
- [#327](https://github.com/askap-vast/vast-tools/pull/327): feat, docs: Allow query epochs to be specified as list
- [#311](https://github.com/askap-vast/vast-tools/pull/311): tests, docs, feat: Added multi-frequency handling and small updates
- [#323](https://github.com/askap-vast/vast-tools/pull/323): feat, tests: New options for MOC generation
- [#315](https://github.com/askap-vast/vast-tools/pull/315): tests, docs, feat: Use offset, rather than absolute, coordinates for postagestamps
- [#317](https://github.com/askap-vast/vast-tools/pull/317): docs: Added warning to docs
- [#298](https://github.com/askap-vast/vast-tools/pull/298): tests, docs, feat: Added tools for new epoch addition

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

