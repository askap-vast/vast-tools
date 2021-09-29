# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), with an added `List of PRs` section and links to the relevant PRs on the individal updates. This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased](https://github.com/askap-vast/vast-tools/compare/v2.0.0-rc.5...HEAD)

**Note** Changelog started from PR[#273](https://github.com/askap-vast/vast-tools/pull/273) hence full notes are not available from before this point.

#### Added

- Added unit testing for the codebase [#285](https://github.com/askap-vast/vast-tools/pull/285).
- Added documentation on tests [#285](https://github.com/askap-vast/vast-tools/pull/285).
- Added github workflow to run pytest on new PRs [#285](https://github.com/askap-vast/vast-tools/pull/285).
- Added documentation using the mkdocs framework [#273](https://github.com/askap-vast/vast-tools/pull/273).

#### Changed

- Refactored data loading to use importlib.resources [#285](https://github.com/askap-vast/vast-tools/pull/285).
- Refactored some data loading to be method base [#285](https://github.com/askap-vast/vast-tools/pull/285).
- Moved `RELEASED_EPOCHS` and `ALLOWED_PLANETS` to top level import [#285](https://github.com/askap-vast/vast-tools/pull/285).
- Changed docstrings to google format for docs compatibility [#273](https://github.com/askap-vast/vast-tools/pull/273).

#### Fixed

- Fixed jsmin dependancy install (2.2.2 -> 3.0.0) [#285](https://github.com/askap-vast/vast-tools/pull/285).
- Remerged docs branch [#287](https://github.com/askap-vast/vast-tools/pull/287).
- Docs branch reverted to avoid squashing to make merge of tests straightforward [#286](https://github.com/askap-vast/vast-tools/pull/286).

#### Removed

- Removed `EpochInfo` class from query as no longer used [#285](https://github.com/askap-vast/vast-tools/pull/285).

#### List of PRs

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

