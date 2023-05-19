import astropy.units as u
import importlib.resources
import numpy as np
import os
import pandas as pd
import pytest

from astropy.coordinates import SkyCoord, Angle
from mocpy import MOC
from pytest_mock import mocker, MockerFixture  # noqa: F401
from typing import List, Union

import vasttools.query as vtq


@pytest.fixture
def pilot_moc_mocker() -> MOC:
    """
    Loads the VAST Pilot Epoch 1 MOC directly.

    Returns:
        The Pilot epoch 1 MOC.
    """
    with importlib.resources.path(
        "vasttools.data.mocs",
        "VAST_PILOT_EPOCH01.moc.fits"
    ) as moc_path:
        moc = MOC.from_fits(moc_path)

    return moc


@pytest.fixture
def vast_query_psrj2129(pilot_moc_mocker: MOC,
                        mocker: MockerFixture
                        ) -> vtq.Query:
    """
    Initialises a Query object with the Pulsar J2129-04 as the search
    subject.

    For a reason I'm not sure of the moc opening in _build_catalog
    cannot open the MOC file within the test. So here I mock the opening
    and return the epoch 1 moc file. It seems to work fine outside of the
    tests.

    Args:
        pilot_moc_mocker: The pytest fixture that directly loads the
            pilot epoch 1 MOC.
        mocker: The pytest mock mocker object.

    Returns:
        The Query instance.
    """
    test_dir = '/testing/folder'
    mocker_isdir = mocker.patch(
        'vasttools.query.os.path.isdir',
        return_value=True
    )
    mocker_moc_open = mocker.patch(
        'mocpy.MOC.from_fits',
        return_value=pilot_moc_mocker
    )

    mocker_file_validation = mocker.patch(
        'vasttools.query.Query._validate_files',
        return_value=True
    )

    psr_coordinate = SkyCoord(322.4387083, -4.4866389, unit=(u.deg, u.deg))

    psr_query = vtq.Query(
        coords=psr_coordinate,
        source_names=['PSR J2129-04'],
        base_folder=test_dir,
        epochs='1,2'
    )

    return psr_query


@pytest.fixture
def vast_fields_object_dummy() -> pd.DataFrame:
    """
    A dummy fields available dataframe.

    Note that id 4000 is made up.

    Returns:
        The dummy fields dataframe.
    """
    fields_df = pd.DataFrame(
        data={
            'SBID': {
                1893: 9668,
                1894: 9668,
                1907: 9668,
                3693: 9673,
                3694: 9673,
                3707: 9673,
                4000: 10000,
            },
            'FIELD_NAME': {
                1893: 'VAST_2118-06',
                1894: 'VAST_2118-06',
                1907: 'VAST_2118-06',
                3693: 'VAST_2143-06',
                3694: 'VAST_2143-06',
                3707: 'VAST_2143-06',
                4000: 'VAST_0612-06'
            },
            'BEAM': {
                1893: 21,
                1894: 22,
                1907: 35,
                3693: 21,
                3694: 22,
                3707: 35,
                4000: 21,
            },
            'RA_HMS': {
                1893: '21:29:08.479',
                1894: '21:29:09.329',
                1907: '21:08:04.67',
                3693: '21:53:58.478',
                3694: '21:53:59.33',
                3707: '21:32:54.67',
                4000: '06:12:54.67'
            },
            'DEC_DMS': {
                1893: '-03:40:02.39',
                1894: '-04:43:04.76',
                1907: '-04:43:04.76',
                3693: '-03:40:02.39',
                3694: '-04:43:04.76',
                3707: '-04:43:04.76',
                4000: '-04:43:04.76'
            },
            'DATEOBS': {
                1893: '2019-08-27 18:52:00.556',
                1894: '2019-08-27 18:52:00.556',
                1907: '2019-08-27 18:52:00.556',
                3693: '2019-08-28 12:45:25.688',
                3694: '2019-08-28 12:45:25.688',
                3707: '2019-08-28 12:45:25.688',
                4000: '2019-08-28 14:45:25.688'
            },
            'DATEEND': {
                1893: '2019-08-27 19:04:27.052',
                1894: '2019-08-27 19:04:27.052',
                1907: '2019-08-27 19:04:27.052',
                3693: '2019-08-28 12:57:52.184',
                3694: '2019-08-28 12:57:52.184',
                3707: '2019-08-28 12:57:52.184',
                4000: '2019-08-28 14:57:52.184'
            },
            'NINT': {
                1893: 76,
                1894: 76,
                1907: 76,
                3693: 76,
                3694: 76,
                3707: 76,
                4000: 76
            },
            'BMAJ': {
                1893: 21.335756,
                1894: 20.366165,
                1907: 24.169145,
                3693: 17.463584,
                3694: 17.308171,
                3707: 21.308849,
                4000: 21.308849
            },
            'BMIN': {
                1893: 10.861959,
                1894: 12.228147,
                1907: 10.532837,
                3693: 11.47515,
                3694: 11.454186,
                3707: 11.042338,
                4000: 11.042338
            },
            'BPA': {
                1893: -69.293772,
                1894: -70.451286,
                1907: -69.100177,
                3693: 71.399327,
                3694: 73.426134,
                3707: 60.658346,
                4000: 60.658346
            }
        }
    )

    return fields_df


@pytest.fixture
def fields_df_expected_result() -> pd.DataFrame:
    """
    The expected result from a fields query.

    I.e. the source matched to the available fields.

    Returns:
        The fields result dataframe.
    """
    def _get_result(add_files: bool = False) -> pd.DataFrame:
        """
        The actual workhorse function to generate the dataframe.

        Args:
            add_files: If True then the dummy file locations is added to the
                results dataframe.

        Returns:
            The results dataframe.
        """
        fields_df = pd.DataFrame(
            data={
                'name': {
                    0: 'PSR J2129-04',
                    1: 'PSR J2129-04'
                },
                'ra': {
                    0: 322.4387083,
                    1: 322.4387083
                },
                'dec': {
                    0: -4.4866389,
                    1: -4.4866389
                },
                'skycoord': {
                    0: SkyCoord(322.4387083, -4.4866389, unit=(u.deg, u.deg)),
                    1: SkyCoord(322.4387083, -4.4866389, unit=(u.deg, u.deg))
                },
                'stokes': {
                    0: 'I',
                    1: 'I'
                },
                'fields': {
                    0: np.array(
                        ['VAST_2118-06A', 'VAST_2143-06A'], dtype=object
                    ),
                    1: np.array(
                        ['VAST_2118-06A', 'VAST_2143-06A'], dtype=object
                    )
                },
                'primary_field': {
                    0: 'VAST_2118-06A',
                    1: 'VAST_2118-06A'
                },
                'epoch': {
                    0: '1',
                    1: '2'
                },
                'field': {
                    0: 'VAST_2118-06A',
                    1: 'VAST_2118-06A'
                },
                'sbid': {
                    0: 9668,
                    1: 10342
                },
                'dateobs': {
                    0: pd.Timestamp('2019-08-27 18:52:00.556000'),
                    1: pd.Timestamp('2019-10-30 10:11:56.913000')
                },
                'frequency': {
                    0: 887.5,
                    1: 887.5
                },
                'planet': {
                    0: False,
                    1: False
                }
            }
        )

        if add_files:
            fields_df['selavy'] = [
                '/testing/folder/EPOCH01/COMBINED/STOKESI_SELAVY'
                '/selavy-VAST_2118-06A.EPOCH01.I.conv.components.xml',
                '/testing/folder/EPOCH02/COMBINED/STOKESI_SELAVY'
                '/selavy-VAST_2118-06A.EPOCH02.I.conv.components.xml',
            ]
            fields_df['image'] = [
                '/testing/folder/EPOCH01/COMBINED/STOKESI_IMAGES'
                '/VAST_2118-06A.EPOCH01.I.conv.fits',
                '/testing/folder/EPOCH02/COMBINED/STOKESI_IMAGES'
                '/VAST_2118-06A.EPOCH02.I.conv.fits',
            ]
            fields_df['rms'] = [
                '/testing/folder/EPOCH01/COMBINED/STOKESI_RMSMAPS'
                '/noiseMap.VAST_2118-06A.EPOCH01.I.conv.fits',
                '/testing/folder/EPOCH02/COMBINED/STOKESI_RMSMAPS'
                '/noiseMap.VAST_2118-06A.EPOCH02.I.conv.fits'
            ]

        return fields_df
    return _get_result


@pytest.fixture
def vast_query_psrj2129_fields(
    vast_query_psrj2129: vtq.Query,
    fields_df_expected_result: pd.DataFrame
) -> vtq.Query:
    """
    A fixture that returns the pulsar Query instance with the files added
    to the result.

    Args:
        vast_query_psrj2129: The defined Query fixture.
        fields_df_expected_result: The expected fields df result which is
            added.
    """
    vast_query_psrj2129.fields_found = True
    vast_query_psrj2129.fields_df = fields_df_expected_result(add_files=True)

    return vast_query_psrj2129


@pytest.fixture
def field_centres_dummy() -> pd.DataFrame:
    """
    A dummy fields centres file, returned as a dataframe.

    Returns:
        Dataframe with field centres information.
    """
    field_centres_df = pd.DataFrame(
        data={
            'field': {89: 'VAST_2118-06', 94: 'VAST_2143-06'},
            'centre-ra': {89: 319.65155983605507, 94: 325.8598931693551},
            'centre-dec': {89: -6.298205277661648, 94: -6.298205277661648}
        }
    )

    return field_centres_df


@pytest.fixture
def selavy_cat() -> pd.DataFrame:
    """
    A fixture to return a dummy selavy catalogue.

    Returns:
        Selavy catalogue loaded as a dataframe.
    """
    def _get_selavy_cat(
        contain_pulsar: bool = False,
        search_around: bool = False,
        search_around_index: bool = False,
        add_detection: bool = False
    ) -> pd.DataFrame:
        """
        Workhorse function to generate the selavy catalogue.

        Allows for arguments to be passed that can change the selavy catalogue
        that is generated.

        Args:
            contain_pulsar: If 'True' then a source candidate is placed
                at the location of PSR J2129-04.
            search_around: If 'True' then the dataframe is appended to itself
                such that multiple results are found if a search around query
                is used.
            search_around_index: If 'True' then an 'index' column is added
                to mimic the act of processing the search around query. Only
                done if 'search_around' is also 'True'.
            add_detection: If `True` then a `detection` column is added that
                mimics the same column that is added in the query process.

        Returns:
            The selavy catalogue as a pandas dataframe.
        """
        if contain_pulsar:
            component_name = {0: 'B2129-0429', 1: 'B2111-0353'}
            ra_hms_cont = {0: '21:29:45.29', 1: '21:11:50.0'}
            dec_dms_cont = {0: '-04:29:11.9', 1: '-03:53:10'}
            ra_deg_cont = {0: 322.4387083, 1: 317.958413}
            dec_deg_cont = {0: -4.4866389, 1: -3.886236}
        else:
            component_name = {0: 'B2126-0214', 1: 'B2111-0353'}
            ra_hms_cont = {0: '21:26:02.7', 1: '21:11:50.0'}
            dec_dms_cont = {0: '-02:14:24', 1: '-03:53:10'}
            ra_deg_cont = {0: 321.511286, 1: 317.958413}
            dec_deg_cont = {0: -2.24011, 1: -3.886236}

        selavy_df = pd.DataFrame(
            data={
                'island_id': {
                    0: 'SB9668_island_1000',
                    1: 'SB9668_island_1001'
                },
                'component_id': {
                    0: 'SB9668_component_1000a',
                    1: 'SB9668_component_1001a'
                },
                'component_name': component_name,
                'ra_hms_cont': ra_hms_cont,
                'dec_dms_cont': dec_dms_cont,
                'ra_deg_cont': ra_deg_cont,
                'dec_deg_cont': dec_deg_cont,
                'ra_err': {0: 0.02, 1: 0.02},
                'dec_err': {0: 0.02, 1: 0.01},
                'freq': {0: -0.0, 1: -0.0},
                'flux_peak': {0: 14.223, 1: 14.145},
                'flux_peak_err': {0: 0.05, 1: 0.034},
                'flux_int': {0: 10.68, 1: 14.558},
                'flux_int_err': {0: 0.06, 1: 0.05},
                'maj_axis': {0: 14.85, 1: 23.25},
                'min_axis': {0: 12.61, 1: 11.04},
                'pos_ang': {0: 92.8, 1: 109.82},
                'maj_axis_err': {0: 0.05, 1: 0.06},
                'min_axis_err': {0: 0.01, 1: 0.0},
                'pos_ang_err': {0: 0.89, 1: 0.12},
                'maj_axis_deconv': {0: 5.23, 1: 9.46},
                'min_axis_deconv': {0: 0.0, 1: 0.0},
                'pos_ang_deconv': {0: 23.31, 1: -69.52},
                'maj_axis_deconv_err': {0: 0.03, 1: 0.04},
                'min_axis_deconv_err': {0: 0.0, 1: 0.0},
                'pos_ang_deconv_err': {0: 0.09, 1: 1.19},
                'chi_squared_fit': {0: 18.206, 1: 12.652},
                'rms_fit_gauss': {0: 442.446, 1: 323.363},
                'spectral_index': {0: -0.92, 1: -1.08},
                'spectral_curvature': {0: -99.0, 1: -99.0},
                'spectral_index_err': {0: 0.0, 1: 0.0},
                'spectral_curvature_err': {0: 0.0, 1: 0.0},
                'rms_image': {0: 0.313, 1: 0.337},
                'has_siblings': {0: 0, 1: 0},
                'fit_is_estimate': {0: 0, 1: 0},
                'spectral_index_from_TT': {0: 1, 1: 1},
                'flag_c4': {0: 0, 1: 0},
                'comment': {0: np.nan, 1: np.nan}
            }
        )

        if add_detection:
            selavy_df['detection'] = [True, False]

        if search_around:
            selavy_df = pd.concat([selavy_df, selavy_df.loc[[0, 0, 0]]])
            if search_around_index:
                selavy_df['#'] = np.nan
                selavy_df['index'] = 0

        return selavy_df
    return _get_selavy_cat


class TestQuery:
    """
    This class contains all the tests related to the Query object in VAST
    Tools.
    """

    def test_init_failure_no_inputs(self) -> None:
        """
        Tests the initialisation failure of a Query object.

        Specifically when no input search objects are entered.

        Returns:
            None
        """
        with pytest.raises(vtq.QueryInitError) as excinfo:
            query = vtq.Query()

        assert str(excinfo.value).startswith(
            "No coordinates or source names have been provided!"
        )

    def test_init_failure_forced_fits_and_search_around(self) -> None:
        """
        Tests the initialisation failure of a Query object.

        Specifically when forced fits and search around options are used
        together.

        Returns:
            None
        """
        with pytest.raises(vtq.QueryInitError) as excinfo:
            query = vtq.Query(
                source_names=['test'],
                forced_fits=True,
                search_around_coordinates=True
            )

        assert str(excinfo.value).startswith(
            "Forced fits and search around"
        )

    def test_init_failure_length_mismatch(self) -> None:
        """
        Tests the initialisation failure of a Query object.

        Specifically the number of source names doesn't match the number of
        entered coordinates.

        Returns:
            None
        """
        with pytest.raises(vtq.QueryInitError) as excinfo:
            query = vtq.Query(
                coords=SkyCoord('00h42.5m', '+41d12m'),
                source_names=['test', 'test2'],
            )

        assert str(excinfo.value).startswith(
            "The number of entered source names"
        )

    def test_init_failure_invalid_planet(self) -> None:
        """
        Tests the initialisation failure of a Query object.

        Specifically when an invalid planet is entered.

        Returns:
            None
        """
        with pytest.raises(ValueError) as excinfo:
            query = vtq.Query(
                planets=['Earth']
            )

        assert str(excinfo.value) == "Invalid planet object provided!"

    def test_init_failure_base_folder(self, mocker: MockerFixture) -> None:
        """
        Tests the initialisation failure of a Query object.

        Specifically when the base folder directory has not been specified, or
        set in the environment.

        Args:
            mocker: The pytest-mock mocker object.

        Returns:
            None
        """
        mocker_getenv = mocker.patch(
            'os.getenv', return_value=None
        )

        with pytest.raises(vtq.QueryInitError) as excinfo:
            query = vtq.Query(
                planets=['Mars']
            )

        assert str(excinfo.value).startswith(
            "The base folder directory could not be determined!"
        )

    def test_init_failure_base_folder_not_found(self) -> None:
        """
        Tests the initialisation failure of a Query object.

        Specifically when no specified base folder is not found.

        Returns:
            None
        """
        with pytest.raises(vtq.QueryInitError) as excinfo:
            test_dir = '/testing/folder'
            query = vtq.Query(
                planets=['Mars'],
                base_folder=test_dir
            )

        assert str(excinfo.value) == f"Base folder {test_dir} not found!"

    @pytest.mark.parametrize(
        "vast_pilot,vast_full,fails",
        [
            (True, False, True),
            (True, True, False),
            (False, True, False),
        ]
    )
    def test_init_failure_stokes_v_tiles(self,
                                         vast_pilot: bool,
                                         vast_full: bool,
                                         fails: bool,
                                         mocker: MockerFixture) -> None:
        """
        Tests the initialisation failure of a Query object.

        Specifically when Stokes V is requested on tile images.

        Args:
            vast_pilot: Whether to include VAST pilot epochs.
            vast_full: Whether to include VAST full survey epochs.
            fails: Whether the initialisation should fail.
            mocker: The pytest-mock mocker object.

        Returns:
            None
        """
        epochs = []
        if vast_pilot:
            epochs.append("1")
        if vast_full:
            epochs.append("22")

        isdir_mocker = mocker.patch(
            'vasttools.query.os.path.isdir',
            return_value=True
        )
        test_dir = '/testing/folder'

        if fails:
            with pytest.raises(vtq.QueryInitError) as excinfo:
                query = vtq.Query(
                    epochs=",".join(epochs),
                    planets=['Mars'],
                    base_folder=test_dir,
                    stokes='v',
                    use_tiles=True
                )
            assert str(excinfo.value).startswith(
                "Problems found in query settings!"
            )
        else:
            query = vtq.Query(
                epochs=",".join(epochs),
                planets=['Mars'],
                base_folder=test_dir,
                stokes='v',
                use_tiles=True
            )

    def test_init_failure_no_sources_in_footprint(
        self,
        pilot_moc_mocker: MOC,
        mocker: MockerFixture
    ) -> None:
        """
        Tests the initialisation failure of a Query object.

        Specifically when none of the entered coordinates are in the
        footprint.

        Args:
            pilot_moc_mocker: The direct loaded epoch 1 MOC (for some reason
                this does not load correctly if not done like this in the
                test environment).
            mocker: The pytest-mock mocker object.

        Returns:
            None
        """
        with pytest.raises(vtq.QueryInitError) as excinfo:
            isdir_mocker = mocker.patch(
                'vasttools.query.os.path.isdir',
                return_value=True
            )
            mocker_data_available = mocker.patch(
                'vasttools.query.Query._check_data_availability',
                return_value=True
            )

            mocker_moc_open = mocker.patch(
                'mocpy.MOC.from_fits',
                return_value=pilot_moc_mocker
            )
            test_dir = '/testing/folder'
            test_coords = SkyCoord(
                ['00h42.5m', '17h42.5m'],
                ['+41d12m', '-82d12m']
            )

            query = vtq.Query(
                epochs='1',
                coords=test_coords,
                base_folder=test_dir,
            )

        assert str(excinfo.value) == (
            'No sources remaining. None of the entered coordinates'
            ' are found in the VAST Pilot survey footprint!'
        )

    def test_init_failure_invalid_scheduler(self,
                                            mocker: MockerFixture
                                            ) -> None:
        """
        Tests the initialisation failure of a Query object.

        Specifically when the requested dask scheduler is invalid.

        Args:
            mocker: The pytest-mock mocker object.

        Returns:
            None
        """
        isdir_mocker = mocker.patch(
            'vasttools.query.os.path.isdir',
            return_value=True
        )
        mocker_data_available = mocker.patch(
            'vasttools.query.Query._check_data_availability',
            return_value=True
        )

        with pytest.raises(vtq.QueryInitError) as excinfo:
            query = vtq.Query(
                planets=['Mars'],
                scheduler='bad-option',
                base_folder='/testing/folder'
            )

        assert str(excinfo.value) == (
            "bad-option is not a suitable scheduler option. Please "
            "select from ['processes', 'single-threaded']"
        )

    def test_init_settings(self, mocker: MockerFixture) -> None:
        """
        Tests the initialisation of a Query object.

        Tests that all provided settings are set correctly in the final Query
        object.

        Args:
            mocker: The pytest-mock mocker object.

        Returns:
            None
        """
        isdir_mocker = mocker.patch(
            'vasttools.query.os.path.isdir',
            return_value=True
        )
        mocker_data_available = mocker.patch(
            'vasttools.query.Query._check_data_availability',
            return_value=True
        )

        test_dir = '/testing/folder'
        epochs = '1,2,3x'
        stokes = 'I'
        crossmatch_radius = 1.
        use_tiles = True
        max_sep = 3.
        islands = False
        no_rms = True
        matches_only = True
        sort_output = True
        forced_fits = True
        forced_allow_nan = True
        forced_cluster_threshold = 7.5
        output_dir = '/output/here'
        incl_observed = False
        scheduler = 'processes'

        expected_settings = {
            'epochs': ["1", "2", "3x"],
            'stokes': stokes,
            'crossmatch_radius': Angle(crossmatch_radius * u.arcsec),
            'max_sep': max_sep,
            'islands': islands,
            'no_rms': no_rms,
            'matches_only': matches_only,
            'sort_output': sort_output,
            'forced_fits': forced_fits,
            'forced_allow_nan': forced_allow_nan,
            'forced_cluster_threshold': forced_cluster_threshold,
            'output_dir': output_dir,
            'search_around': False,
            'tiles': use_tiles,
            'incl_observed': False,
            'scheduler': 'processes'
        }

        query = vtq.Query(
            planets=['Mars'],
            base_folder=test_dir,
            stokes=stokes,
            epochs=epochs,
            crossmatch_radius=crossmatch_radius,
            max_sep=max_sep,
            use_islands=islands,
            use_tiles=use_tiles,
            no_rms=no_rms,
            matches_only=matches_only,
            sort_output=sort_output,
            forced_fits=forced_fits,
            forced_allow_nan=forced_allow_nan,
            forced_cluster_threshold=forced_cluster_threshold,
            output_dir=output_dir,
            incl_observed=incl_observed,
            scheduler=scheduler
        )

        assert query.settings == expected_settings

    @pytest.mark.parametrize(
        "epoch_exists,data_dir_exists,images_exist,"
        "cats_exist,rmsmaps_exist,no_rms,all_available",
        [
            (True, True, True, True, True, False, True),
            (True, True, True, True, True, True, True),
            (False, True, True, True, True, False, False),
            (True, False, True, True, True, False, False),
            (True, True, False, True, True, False, False),
            (True, True, True, False, True, False, False),
            (True, True, True, True, False, False, False),
            (True, True, True, True, False, True, True),
        ],
        ids=('all-available',
             'all-available-no-rms',
             'no-epoch',
             'no-data-dir',
             'no-image-dir',
             'no-selavy-dir',
             'no-rms-dir-rms',
             'no-rms-dir-no-rms'
             )
    )
    def test__check_data_availability(self,
                                      epoch_exists: bool,
                                      data_dir_exists: bool,
                                      images_exist: bool,
                                      cats_exist: bool,
                                      rmsmaps_exist: bool,
                                      no_rms: bool,
                                      all_available: bool,
                                      tmp_path
                                      ) -> None:
        """
        Test the data availability check

        Args:
            epoch_exists: The epoch directory exists.
            data_dir_exists: The data directory (i.e. COMBINED/TILES) exists.
            images_exist: The image directory (e.g. STOKESI_IMAGES) exists.
            cats_exist: The selavy directory (e.g. STOKESI_SELAVY) exists.
            rmsmaps_exist: The RMS map directory (e.g. STOKESI_RMSMAPS) exists.
            no_rms: The `no_rms` Query option has been selected.
            all_available: The expected result from _check_data_availability().
            tmp_path: Pathlib temporary directory path.

        Returns:
            None.
        """
        stokes = "I"
        epoch = "10x"
        data_type = "COMBINED"

        base_dir = tmp_path
        epoch_dir = base_dir / f"EPOCH{epoch}"
        data_dir = epoch_dir / data_type
        image_dir = data_dir / f"STOKES{stokes}_IMAGES"
        selavy_dir = data_dir / f"STOKES{stokes}_SELAVY"
        rms_dir = data_dir / f"STOKES{stokes}_RMSMAPS"

        if epoch_exists:
            epoch_dir.mkdir()
            if data_dir_exists:
                data_dir.mkdir()
                if images_exist:
                    image_dir.mkdir()
                if cats_exist:
                    selavy_dir.mkdir()
                if rmsmaps_exist:
                    rms_dir.mkdir()

        query = vtq.Query(
            epochs=epoch,
            planets=['Mars'],
            base_folder=base_dir,
            stokes=stokes,
            no_rms=no_rms
        )

        assert all_available == query._check_data_availability()

    def test__field_matching(
        self,
        vast_query_psrj2129: vtq.Query,
        vast_fields_object_dummy: pd.DataFrame,
        field_centres_dummy: pd.DataFrame,
        mocker: MockerFixture
    ) -> None:
        """
        Tests the field matching method of the query object.

        Checks that the matching fields are correctly identified from the
        dummy input data.

        Args:
            vast_query_psrj2129: The dummy Query instance that includes a
                search for PSR J2129-04.
            vast_fields_object_dummy: The dummy fields available to perform
                the search against.
            field_centres_dummy: The dummy field centres file.
            mocker: The pytest-mock mocker object.

        Returns:
            None
        """
        field_sc = SkyCoord(
            vast_fields_object_dummy["RA_HMS"],
            vast_fields_object_dummy["DEC_DMS"],
            unit=(u.hourangle, u.deg)
        )

        field_centres_sc = SkyCoord(
            field_centres_dummy["centre-ra"],
            field_centres_dummy["centre-dec"],
            unit=(u.deg, u.deg)
        )
        field_centre_names = field_centres_dummy.field

        row = vast_query_psrj2129.query_df.iloc[0]
        results = vast_query_psrj2129._field_matching(
            row,
            field_sc,
            vast_fields_object_dummy.FIELD_NAME,
            field_centres_sc,
            field_centre_names
        )

        assert np.all(results[0] == np.array(
            ['VAST_2118-06', 'VAST_2143-06']
        ))
        assert results[1] == 'VAST_2118-06'
        assert results[2] == ['1', '2']

    def test_find_fields(
        self,
        vast_query_psrj2129: vtq.Query,
        fields_df_expected_result: pd.DataFrame,
        mocker: MockerFixture
    ) -> None:
        """
        Tests the front facing find fields method.

        The Dask called is mocked and the expected result is returned. This is
        ok as the function is previously tested.

        Args:
            vast_query_psrj2129: The dummy Query instance that includes a
                search for PSR J2129-04.
            fields_df_expected_result: The expected fields_df result of the
                search.
            mocker: The pytest-mock mocker object.

        Returns:
            None
        """
        mocked_field_matching_results = (
            [np.array(['VAST_2118-06A', 'VAST_2143-06A'], dtype='object')],
            'VAST_2118-06A',
            [['1', '2']],
            [[
                ['1', 'VAST_2118-06A', 9668, '2019-08-27 18:52:00.556', 887.5],
                ['2', 'VAST_2118-06A', 10342, '2019-10-30 10:11:56.913', 887.5]
            ]],
            [[9668, 10342]],
            [['2019-08-27 18:52:00.556', '2019-10-30 10:11:56.913']],
            [[887.5, 887.5]]
        )

        dask_from_pandas_mocker = mocker.patch(
            'vasttools.query.dd.from_pandas',
        )

        (
            dask_from_pandas_mocker
            .return_value
            .apply
            .return_value
            .compute
            .return_value
        ) = mocked_field_matching_results

        vast_query_psrj2129.find_fields()

        assert vast_query_psrj2129.fields_df.equals(
            fields_df_expected_result()
        )

    @pytest.mark.parametrize("stokes, tiles, conv, islands, expected_file",
                             [('I',
                               True,
                               False,
                               None,
                               'selavy-image.i.VAST_2118-06A.SB9668.cont'
                               '.taylor.0.restored.components.corrected.xml'
                               ),
                              ('I',
                               True,
                               True,
                               None,
                               'selavy-image.i.VAST_2118-06A.SB9668.cont'
                               '.taylor.0.restored.conv.components.corrected'
                               '.xml'
                               ),
                              ('I',
                               False,
                               None,
                               True,
                               'selavy-VAST_2118-06A.EPOCH01.I.conv'
                               '.islands.xml'
                               ),
                              ('I',
                               False,
                               None,
                               False,
                               'selavy-VAST_2118-06A.EPOCH01.I.conv'
                               '.components.xml'
                               ),
                              ('V',
                               True,
                               None,
                               False,
                               'selavy-image.v.VAST_2118-06A.SB9668.cont'
                               '.taylor.0.restored.components.corrected.xml'
                               ),
                              ('V',
                               False,
                               None,
                               False,
                               'selavy-VAST_2118-06A.EPOCH01.V.conv'
                               '.components.xml'
                               )
                              ],
                             ids=('tiles-noconv',
                                  'tiles-conv',
                                  'comb-islands',
                                  'comb-noislands',
                                  'tiles-stokesv',
                                  'comb-stokesv',
                                  )
                             )
    def test__get_selavy_path(
        self,
        vast_query_psrj2129_fields: vtq.Query,
        stokes: str,
        tiles: bool,
        conv: bool,
        islands: bool,
        expected_file: str,
        mocker: MockerFixture
    ) -> None:
        """
        Tests adding the paths to the combined data in the query.
        Assumes the standard VAST Pilot directory and file structure.
        Args:
            vast_query_psrj2129_fields: The dummy Query instance that includes
                a search for PSR J2129-04 with the included found fields data.
            stokes: Which Stokes paramter to query.
            tiles: Whether to query the TILES or COMBINED data.
            conv: Whether `.conv` is present in the filename.
                This argument is only relevant if tiles is True
            islands: Whether to return the islands or components catalogue.
                This argument is only relevant if tiles is False.
            expected_file: The expected filename to be returned.
            mocker: The pytest-mock mocker object.
        Returns:
            None
        """
        epoch_string = 'EPOCH01'
        test_query = vast_query_psrj2129_fields

        test_query.settings['tiles'] = tiles
        test_query.settings['islands'] = islands
        test_query.settings['stokes'] = stokes

        row = test_query.fields_df.loc[0]

        if conv is not None:
            mock_selavy_isfile = mocker.patch(
                'vasttools.query.Path.is_file',
                return_value=conv
            )

        path = test_query._get_selavy_path(epoch_string, row)

        assert os.path.split(path)[1] == expected_file

    def test__add_files_combined(
        self,
        vast_query_psrj2129_fields: vtq.Query,
        mocker: MockerFixture
    ) -> None:
        """
        Tests adding the paths to the combined data in the query.

        Assumes the standard VAST Pilot directory and file structure.

        Args:
            vast_query_psrj2129_fields: The dummy Query instance that includes
                a search for PSR J2129-04 with the included found fields data.
            mocker: The pytest-mock mocker object.

        Returns:
            None
        """
        expected_results = (
            '/testing/folder/EPOCH01/COMBINED/STOKESI_SELAVY'
            '/selavy-VAST_2118-06A.EPOCH01.I.conv.components.xml',
            '/testing/folder/EPOCH01/COMBINED/STOKESI_IMAGES'
            '/VAST_2118-06A.EPOCH01.I.conv.fits',
            '/testing/folder/EPOCH01/COMBINED/STOKESI_RMSMAPS'
            '/noiseMap.VAST_2118-06A.EPOCH01.I.conv.fits'
        )

        test_query = vast_query_psrj2129_fields

        mock_selavy_path = mocker.patch(
            'vasttools.query.Query._get_selavy_path',
            return_value=expected_results[0]
        )

        results = test_query._add_files(test_query.fields_df.loc[0])

        for result, expected in zip(results, expected_results):
            assert result == expected

    @pytest.mark.parametrize("corrected, stokes",
                             [(True, "I"),
                              (True, "V"),
                              (False, "I"),
                              (False, "V"),
                              ],
                             ids=('corrected-i',
                                  'corrected-v',
                                  'uncorrected-i',
                                  'uncorrected-v',
                                  )
                             )
    def test__add_files_tiles(
        self,
        corrected: bool,
        stokes: str,
        vast_query_psrj2129_fields: vtq.Query,
        mocker: MockerFixture
    ) -> None:
        """
        Tests adding the paths to the tiles data in the query.

        Assumes the standard VAST Pilot directory and file structure.

        Args:
            corrected: Whether to test the corrected paths or not.
            stokes: Stokes parameter.
            vast_query_psrj2129_fields: The dummy Query instance that includes
                a search for PSR J2129-04 with the included found fields data.
            mocker: The pytest-mock mocker object.

        Returns:
            None
        """

        stokes_lower = stokes.lower()
        if corrected:
            expected_results = (
                f'/testing/folder/EPOCH01/TILES/STOKES{stokes}_SELAVY'
                f'_CORRECTED/selavy-image.{stokes_lower}.VAST_2118-06A.SB9668'
                '.cont.taylor.0.restored.components.corrected.xml',
                f'/testing/folder/EPOCH01/TILES/STOKES{stokes}_IMAGES'
                f'_CORRECTED/image.{stokes_lower}.VAST_2118-06A.SB9668.cont'
                '.taylor.0.restored.corrected.fits',
                f'/testing/folder/EPOCH01/TILES/STOKES{stokes}_RMSMAPS'
                f'_CORRECTED/noiseMap.image.{stokes_lower}.VAST_2118-06A'
                '.SB9668.cont.taylor.0.restored.corrected.fits'
            )
        else:
            expected_results = (
                f'/testing/folder/EPOCH01/TILES/STOKES{stokes}_SELAVY'
                f'/selavy-image.{stokes_lower}.VAST_2118-06A.SB9668.cont'
                '.taylor.0.restored.components.xml',
                f'/testing/folder/EPOCH01/TILES/STOKES{stokes}_IMAGES'
                f'/image.{stokes_lower}.VAST_2118-06A.SB9668.cont.taylor.0.'
                'restored.fits',
                f'/testing/folder/EPOCH01/TILES/STOKES{stokes}_RMSMAPS'
                f'/noiseMap.image.{stokes_lower}.VAST_2118-06A.SB9668.cont'
                '.taylor.0.restored.fits'
            )
        test_query = vast_query_psrj2129_fields
        test_query.settings['tiles'] = True
        test_query.settings['stokes'] = stokes

        test_query.corrected_data = corrected

        mock_selavy_path = mocker.patch(
            'vasttools.query.Query._get_selavy_path',
            return_value=expected_results[0]
        )

        results = test_query._add_files(
            test_query.fields_df.loc[0]
        )

        assert results == expected_results

    @pytest.mark.parametrize("stokes",
                             [("I"),
                              ("Q"),
                              ("U"),
                              ("V"),
                              ],
                             ids=('stokes-i',
                                  'stokes-q',
                                  'stokes-u',
                                  'stokes-v',
                                  )
                             )
    def test__add_files_stokes_combined(
        self,
        vast_query_psrj2129_fields: vtq.Query,
        stokes,
        mocker: MockerFixture
    ) -> None:
        """
        Tests adding the paths to the stokes v combined data in the query.

        Assumes the standard VAST Pilot directory and file structure.

        Args:
            vast_query_psrj2129_fields: The dummy Query instance that includes
                a search for PSR J2129-04 with the included found fields data.
            stokes: Stokes parameter
            mocker: The pytest-mock mocker object.

        Returns:
            None
        """
        expected_results = (
            f'/testing/folder/EPOCH01/COMBINED/STOKES{stokes}_SELAVY'
            f'/selavy-VAST_2118-06A.EPOCH01.{stokes}.conv.components.xml',
            f'/testing/folder/EPOCH01/COMBINED/STOKES{stokes}_IMAGES'
            f'/VAST_2118-06A.EPOCH01.{stokes}.conv.fits',
            f'/testing/folder/EPOCH01/COMBINED/STOKES{stokes}_RMSMAPS'
            f'/noiseMap.VAST_2118-06A.EPOCH01.{stokes}.conv.fits'
        )

        test_query = vast_query_psrj2129_fields
        test_query.settings['stokes'] = stokes
        test_query.fields_df['stokes'] = stokes

        mock_selavy_path = mocker.patch(
            'vasttools.query.Query._get_selavy_path',
            return_value=expected_results[0]
        )

        results = test_query._add_files(test_query.fields_df.loc[0])

        assert results == expected_results

    def test__get_components_detection(
        self,
        vast_query_psrj2129_fields: vtq.Query,
        selavy_cat: pd.DataFrame,
        mocker: MockerFixture
    ) -> None:
        """
        Tests the _get_components method of the Query class where a detection
        is expected.

        The selavy loading is mocked with the selavy catalogue fixture passed
        in it's place, where the pulsar is contained in the catalogue.

        Args:
            vast_query_psrj2129_fields: The dummy Query instance that includes
                a search for PSR J2129-04 with the included found fields data.
            selavy_cat: The dummy selavy catalogue.
            mocker: The pytest-mock mocker object.

        Returns:
            None
        """
        test_query = vast_query_psrj2129_fields

        mocked_input = test_query.fields_df.groupby('selavy').get_group(
            test_query.fields_df['selavy'].iloc[0]
        )

        mocker_selavy = mocker.patch(
            'vasttools.query.read_selavy',
            return_value=selavy_cat(contain_pulsar=True)
        )

        result = test_query._get_components(mocked_input)

        expected = selavy_cat(contain_pulsar=True).drop(1)
        expected['detection'] = True

        assert expected.equals(result)

    def test__get_components_non_detection(
        self,
        vast_query_psrj2129_fields: vtq.Query,
        selavy_cat: pd.DataFrame,
        mocker: MockerFixture
    ) -> None:
        """
        Tests the _get_components method of the Query class where no detection
        is expected.

        The selavy loading is mocked with the selavy catalogue fixture passed
        in it's place, where the pulsar is not present.

        Args:
            vast_query_psrj2129_fields: The dummy Query instance that includes
                a search for PSR J2129-04 with the included found fields data.
            selavy_cat: The dummy selavy catalogue.
            mocker: The pytest-mock mocker object.

        Returns:
            None
        """
        test_query = vast_query_psrj2129_fields

        mocked_input = test_query.fields_df.groupby('selavy').get_group(
            test_query.fields_df['selavy'].iloc[0]
        )

        mocker_selavy = mocker.patch(
            'vasttools.query.read_selavy',
            return_value=selavy_cat()
        )

        mocker_image = mocker.patch(
            'vasttools.query.Image'
        )

        (
            mocker_image
            .return_value
            .measure_coord_pixel_values
            .return_value
        ) = np.array([0.001])

        result = test_query._get_components(mocked_input)

        assert result.shape[0] == 1
        assert result.iloc[0]['rms_image'] == 1.
        assert result.iloc[0]['detection'] == False

    def test__get_components_search_around(
        self,
        vast_query_psrj2129_fields: vtq.Query,
        selavy_cat: pd.DataFrame,
        mocker: MockerFixture
    ) -> None:
        """
        Tests the _get_components method of the Query class where the
        'search_around' option is used and multiple matches are expected.

        The selavy loading is mocked with the selavy catalogue fixture passed
        in it's place, where the pulsar present four times.

        Args:
            vast_query_psrj2129_fields: The dummy Query instance that includes
                a search for PSR J2129-04 with the included found fields data.
            selavy_cat: The dummy selavy catalogue.
            mocker: The pytest-mock mocker object.

        Returns:
            None
        """
        test_query = vast_query_psrj2129_fields
        test_query.settings['search_around'] = True

        mocked_input = test_query.fields_df.groupby('selavy').get_group(
            test_query.fields_df['selavy'].iloc[0]
        )

        mocker_selavy = mocker.patch(
            'vasttools.query.pd.read_fwf',
            return_value=selavy_cat(contain_pulsar=True, search_around=True)
        )

        result = test_query._get_components(mocked_input)

        assert result.shape[0] == 4

    def test__get_forced_fits(
        self,
        vast_query_psrj2129_fields: vtq.Query,
        mocker: MockerFixture
    ) -> None:
        """
        Tests the method to get the forced fits of the query.

        The actual forced phot package is not tested here. Instead the results
        and calls are mocked and the calls and provided results are asserted
        against. The vasttools Image object is also mocked.

        Args:
            vast_query_psrj2129_fields: The dummy Query instance that includes
                a search for PSR J2129-04 with the included found fields data.
            mocker: The pytest-mock mocker object.

        Returns:
            None
        """
        mocked_Image = mocker.patch('vasttools.query.Image')
        mocked_Image.return_value.beam.major.to.return_value.value = 15.
        mocked_Image.return_value.beam.minor.to.return_value.value = 10.
        mocked_Image.return_value.beam.pa.to.return_value.value = 0.

        mocked_FP = mocker.patch('vasttools.query.ForcedPhot')
        input_fluxes = np.array([1., 1.5])
        mocked_FP.return_value.measure.return_value = (
            input_fluxes,
            np.array([0.1, 0.1]),
            np.array([1., 1.]),
            np.array([1., 1.]),
            np.array([True, True])
        )

        test_query = vast_query_psrj2129_fields
        group_name = test_query.fields_df['image'].iloc[0]
        mocked_input = (
            test_query
            .fields_df
            .groupby('image').get_group(group_name)
        )
        # need to drop the name column as it gets confused with the group name
        mocked_input = mocked_input.drop('name', axis=1)
        # add an extra source to the input
        to_add = mocked_input.iloc[0].copy()
        to_add['ra'] += 1.
        to_add['dec'] += 1.
        mocked_input = pd.concat(
            # need to transpose the series to a dataframe to concat
            [mocked_input, to_add.to_frame().T.reset_index(drop=True)]
        )

        mocked_input.name = group_name

        result = test_query._get_forced_fits(mocked_input)

        # call checks
        expected_skycoord = SkyCoord(
            mocked_input['ra'].to_numpy(),
            mocked_input['dec'].to_numpy(),
            unit=(u.deg, u.deg)
        )

        call_args = mocked_FP.return_value.measure.call_args.args

        assert np.all(call_args[0] == expected_skycoord)

        # result checks
        assert np.all(result['f_flux_int'] == input_fluxes)

    def test__init_sources(
        self,
        vast_query_psrj2129_fields: vtq.Query,
        selavy_cat: pd.DataFrame,
        mocker: MockerFixture
    ) -> None:
        """
        Tests the init sources method that groups the results into vasttools
        Source objects.

        The Source objects are not actually initialised and are instead
        mocked. The calls made are asserted against.

        Args:
            vast_query_psrj2129_fields: The dummy Query instance that includes
                a search for PSR J2129-04 with the included found fields data.
            selavy_cat: The dummy selavy catalogue.
            mocker: The pytest-mock mocker object.

        Returns:
            None
        """
        test_query = vast_query_psrj2129_fields
        selavy_df = selavy_cat(contain_pulsar=True)
        sources_df = pd.merge(
            test_query.fields_df,
            selavy_df,
            left_index=True,
            right_index=True
        )

        mocker_Source = mocker.patch('vasttools.query.Source')

        result = test_query._init_sources(sources_df)
        source_call_args = mocker_Source.call_args.args

        # assert on some of the source call arguments.
        assert source_call_args[0] == SkyCoord(
            ra=sources_df['ra'].iloc[0],
            dec=sources_df['dec'].iloc[0],
            unit=(u.deg, u.deg)
        )
        assert source_call_args[1] == 'PSR J2129-04'
        assert source_call_args[2] == ['1', '2']
        assert source_call_args[3] == ['VAST_2118-06A', 'VAST_2118-06A']
        assert source_call_args[4] == 'I'
        assert source_call_args[7].equals(sources_df)

    def test__check_for_duplicate_epochs(
        self,
        vast_query_psrj2129: vtq.Query
    ) -> None:
        """
        Tests the duplicate epochs renaming of epochs that is used with planet
        searches.

        Args:
            vast_query_psrj2129: The dummy Query instance that includes
                a search for PSR J2129-04.

        Returns:
            None
        """
        epochs = pd.Series(['1', '1', '1', '2', '3'])
        expected = np.array(['1-1', '1-2', '1-3', '2', '3'])
        result = vast_query_psrj2129._check_for_duplicate_epochs(epochs)

        assert np.all(expected == result.to_numpy())

    @pytest.mark.parametrize(
        "search_around",
        [(False), (True)],
        ids=('search_around_false', 'search_around_true')
    )
    def test_find_sources(
        self,
        vast_query_psrj2129_fields: vtq.Query,
        selavy_cat: pd.DataFrame,
        search_around: bool,
        mocker: MockerFixture
    ) -> None:
        """
        Smoke test for the user facing 'find_sources' method.

        All components of this method have been tested individually in the
        tests above. The from_pandas mocker will be used for the
        get_components part and the init_sources. The result is asserted
        against to check it was returned correctly.

        Args:
            vast_query_psrj2129_fields: The dummy Query instance that includes
                a search for PSR J2129-04 with the included found fields data.
            selavy_cat: The dummy selavy catalogue.
            mocker: The pytest-mock mocker object.

        Returns:
            None
        """

        mocker_validate_files = mocker.patch(
            'vasttools.query.Query._validate_files',
            return_value=True
        )
        test_query = vast_query_psrj2129_fields
        test_query.settings['search_around'] = search_around

        return_df = selavy_cat(
            contain_pulsar=True,
            add_detection=True,
            search_around=search_around,
            search_around_index=search_around
        )

        # comment is dropped as what is done in the function
        # the # column becomes 'distance' in search_around
        return_df = return_df.drop('comment', axis=1)

        dask_from_pandas_mocker = mocker.patch(
            'vasttools.query.dd.from_pandas',
        )

        (
            dask_from_pandas_mocker
            .return_value
            .groupby
            .return_value
            .apply
            .return_value
            .compute
            .return_value
        ) = return_df

        test_query.find_sources()

        # need to set index and merge with fields_df in the search around
        # case as this is performed in the find_sources function. Also this
        # is where the # column is renamed
        if search_around:
            return_df = return_df.set_index('index')
            return_df = test_query.fields_df.merge(
                return_df, how='inner', left_index=True, right_index=True
            )
            return_df = return_df.rename(columns={'#': 'distance'})

        assert test_query.results.equals(return_df)

    @pytest.mark.parametrize("epochs, racs, vast_p1, vast_p2, vast_full",
                             [(["0"], True, False, False, False),
                              (["1"], False, True, False, False),
                              (["17"], False, False, True, False),
                              (["23"], False, False, False, True),
                              (["0", "14", "28"], True, False, False, False),
                              (["0", "1"], True, True, False, False),
                              (["0", "17"], True, False, True, False),
                              (["0", "23"], True, False, False, True),
                              (["1", "17"], False, True, True, False),
                              (["0", "1", "17"], True, True, True, False),
                              (["0", "1", "17", "23"], True, True, True, True),
                              ],
                             ids=('racs-only',
                                  'p1-only',
                                  'p2-only',
                                  'full-only',
                                  'all-racs',
                                  'racs+p1',
                                  'racs+p2',
                                  'racs+full',
                                  'pilot-only',
                                  'racs+pilot',
                                  'all-data')
                             )
    def test__check_survey(
        self,
        vast_query_psrj2129: vtq.Query,
        epochs: List[str],
        racs: bool,
        vast_p1: bool,
        vast_p2: bool,
        vast_full: bool,
    ) -> None:
        """
        Test the survey check.

        Args:
            vast_query_psrj2129: The dummy Query instance that includes
                a search for PSR J2129-04 with the included found fields data.
            epochs: List of epochs to check.
            racs: Whether the epochs include a RACS epoch.
            vast_p1: Whether the epochs include a VAST-P1 epoch.
            vast_p2: Whether the epochs include a VAST-P2 epoch.
            vast_full: Whether the epochs include a full VAST survey epoch.

        Returns:
            None
        """

        vast_query_psrj2129._check_survey(epochs)

        assert vast_query_psrj2129.racs == racs
        assert vast_query_psrj2129.vast_p1 == vast_p1
        assert vast_query_psrj2129.vast_p2 == vast_p2
        assert vast_query_psrj2129.vast_full == vast_full

    @pytest.mark.parametrize("req_epochs, epochs_expected",
                             [("0", ["0"]),
                              ("3x", ["3x"]),
                              ("3", ["3x"]),
                              ("0,1", ["0", "1"]),
                              ("0,3x", ["0", "3x"]),
                              ("0,3", ["0", "3x"]),
                              ([0, 1], ["0", "1"]),
                              (0, ["0"]),
                              (["0", "1"], ["0", "1"]),
                              (["0", 1], ["0", "1"]),
                              ([1, "3x"], ["1", "3x"]),
                              ([1, "3"], ["1", "3x"]),
                              ("all", ["0", "1", "3x"]),
                              ("all-vast", ["1", "3x"]),
                              ],
                             ids=('single-str',
                                  'single-str-x-provided',
                                  'single-str-x-missing',
                                  'multiple-str',
                                  'multiple-str-x-provided',
                                  'multiple-str-x-missing',
                                  'single-int',
                                  'int-list',
                                  'str-list',
                                  'mixed-list',
                                  'mixed-list-x-provided',
                                  'mixed-list-x-missing',
                                  'all',
                                  'all-vast'
                                  )
                             )
    def test__get_epochs(
        self,
        vast_query_psrj2129: vtq.Query,
        req_epochs: Union[str, List[str], List[int]],
        epochs_expected,
        mocker: MockerFixture,
    ) -> None:
        """
        Test the get_epochs function.

        Args:
            vast_query_psrj2129: The dummy Query instance that includes
                a search for PSR J2129-04 with the included found fields data.
            req_epochs: The requested epochs.
            epochs_expected: The expected output of the function.

        Returns:
            None
        """
        mocked_released_epochs = {"0": "00", "1": "01", "3x": "03x"}

        mocker.patch("vasttools.query.RELEASED_EPOCHS",
                     new=mocked_released_epochs
                     )

        returned_epochs = vast_query_psrj2129._get_epochs(req_epochs)
        assert returned_epochs == epochs_expected
