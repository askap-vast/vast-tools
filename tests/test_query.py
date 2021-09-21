import astropy.units as u
import numpy as np
import os
import pandas as pd
import pytest

from astropy.coordinates import SkyCoord, Angle
from astropy.io import fits
from astropy.wcs import WCS
import importlib.resources
from mocpy import MOC
from pytest_mock import mocker
from radio_beam import Beam

from vasttools import RELEASED_EPOCHS
from vasttools.survey import load_field_centres, Fields
from vasttools.moc import VASTMOCS
import vasttools.query as vtq


@pytest.fixture
def pilot_moc_mocker():
    with importlib.resources.path(
        "vasttools.data.mocs",
        "VAST_PILOT_EPOCH01.moc.fits"
    ) as moc_path:
        moc = MOC.from_fits(moc_path)

    return moc


@pytest.fixture
def vast_query_psrj2129(pilot_moc_mocker, mocker):
    # For a reason I'm not sure of the moc opening in _build_catalog
    # cannot open the MOC file within the test. So here I mock the opening
    # and return the epoch 1 moc file. It seems to work fine outside of the
    # tests.
    test_dir = '/testing/folder'
    mocker_isdir = mocker.patch(
        'vasttools.query.os.path.isdir',
        return_value=True
    )
    mocker_moc_open = mocker.patch(
        'mocpy.MOC.from_fits',
        return_value = pilot_moc_mocker
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
def vast_fields_object_mock():
    """
    Note that id 4000 is made up.
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
                1893: 'VAST_2118-06A',
                1894: 'VAST_2118-06A',
                1907: 'VAST_2118-06A',
                3693: 'VAST_2143-06A',
                3694: 'VAST_2143-06A',
                3707: 'VAST_2143-06A',
                4000: 'VAST_0612-06A'
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
def fields_df_expected_result():
    def _get_result(add_files: bool = False):
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
                'planet': {
                    0: False,
                    1: False
                }
            }
        )

        if add_files:
            fields_df['selavy'] = [
                '/testing/folder/EPOCH01/COMBINED/STOKESI_SELAVY'
                '/VAST_2118-06A.EPOCH01.I.selavy.components.txt',
                '/testing/folder/EPOCH02/COMBINED/STOKESI_SELAVY'
                '/VAST_2118-06A.EPOCH02.I.selavy.components.txt'
            ]
            fields_df['image'] = [
                '/testing/folder/EPOCH01/COMBINED/STOKESI_IMAGES'
                '/VAST_2118-06A.EPOCH01.I.fits',
                '/testing/folder/EPOCH02/COMBINED/STOKESI_IMAGES'
                '/VAST_2118-06A.EPOCH02.I.fits',
            ]
            fields_df['rms'] = [
                '/testing/folder/EPOCH01/COMBINED/STOKESI_RMSMAPS'
                '/VAST_2118-06A.EPOCH01.I_rms.fits',
                '/testing/folder/EPOCH02/COMBINED/STOKESI_RMSMAPS'
                '/VAST_2118-06A.EPOCH02.I_rms.fits'
            ]

        return fields_df
    return _get_result


@pytest.fixture
def vast_query_psrj2129_fields(vast_query_psrj2129, fields_df_expected_result):
    def _add_fields_df(add_files: bool = True):
        vast_query_psrj2129.fields_found = True
        vast_query_psrj2129.fields_df = fields_df_expected_result(
            add_files=add_files
        )

        return vast_query_psrj2129
    return _add_fields_df


@pytest.fixture
def field_centres_mock():
    field_centres_df = pd.DataFrame(
        data={
            'field': {89: 'VAST_2118-06A', 94: 'VAST_2143-06A'},
            'centre-ra': {89: 319.65155983605507, 94: 325.8598931693551},
            'centre-dec': {89: -6.298205277661648, 94: -6.298205277661648}
        }
    )

    return field_centres_df


@pytest.fixture
def selavy_cat():
    def _get_selavy_cat(
        contain_pulsar: bool = False,
        search_around: bool = False,
        add_detection: bool = False
    ):
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
                '#': {0: np.nan, 1: np.nan},
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

        if search_around:
            selavy_df = selavy_df.append(selavy_df.loc[[0,0,0]])

        if add_detection:
            selavy_df['detection'] = [True, False]

        return selavy_df
    return _get_selavy_cat


class TestQuery:
    def test_init_failure_no_inputs(self):
        with pytest.raises(vtq.QueryInitError) as excinfo:
            query = vtq.Query()

        assert str(excinfo.value).startswith(
            "No coordinates or source names have been provided!"
        )

    def test_init_failure_no_forced_fits_and_search_around(self):
        with pytest.raises(vtq.QueryInitError) as excinfo:
            query = vtq.Query(
                source_names=['test'],
                forced_fits=True,
                search_around_coordinates=True
            )

        assert str(excinfo.value).startswith(
            "Forced fits and search around"
        )

    def test_init_failure_length_mismatch(self):
        with pytest.raises(vtq.QueryInitError) as excinfo:
            query = vtq.Query(
                coords=SkyCoord('00h42.5m', '+41d12m'),
                source_names=['test', 'test2'],
            )

        assert str(excinfo.value).startswith(
            "The number of entered source names"
        )

    def test_init_failure_invalid_planet(self):
        with pytest.raises(ValueError) as excinfo:
            query = vtq.Query(
                planets=['Earth']
            )

        assert str(excinfo.value) == "Invalid planet object provided!"

    def test_init_failure_base_folder(self):
        with pytest.raises(vtq.QueryInitError) as excinfo:
            query = vtq.Query(
                planets=['Mars']
            )

        assert str(excinfo.value).startswith(
            "The base folder directory could not be determined!"
        )

    def test_init_failure_base_folder_not_found(self):
        with pytest.raises(vtq.QueryInitError) as excinfo:
            test_dir = '/testing/folder'
            query = vtq.Query(
                planets=['Mars'],
                base_folder=test_dir
            )

        assert str(excinfo.value) == f"Base folder {test_dir} not found!"

    def test_init_failure_stokes_v_tiles(self, mocker):
        with pytest.raises(vtq.QueryInitError) as excinfo:
            isdir_mocker = mocker.patch(
                'vasttools.query.os.path.isdir',
                return_value=True
            )
            test_dir = '/testing/folder'
            query = vtq.Query(
                epochs='all-vast',
                planets=['Mars'],
                base_folder=test_dir,
                stokes='v',
                use_tiles=True
            )

        assert str(excinfo.value).startswith(
            "Problems found in query settings!"
        )

    def test_init_failure_no_sources_in_footprint(
        self,
        pilot_moc_mocker,
        mocker
    ):
        with pytest.raises(vtq.QueryInitError) as excinfo:
            isdir_mocker = mocker.patch(
                'vasttools.query.os.path.isdir',
                return_value=True
            )

            mocker_moc_open = mocker.patch(
                'mocpy.MOC.from_fits',
                return_value = pilot_moc_mocker
            )
            test_dir = '/testing/folder'
            test_coords = SkyCoord(
                ['00h42.5m', '17h42.5m'],
                ['+41d12m', '-82d12m']
            )

            query = vtq.Query(
                epochs='all-vast',
                coords=test_coords,
                base_folder=test_dir,
            )

        assert str(excinfo.value) == (
            'No sources remaining. None of the entered coordinates'
            ' are found in the VAST Pilot survey footprint!'
        )

    def test_init_settings(self, mocker):
        isdir_mocker = mocker.patch(
            'vasttools.query.os.path.isdir',
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
            'tiles': use_tiles
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
            output_dir=output_dir
        )


        assert query.settings == expected_settings

    def test__field_matching(
        self,
        vast_query_psrj2129,
        vast_fields_object_mock,
        field_centres_mock,
        mocker
    ):
        field_sc = SkyCoord(
            vast_fields_object_mock["RA_HMS"],
            vast_fields_object_mock["DEC_DMS"],
            unit=(u.hourangle, u.deg)
        )

        field_centres_sc = SkyCoord(
            field_centres_mock["centre-ra"],
            field_centres_mock["centre-dec"],
            unit=(u.deg, u.deg)
        )
        field_centre_names = field_centres_mock.field

        row = vast_query_psrj2129.query_df.iloc[0]
        results = vast_query_psrj2129._field_matching(
            row,
            field_sc,
            vast_fields_object_mock.FIELD_NAME,
            field_centres_sc,
            field_centre_names
        )

        assert np.all(results[0] == np.array(
            ['VAST_2118-06A', 'VAST_2143-06A']
        ))
        assert results[1] == 'VAST_2118-06A'
        assert results[2] == ['1', '2']

    def test_find_fields(
        self, vast_query_psrj2129, fields_df_expected_result, mocker):

        mocked_field_matching_results = (
            [np.array(['VAST_2118-06A', 'VAST_2143-06A'], dtype='object')],
            'VAST_2118-06A',
            [['1', '2']],
            [[
                ['1', 'VAST_2118-06A', 9668, '2019-08-27 18:52:00.556'],
                ['2', 'VAST_2118-06A', 10342, '2019-10-30 10:11:56.913']
            ]],
            [[9668, 10342]],
            [['2019-08-27 18:52:00.556', '2019-10-30 10:11:56.913']]
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

    def test__add_files_combined(self, vast_query_psrj2129_fields, mocker):
        test_query = vast_query_psrj2129_fields()
        results = test_query._add_files(test_query.fields_df.loc[0])

        expected_results = (
            '/testing/folder/EPOCH01/COMBINED/STOKESI_SELAVY'
            '/VAST_2118-06A.EPOCH01.I.selavy.components.txt',
            '/testing/folder/EPOCH01/COMBINED/STOKESI_IMAGES'
            '/VAST_2118-06A.EPOCH01.I.fits',
            '/testing/folder/EPOCH01/COMBINED/STOKESI_RMSMAPS'
            '/VAST_2118-06A.EPOCH01.I_rms.fits'
        )

        assert results == expected_results

    def test__add_files_tiles(self, vast_query_psrj2129_fields, mocker):
        test_query = vast_query_psrj2129_fields()

        test_query.settings['tiles'] = True

        results = test_query._add_files(
            test_query.fields_df.loc[0]
        )

        expected_results = (
            '/testing/folder/EPOCH01/TILES/STOKESI_SELAVY'
            '/selavy-image.i.SB9668.cont.VAST_2118-06A.linmos'
            '.taylor.0.restored.components.txt',
            '/testing/folder/EPOCH01/TILES/STOKESI_IMAGES'
            '/image.i.SB9668.cont.VAST_2118-06A.linmos'
            '.taylor.0.restored.fits',
            'N/A'
        )

        assert results == expected_results

    def test__add_files_stokesv_combined(
        self, vast_query_psrj2129_fields, mocker
    ):
        test_query = vast_query_psrj2129_fields()
        test_query.settings['stokes'] = 'V'
        test_query.fields_df['stokes'] = 'V'
        results = test_query._add_files(test_query.fields_df.loc[0])

        expected_results = (
            '/testing/folder/EPOCH01/COMBINED/STOKESV_SELAVY'
            '/VAST_2118-06A.EPOCH01.V.selavy.components.txt',
            '/testing/folder/EPOCH01/COMBINED/STOKESV_IMAGES'
            '/VAST_2118-06A.EPOCH01.V.fits',
            '/testing/folder/EPOCH01/COMBINED/STOKESV_RMSMAPS'
            '/VAST_2118-06A.EPOCH01.V_rms.fits'
        )

        assert results == expected_results


    def test__get_components_detection(
        self,
        vast_query_psrj2129_fields,
        selavy_cat,
        mocker
    ):
        test_query = vast_query_psrj2129_fields(add_files=True)

        mocked_input = test_query.fields_df.groupby('selavy').get_group(
            test_query.fields_df['selavy'].iloc[0]
        )

        mocker_selavy = mocker.patch(
            'vasttools.query.pd.read_fwf',
            return_value=selavy_cat(contain_pulsar=True)
        )

        result = test_query._get_components(mocked_input)

        expected = selavy_cat(contain_pulsar=True).drop(1)
        expected['detection'] = True

        assert expected.equals(result)


    def test__get_components_non_detection(
        self,
        vast_query_psrj2129_fields,
        selavy_cat,
        mocker
    ):
        test_query = vast_query_psrj2129_fields(add_files=True)

        mocked_input = test_query.fields_df.groupby('selavy').get_group(
            test_query.fields_df['selavy'].iloc[0]
        )

        mocker_selavy = mocker.patch(
            'vasttools.query.pd.read_fwf',
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
        ) = np.array([0.001,])

        result = test_query._get_components(mocked_input)

        assert result.shape[0] == 1
        assert result.iloc[0]['rms_image'] == 1.
        assert result.iloc[0]['detection'] == False

    def test__get_components_search_around(
        self,
        vast_query_psrj2129_fields,
        selavy_cat,
        mocker
    ):
        # manually add the files on
        test_query = vast_query_psrj2129_fields(add_files=True)
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
        vast_query_psrj2129_fields,
        mocker
    ):
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

        test_query = vast_query_psrj2129_fields(add_files=True)
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
        mocked_input = mocked_input.append(to_add)

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
        vast_query_psrj2129_fields,
        selavy_cat,
        mocker
    ):
        test_query = vast_query_psrj2129_fields(add_files=True)
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
        assert source_call_args[7].equals(sources_df.drop('#', axis=1))

    def test__check_for_duplicate_epochs(self, vast_query_psrj2129):
        epochs = pd.Series(['1', '1', '1', '2', '3'])
        expected = np.array(['1-1', '1-2', '1-3', '2', '3'])
        result = vast_query_psrj2129._check_for_duplicate_epochs(epochs)

        assert np.all(expected == result.to_numpy())

    def test_find_sources(
        self,
        vast_query_psrj2129_fields,
        selavy_cat,
        mocker
    ):
        # smoke test for find sources all other components are tested
        # in the above tests. The from_pandas mocker will be used for the
        # get_components bit and the init_sources.
        test_query = vast_query_psrj2129_fields()

        return_df = selavy_cat(
            contain_pulsar=True,
            add_detection=True
        ).drop(['#', 'comment'], axis=1)
        # these are dropped to force a return value.

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

        assert test_query.results.equals(return_df)
