import astropy.units as u
import copy
import numpy as np
import os
import pandas as pd
import pytest
import vaex

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from mocpy import MOC
from pathlib import Path
from pytest_mock import mocker
from random import sample

from vasttools import RELEASED_EPOCHS
import vasttools.pipeline as vtp


TEST_DATA_DIR = Path(__file__).resolve().parent / 'data'


@pytest.fixture
def mocked_pipeline_object(mocker):
    expected_path = '/path/to/pipelineruns/'
    mocker_getenv = mocker.patch(
        'os.getenv', return_value=expected_path
    )
    mock_isdir = mocker.patch('os.path.isdir', return_value=True)

    pipe = vtp.Pipeline()

    return pipe


def mocked_pipeline_images():
    filepath = TEST_DATA_DIR / 'test_images.csv'

    images_df = pd.read_csv(filepath)
    images_df['datetime'] = pd.to_datetime(images_df['datetime'])

    return images_df


def mocked_pipeline_bands():
    bands_df = pd.DataFrame(
        data={
            'id': {0: 1},
            'name': {0: '887'},
            'frequency': {0: 887},
            'bandwidth': {0: 0}
        }
    )

    return bands_df


def mocked_pipeline_skyregions():
    skyregions_df = pd.DataFrame(
        data={
            'id': {1: 2},
            'centre_ra': {1: 319.65225849781234},
            'centre_dec': {1: -6.298899722106099},
            'width_ra': {1: 9.534722222221612},
            'width_dec': {1: 9.529166666666057},
            'xtr_radius': {1: 6.740102840758925},
            'x': {1: 0.7575282038568381},
            'y': {1: -0.6435165810439755},
            'z': {1: -0.10971522356949984}
        }
    )

    return skyregions_df


def mocked_pipeline_relations():
    relations_df = pd.DataFrame(
        data={
            'from_source_id': {6: 729, 74: 2251},
            'to_source_id': {6: 2251, 74: 729}
        }
    )

    return relations_df


def mocked_pipeline_sources():
    filepath = TEST_DATA_DIR / 'test_sources.csv'

    sources_df = pd.read_csv(filepath, index_col='id')

    return sources_df


def mocked_pipeline_associations():
    associations_df = pd.DataFrame(
        data={
            'source_id': {
                22402: 729,
                22403: 729,
                22404: 729,
                22405: 729,
                22406: 729,
                22407: 2251,
                22408: 2251,
                22409: 2251,
                22410: 2251,
                22411: 2251,
                22472: 730,
                22473: 730,
                22474: 730,
                22475: 730,
                22476: 730
            },
            'meas_id': {
                22402: 7881,
                22403: 23917,
                22404: 23955,
                22405: 40840,
                22406: 40908,
                22407: 7882,
                22408: 23917,
                22409: 23955,
                22410: 40840,
                22411: 40908,
                22472: 7906,
                22473: 15671,
                22474: 24008,
                22475: 32441,
                22476: 40995
            },
            'd2d': {
                22402: 0.0,
                22403: 3.7733754759011897,
                22404: 2.943794115878134,
                22405: 3.1116211278179327,
                22406: 1.626252891950026,
                22407: 0.0,
                22408: 8.078039449654458,
                22409: 3.5230660519200083,
                22410: 2.7889744783623303,
                22411: 2.2741280251926,
                22472: 0.0,
                22473: 1.1626143525209363,
                22474: 0.4697219348160675,
                22475: 2.080611695755226,
                22476: 1.4488538103139523},
            'dr': {
                22402: 0.0,
                22403: 0.0,
                22404: 0.0,
                22405: 0.0,
                22406: 0.0,
                22407: 0.0,
                22408: 0.0,
                22409: 0.0,
                22410: 0.0,
                22411: 0.0,
                22472: 0.0,
                22473: 0.0,
                22474: 0.0,
                22475: 0.0,
                22476: 0.0
            }
        }
    )

    return associations_df


def mocked_pipeline_measurements_vaex():
    filepath = TEST_DATA_DIR / 'test_measurements_vaex.csv'
    # TODO: check measurements_df = vaex.read_csv(filepath)
    #       Annoyingly vaex fails to read directly from within the testing
    #       environment - it thinks the file is a directory!
    measurements_df = pd.read_csv(filepath)
    measurements_df = vaex.from_pandas(measurements_df)

    return measurements_df


def mocked_pipeline_measurements():
    filepath = TEST_DATA_DIR / 'test_measurements.csv'
    measurements_df = pd.read_csv(filepath, index_col='Unnamed: 0')

    return measurements_df


def mocked_pipeline_measurement_pairs(*args, **kwargs):
    filepath = TEST_DATA_DIR / 'test_measurement_pairs.csv'
    measurement_pairs_df = pd.read_csv(filepath)

    return measurement_pairs_df


def mocked_pipeline_measurement_pairs_vaex(*args, **kwargs):
    filepath = TEST_DATA_DIR / 'test_measurement_pairs.csv'
    measurements_pairs_df = pd.read_csv(filepath)
    measurements_pairs_df = vaex.from_pandas(measurements_pairs_df)

    return measurements_pairs_df


@pytest.fixture
def mocked_pipeline_pairs_df():
    filepath = TEST_DATA_DIR / 'test_pairs_df_result.csv'
    pairs_df = pd.read_csv(filepath, index_col='id')
    pairs_df['datetime_a'] = pd.to_datetime(
        pairs_df['datetime_a']
    )
    pairs_df['datetime_b'] = pd.to_datetime(
        pairs_df['datetime_b']
    )
    pairs_df['td'] = pd.to_timedelta(pairs_df['td'])

    return pairs_df


def load_parquet_side_effect(
    value, **kwargs
):
    if 'bands.parquet' in value:
        return mocked_pipeline_bands()
    elif 'associations.parquet' in value:
        return mocked_pipeline_associations()
    elif 'images.parquet' in value:
        return mocked_pipeline_images()
    elif 'relations.parquet' in value:
        return mocked_pipeline_relations()
    elif 'skyregions.parquet' in value:
        return mocked_pipeline_skyregions()
    elif 'sources.parquet' in value:
        return mocked_pipeline_sources()
    else:
        raise ValueError('File not recognised.')


@pytest.fixture
def mocked_PipeAnalysis(
    mocked_pipeline_object,
    mocker
):
    """
    Because the raw pipeline outputs are processed in the load run function
    it is easier to test while creating the object each time. It is a little
    inefficient and really the pipeline process should be refactored slightly
    to support better testing.
    """
    mock_isdir = mocker.patch('os.path.isdir', return_value=True)
    mock_isfile = mocker.patch('os.path.isfile', return_value=False)
    pandas_read_parquet_mocker = mocker.patch(
        'vasttools.pipeline.pd.read_parquet',
        side_effect=load_parquet_side_effect
    )
    dask_read_parquet_mocker = mocker.patch(
        'vasttools.pipeline.dd.read_parquet',
    )
    dask_read_parquet_mocker.return_value.compute.return_value = (
        mocked_pipeline_measurements()
    )

    pipe = mocked_pipeline_object
    run_name = 'test_run'
    run = pipe.load_run(run_name)

    return run


@pytest.fixture
def mocked_PipeAnalysis_wtwoepoch(
    mocked_PipeAnalysis,
    mocker
):
    pandas_read_parquet_mocker = mocker.patch(
        'vasttools.pipeline.pd.read_parquet',
        side_effect=mocked_pipeline_measurement_pairs
    )

    mocked_PipeAnalysis.load_two_epoch_metrics()

    return mocked_PipeAnalysis


@pytest.fixture
def mocked_PipeAnalysis_vaex(
    mocked_pipeline_object,
    mocker
):
    """
    Because the raw pipeline outputs are processed in the load run function
    it is easier to test while creating the object each time. It is a little
    inefficient and really the pipeline process should be refactored slightly
    to support better testing.
    """
    mock_isdir = mocker.patch('os.path.isdir', return_value=True)
    mock_isfile = mocker.patch('os.path.isfile', return_value=True)
    pandas_read_parquet_mocker = mocker.patch(
        'vasttools.pipeline.pd.read_parquet',
        side_effect=load_parquet_side_effect
    )
    vaex_open_mocker = mocker.patch(
        'vasttools.pipeline.vaex.open',
        return_value=mocked_pipeline_measurements_vaex()
    )

    pipe = mocked_pipeline_object
    run_name = 'test_run'
    run = pipe.load_run(run_name)

    return run


@pytest.fixture
def mocked_PipeAnalysis_vaex_wtwoepoch(
    mocked_PipeAnalysis,
    mocker
):
    vaex_open_mocker = mocker.patch(
        'vasttools.pipeline.vaex.open',
        side_effect=mocked_pipeline_measurement_pairs_vaex
    )

    mocked_PipeAnalysis.load_two_epoch_metrics()

    return mocked_PipeAnalysis


@pytest.fixture
def expected_sources_skycoord():
    sources = mocked_pipeline_sources()
    sources_sc = SkyCoord(
        ra=sources['wavg_ra'],
        dec=sources['wavg_dec'],
        unit=(u.deg, u.deg)
    )

    return sources_sc


@pytest.fixture
def expected_source_measurements_pd(mocked_PipeAnalysis):
    def _filter_source(id: int):
        meas = mocked_PipeAnalysis.measurements
        meas = meas.loc[meas['source'] == id]

        return meas

    return _filter_source


@pytest.fixture
def filter_moc():
    coords = SkyCoord(
        ra=[321.0, 322.0, 322.0, 321.0],
        dec=[-7., -7., -6., -6.],
        unit=(u.deg, u.deg)
    )

    moc = MOC.from_polygon_skycoord(coords, 9)

    return moc


@pytest.fixture
def gen_measurement_pairs_df(mocked_PipeAnalysis_wtwoepoch):
    def _gen_df(epoch_id: int = 2):
        epoch_key = (
            mocked_PipeAnalysis_wtwoepoch
            .pairs_df.loc[epoch_id]['pair_epoch_key']
        )

        measurement_pairs_df = (
            mocked_PipeAnalysis_wtwoepoch.measurement_pairs_df.loc[
                mocked_PipeAnalysis_wtwoepoch.measurement_pairs_df[
                    'pair_epoch_key'
                ] == epoch_key
            ]
        ).copy()

        return measurement_pairs_df

    return _gen_df


@pytest.fixture
def gen_sources_metrics_df():
    data = pd.DataFrame(
        data={
            'eta_peak': 10**(np.random.default_rng().normal(0.5, 1., 10000)),
            'v_peak': 10**(np.random.default_rng().normal(1., 2., 10000)),
            'eta_int': 10**(np.random.default_rng().normal(1.5, 3., 10000)),
            'v_int': 10**(np.random.default_rng().normal(2., 4., 10000)),
            'max_flux_peak': np.random.rand(10000),
            'avg_flux_peak': np.random.rand(10000),
            'n_selavy': 5
        }
    )

    return data


class TestPipeline:
    def test_init(self, mocker):
        expected_path = '/path/to/pipelineruns/'
        mocker_getenv = mocker.patch(
            'os.getenv', return_value=expected_path
        )
        mock_isdir = mocker.patch('os.path.isdir', return_value=True)

        pipe = vtp.Pipeline()

        assert pipe.project_dir == expected_path

    def test_init_projectdir(self, mocker):
        expected_path = '/path/to/projectdir/'
        mocker_abspath = mocker.patch(
            'os.path.abspath', return_value=expected_path
        )
        mock_isdir = mocker.patch('os.path.isdir', return_value=True)

        pipe = vtp.Pipeline(project_dir=expected_path)

        assert pipe.project_dir == expected_path

    def test_init_env_fail(self, mocker):
        expected_path = '/path/to/pipelineruns/'
        mocker_getenv = mocker.patch(
            'os.getenv', return_value=None
        )

        with pytest.raises(vtp.PipelineDirectoryError) as excinfo:
            pipe = vtp.Pipeline()

        assert str(excinfo.value).startswith(
            "The pipeline run directory could not be determined!"
        )

    def test_init_project_dir_fail(self, mocker):
        expected_path = '/path/to/projectdir/'
        mocker_abspath = mocker.patch(
            'os.path.abspath', return_value=expected_path
        )
        mock_isdir = mocker.patch('os.path.isdir', return_value=False)

        with pytest.raises(vtp.PipelineDirectoryError) as excinfo:
            pipe = vtp.Pipeline(project_dir=expected_path)

        assert str(excinfo.value).startswith("Pipeline run directory")

    def test_list_piperuns(self, mocked_pipeline_object, mocker):
        expected_path = '/path/to/pipelineruns'

        expected_result = ['job1', 'job2']

        mocker_glob = mocker.patch(
            'glob.glob', return_value=[
                os.path.join(expected_path, 'job1'),
                os.path.join(expected_path, 'job2'),
                os.path.join(expected_path, 'images')
            ]
        )

        result = mocked_pipeline_object.list_piperuns()

        mocker_glob.assert_called_once_with(os.path.join(expected_path, '*'))
        assert result == expected_result

    def test_list_images(self, mocked_pipeline_object, mocker):
        """
        Docstring

        Assumes pipeline run directory has 'images'.
        """
        expected_path = '/path/to/pipelineruns'

        expected_result = ['image1', 'image2']

        mocker_glob = mocker.patch(
            'glob.glob', return_value=[
                os.path.join(expected_path, 'images', 'image1'),
                os.path.join(expected_path, 'images', 'image2'),
            ]
        )

        result = mocked_pipeline_object.list_images()

        mocker_glob.assert_called_once_with(os.path.join(
            expected_path, 'images', '*'
        ))
        assert result == expected_result

    def test_load_run_dir_fail(
        self,
        mocked_pipeline_object,
        mocker
    ):
        mock_isdir = mocker.patch('os.path.isdir', return_value=False)

        pipe = mocked_pipeline_object

        with pytest.raises(OSError) as excinfo:
            pipe.load_run('test')

    def test_load_run_no_vaex(
        self,
        mocked_pipeline_object,
        mocker
    ):
        mock_isdir = mocker.patch('os.path.isdir', return_value=True)
        mock_isfile = mocker.patch('os.path.isfile', return_value=False)
        pandas_read_parquet_mocker = mocker.patch(
            'vasttools.pipeline.pd.read_parquet',
            side_effect=load_parquet_side_effect
        )
        dask_read_parquet_mocker = mocker.patch(
            'vasttools.pipeline.dd.read_parquet',
        )
        dask_read_parquet_mocker.return_value.compute.return_value = (
            mocked_pipeline_measurements()
        )

        pipe = mocked_pipeline_object
        run_name = 'test_run'
        run = pipe.load_run(run_name)

        assert run.name == run_name
        assert run._vaex_meas == False

    def test_load_run_no_vaex_check_columns(
        self,
        mocked_pipeline_object,
        mocker
    ):
        mock_isdir = mocker.patch('os.path.isdir', return_value=True)
        mock_isfile = mocker.patch('os.path.isfile', return_value=False)
        pandas_read_parquet_mocker = mocker.patch(
            'vasttools.pipeline.pd.read_parquet',
            side_effect=load_parquet_side_effect
        )
        dask_read_parquet_mocker = mocker.patch(
            'vasttools.pipeline.dd.read_parquet',
        )
        dask_read_parquet_mocker.return_value.compute.return_value = (
            mocked_pipeline_measurements()
        )

        pipe = mocked_pipeline_object
        run_name = 'test_run'
        run = pipe.load_run(run_name)

        assert 'centre_ra' in run.images.columns
        assert run.images.shape[1] == 29
        assert run.measurements.shape[1] == 42

    def test_load_run_vaex(
        self,
        mocked_pipeline_object,
        mocker
    ):
        mock_isdir = mocker.patch('os.path.isdir', return_value=True)
        mock_isfile = mocker.patch('os.path.isfile', return_value=True)
        pandas_read_parquet_mocker = mocker.patch(
            'vasttools.pipeline.pd.read_parquet',
            side_effect=load_parquet_side_effect
        )
        vaex_open_mocker = mocker.patch(
            'vasttools.pipeline.vaex.open',
            return_value = mocked_pipeline_measurements_vaex()
        )

        pipe = mocked_pipeline_object
        run_name = 'test_run'
        run = pipe.load_run(run_name)

        assert run.name == run_name
        assert run._vaex_meas == True


class TestPipeAnalysis:
    def test_combine_with_run(
        self,
        mocked_PipeAnalysis
    ):
        run_2 = copy.deepcopy(mocked_PipeAnalysis)
        run_2.sources.index = [100, 200, 300]
        new_run = mocked_PipeAnalysis.combine_with_run(run_2)

        assert new_run.sources.shape[0] == 6

    def test_combine_with_run_pandas_vaex(
        self,
        mocked_PipeAnalysis,
        mocked_PipeAnalysis_vaex
    ):
        mocked_PipeAnalysis.sources.index = [100, 200, 300]
        new_run = mocked_PipeAnalysis.combine_with_run(
            mocked_PipeAnalysis_vaex
        )

        assert new_run.sources.shape[0] == 6

    def test_combine_with_run_both_vaex(
        self,
        mocked_PipeAnalysis_vaex
    ):
        run_2 = copy.deepcopy(mocked_PipeAnalysis_vaex)
        run_2.sources.index = [100, 200, 300]
        new_run = mocked_PipeAnalysis_vaex.combine_with_run(run_2)

        assert new_run.sources.shape[0] == 6

    def test_pipeanalysis_get_sources_skycoord(
        self,
        mocked_PipeAnalysis,
        mocker
    ):
        pipe = mocked_PipeAnalysis

        expected = mocked_pipeline_sources()
        expected = SkyCoord(
            ra=expected['wavg_ra'],
            dec=expected['wavg_dec'],
            unit=(u.deg, u.deg)
        )

        result = pipe.get_sources_skycoord()

        assert np.all(result == expected)

    def test_pipeanalysis_get_sources_skycoord_user_sources(
        self,
        mocked_PipeAnalysis,
        mocker
    ):
        pipe = mocked_PipeAnalysis

        test_sources_df = pd.DataFrame(
            data={
                'ra': [0.0, 90., 180.],
                'dec': [0.0, 10., 20.]
            }
        )
        expected = SkyCoord(
            ra=test_sources_df['ra'],
            dec=test_sources_df['dec'],
            unit=(u.deg, u.deg)
        )

        result = pipe.get_sources_skycoord(
            user_sources = test_sources_df,
            ra_col='ra',
            dec_col='dec'
        )

        assert np.all(result == expected)

    def test_pipeanalysis_get_sources_skycoord_user_sources_hms(
        self,
        mocked_PipeAnalysis,
        mocker
    ):
        pipe = mocked_PipeAnalysis

        test_sources_df = pd.DataFrame(
            data={
                'wavg_ra': ['00:00:00', "06:00:00", "12:00:00"],
                'wavg_dec': [0.0, 10., 20.]
            }
        )
        expected = SkyCoord(
            ra=test_sources_df['wavg_ra'],
            dec=test_sources_df['wavg_dec'],
            unit=(u.hourangle, u.deg)
        )

        result = pipe.get_sources_skycoord(
            user_sources = test_sources_df,
            ra_unit=u.hourangle
        )

        assert np.all(result == expected)

    def test_pipeanalysis_get_source(
        self,
        mocked_PipeAnalysis,
        expected_sources_skycoord,
        expected_source_measurements_pd,
        mocker
    ):
        mocker_source = mocker.patch(
            'vasttools.pipeline.Source',
            return_value = -99
        )

        # Source init args for reference:
        #     self,
        #     coord: SkyCoord,
        #     name: str,
        #     epochs: List[str],
        #     fields: List[str],
        #     stokes: str,
        #     primary_field: str,
        #     crossmatch_radius: Angle,
        #     measurements: pd.DataFrame,
        #     base_folder: str,
        #     image_type: str = "COMBINED",
        #     islands: bool = False,
        #     outdir: str = ".",
        #     planet: bool = False,
        #     pipeline: bool = False,
        #     tiles: bool = False,
        #     forced_fits: bool = False,

        expected_source_name = "VAST {}".format(
            expected_sources_skycoord[0].to_string(
                "hmsdms", sep='', precision=1
            ).replace(
                " ", ""
            )[:15]
        )
        expected_epochs = ['1', '2', '3', '4', '5']
        expected_fields = [
            'test_run',
            'test_run',
            'test_run',
            'test_run',
            'test_run'
        ]
        expected_stokes = 'I'
        expected_primary_field = None
        expected_crossmatch_radius = None
        expected_measurements = expected_source_measurements_pd(729)
        expected_source_base_folder = None
        expected_source_image_type = None
        expected_source_outdir = "."

        expected_call = mocker.call(
            expected_sources_skycoord,
            expected_source_name,
            expected_epochs,
            expected_fields,
            expected_stokes,
            expected_primary_field,
            expected_crossmatch_radius,
            expected_measurements,
            expected_source_base_folder,
            expected_source_image_type,
            islands=False,
            outdir=expected_source_outdir,
            pipeline=True
        )

        the_source = mocked_PipeAnalysis.get_source(729)

        mocker_source.assert_called_once()

        mocker_calls = mocker_source.call_args
        assert mocker_calls.kwargs == expected_call.kwargs
        assert mocker_calls.args[7].shape[1] == 51
        assert mocker_calls.args[0] == expected_sources_skycoord[0]
        assert the_source == -99

    def test_pipeanalysis_get_source_vaex(
        self,
        mocked_PipeAnalysis_vaex,
        expected_sources_skycoord,
        expected_source_measurements_pd,
        mocker
    ):
        mocker_source = mocker.patch(
            'vasttools.pipeline.Source',
            return_value = -99
        )

        # Source init args for reference:
        #     self,
        #     coord: SkyCoord,
        #     name: str,
        #     epochs: List[str],
        #     fields: List[str],
        #     stokes: str,
        #     primary_field: str,
        #     crossmatch_radius: Angle,
        #     measurements: pd.DataFrame,
        #     base_folder: str,
        #     image_type: str = "COMBINED",
        #     islands: bool = False,
        #     outdir: str = ".",
        #     planet: bool = False,
        #     pipeline: bool = False,
        #     tiles: bool = False,
        #     forced_fits: bool = False,

        expected_source_name = "VAST {}".format(
            expected_sources_skycoord[0].to_string(
                "hmsdms", sep='', precision=1
            ).replace(
                " ", ""
            )[:15]
        )
        expected_epochs = ['1', '2', '3', '4', '5']
        expected_fields = [
            'test_run',
            'test_run',
            'test_run',
            'test_run',
            'test_run'
        ]
        expected_stokes = 'I'
        expected_primary_field = None
        expected_crossmatch_radius = None
        expected_measurements = expected_source_measurements_pd(729)
        expected_source_base_folder = None
        expected_source_image_type = None
        expected_source_outdir = "."

        expected_call = mocker.call(
            expected_sources_skycoord,
            expected_source_name,
            expected_epochs,
            expected_fields,
            expected_stokes,
            expected_primary_field,
            expected_crossmatch_radius,
            expected_measurements,
            expected_source_base_folder,
            expected_source_image_type,
            islands=False,
            outdir=expected_source_outdir,
            pipeline=True
        )

        the_source = mocked_PipeAnalysis_vaex.get_source(729)

        mocker_source.assert_called_once()

        mocker_calls = mocker_source.call_args
        assert mocker_calls.kwargs == expected_call.kwargs
        assert mocker_calls.args[7].shape[1] == 51
        assert mocker_calls.args[0] == expected_sources_skycoord[0]
        assert the_source == -99

    def test_pipeanalysis_load_two_epoch_metrics_pandas(
        self,
        mocked_PipeAnalysis,
        mocked_pipeline_pairs_df,
        mocker
    ):
        pandas_read_parquet_mocker = mocker.patch(
            'vasttools.pipeline.pd.read_parquet',
            side_effect=mocked_pipeline_measurement_pairs
        )

        mocked_PipeAnalysis.load_two_epoch_metrics()

        assert mocked_PipeAnalysis.pairs_df.equals(mocked_pipeline_pairs_df)
        assert mocked_PipeAnalysis.measurement_pairs_df.shape[0] == 30

    def test_pipeanalysis_load_two_epoch_metrics_vaex(
        self,
        mocked_PipeAnalysis_vaex,
        mocked_pipeline_pairs_df,
        mocker
    ):
        vaex_open_mocker = mocker.patch(
            'vasttools.pipeline.vaex.open',
            side_effect=mocked_pipeline_measurement_pairs_vaex
        )

        mocked_PipeAnalysis_vaex.load_two_epoch_metrics()

        assert mocked_PipeAnalysis_vaex.pairs_df.equals(
            mocked_pipeline_pairs_df
        )
        assert mocked_PipeAnalysis_vaex.measurement_pairs_df.shape[0] == 30

    @pytest.mark.parametrize("row, kwargs, expected", [
        (
            pd.Series({
                'DATEOBS': pd.Timestamp('2022-01-01 00:00'),
                'duration': 12600
            }),
            {'every_hour': False},
            [
                pd.Timestamp('2022-01-01 00:00'),
                pd.Timestamp('2022-01-01 03:30')
            ]
        ),
        (
            pd.Series({
                'DATEOBS': pd.Timestamp('2022-01-01 00:00'),
                'duration': 12600
            }),
            {'every_hour': True},
            [
                pd.Timestamp('2022-01-01 00:00'),
                pd.Timestamp('2022-01-01 01:00'),
                pd.Timestamp('2022-01-01 02:00'),
                pd.Timestamp('2022-01-01 03:00')
            ]
        )
    ])
    def test__add_times(
        self,
        row,
        kwargs,
        expected,
        mocked_PipeAnalysis
    ):
        result = mocked_PipeAnalysis._add_times(row, **kwargs)

        assert result == expected

    def test_check_for_planets(
        self,
        mocked_PipeAnalysis,
        mocker
    ):
        dask_from_pandas_mocker = mocker.patch(
            'vasttools.pipeline.dd.from_pandas',
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
        ) = pd.DataFrame(data={'planet_test': [True,]})

        expected_planet_counts = pd.Series({
            'Sun': 20,
            'Moon': 20,
            'Mercury': 10,
            'Venus': 10,
            'Mars': 10,
            'Jupiter': 10,
            'Saturn': 10,
            'Uranus': 10,
            'Neptune': 10,
        })

        result = mocked_PipeAnalysis.check_for_planets()
        call = dask_from_pandas_mocker.call_args.args[0]

        dask_from_pandas_mocker.assert_called_once()
        assert call.isnull().values.any() == False
        assert call.planet.value_counts().equals(expected_planet_counts)

    def test_filter_by_moc(
        self,
        mocked_PipeAnalysis,
        filter_moc,
        mocker
    ):
        result = mocked_PipeAnalysis.filter_by_moc(filter_moc)

        assert result.sources.shape[0] == 2
        assert result.measurements.shape[0] == 10

    def test__distance_from_edge(self, mocked_PipeAnalysis):
        input_array = np.array(
            [[0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0]]
        )

        expected = np.array(
            [[0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 2, 2, 1, 0],
            [0, 1, 2, 2, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0]]
        )

        result = mocked_PipeAnalysis._distance_from_edge(input_array)

        assert np.all(result == expected)

    def test__create_moc_from_fits(
        self,
        mocked_PipeAnalysis,
        mocker
    ):
        image_data = np.ones((4,4), dtype=np.float32)
        image_data= np.pad(
            image_data, pad_width=1, mode='constant', constant_values=np.nan
        )

        hdu = fits.PrimaryHDU(data=image_data)

        hdu.header['RADESYS'] = "ICRS"
        hdu.header['CTYPE1'] = "RA---SIN"
        hdu.header['CUNIT1'] = "deg"
        hdu.header['CRVAL1'] = 0.0
        hdu.header['CRPIX1'] = 2.0
        hdu.header['CD1_1'] = 1.0
        hdu.header['CD1_2'] = 0.0
        hdu.header['CTYPE2'] = "DEC--SIN"
        hdu.header['CUNIT2'] = "deg"
        hdu.header['CRVAL2'] = 0.0
        hdu.header['CRPIX2'] = 2.0
        hdu.header['CD2_1'] = 0.0
        hdu.header['CD2_2'] = 1.0

        image_wcs = WCS(hdu.header)

        image_mocker = mocker.patch('vasttools.pipeline.Image')
        (image_mocker.return_value).data = image_data
        (image_mocker.return_value).wcs = image_wcs

        moc_from_polygon_skycoord_mocker = mocker.patch(
            'mocpy.MOC.from_polygon_skycoord',
            return_value=-99
        )

        result = mocked_PipeAnalysis._create_moc_from_fits('test.fits')
        called_coords = moc_from_polygon_skycoord_mocker.call_args.args[0]
        pixels = image_wcs.world_to_array_index(called_coords)

        assert len(pixels[0]) == 12
        assert np.all(pixels != 0)
        assert result == -99

    def test_create_moc(self, mocked_PipeAnalysis, mocker):
        create_moc_from_fits_mocker = mocker.patch(
            'vasttools.pipeline.PipeRun._create_moc_from_fits'
        )

        # moc_union_mocker = mocker.patch(
        #     'mocpy.MOC.union',
        # )

        result = mocked_PipeAnalysis.create_moc()

        create_moc_from_fits_mocker.assert_called_once()
        # union_calls = moc_union_mocker.call_args_list

    def test_create_moc_multiple_regions(
        self,
        mocked_PipeAnalysis,
        mocker
    ):
        create_moc_from_fits_mocker = mocker.patch(
            'vasttools.pipeline.PipeRun._create_moc_from_fits'
        )

        moc_union_mocker = create_moc_from_fits_mocker.return_value.union

        new_image_row = mocked_PipeAnalysis.images.iloc[0]
        new_image_row.name = 10

        mocked_PipeAnalysis.images = mocked_PipeAnalysis.images.append(
            new_image_row
        )

        mocked_PipeAnalysis.images.loc[10, 'skyreg_id'] = 4

        result = mocked_PipeAnalysis.create_moc()

        create_calls = create_moc_from_fits_mocker.call_args_list
        union_calls = moc_union_mocker.call_args_list

        assert len(create_calls) == 2
        moc_union_mocker.assert_called_once()

    def test_recalc_sources_df(
        self,
        mocked_PipeAnalysis,
        mocker
    ):
        pandas_read_parquet_mocker = mocker.patch(
            'vasttools.pipeline.pd.read_parquet',
            side_effect=mocked_pipeline_measurement_pairs
        )

        # define this to speed up the test to avoid dask
        dask_from_pandas_mocker = mocker.patch(
            'vasttools.pipeline.dd.from_pandas',
        )

        metrics_return_value = pd.DataFrame(data={
            'v_int': {
                729: 0.01738472025034037,
                730: 0.1481646810260763,
                2251: 0.01738472025034037
            },
            'v_peak': {
                729: 0.06053938024395404,
                730: 0.04956644262980651,
                2251: 0.06053938024395403
            },
            'eta_int': {
                729: 16.072133072157158,
                730: 15.489511624915242,
                2251: 16.072133072157158
            },
            'eta_peak': {
                729: 327.6134309054469,
                730: 5.842483557954741,
                 2251: 327.61343090548564
             }
        })

        (
            dask_from_pandas_mocker
            .return_value
            .groupby
            .return_value
            .apply
            .return_value
            .compute
            .return_value
        ) = metrics_return_value

        mocked_PipeAnalysis.load_two_epoch_metrics()

        # remove measurements from image id 2
        new_measurements = mocked_PipeAnalysis.measurements[
            mocked_PipeAnalysis.measurements.image_id != 2
        ].copy()

        result = mocked_PipeAnalysis.recalc_sources_df(new_measurements)

        assert result['n_selavy'].to_list() == [4, 4, 4]
        assert result.shape[1] == mocked_PipeAnalysis.sources.shape[1]

    def test__get_epoch_pair_plotting_df(
        self,
        mocked_PipeAnalysis,
        mocked_pipeline_pairs_df,
        mocker
    ):
        pandas_read_parquet_mocker = mocker.patch(
            'vasttools.pipeline.pd.read_parquet',
            side_effect=mocked_pipeline_measurement_pairs
        )

        mocked_PipeAnalysis.load_two_epoch_metrics()

        epoch_id = 2

        df_filter, num_pairs, num_candidates, td_days = (
            mocked_PipeAnalysis._get_epoch_pair_plotting_df(
                mocked_PipeAnalysis.measurement_pairs_df,
                epoch_id,
                'vs_peak',
                'm_peak',
                4.3,
                0.26
            )
        )

        expected_td_days = (
            mocked_pipeline_pairs_df.loc[2]['td'].total_seconds() / 86400.
        )

        assert num_pairs == 30
        assert num_candidates == 4
        assert td_days == expected_td_days

    def test__plot_epoch_pair_matplotlib(
        self,
        mocked_PipeAnalysis_wtwoepoch,
        gen_measurement_pairs_df,
        mocker
    ):
        epoch_id = 2
        expected_measurement_pairs_df = gen_measurement_pairs_df(epoch_id)

        result = mocked_PipeAnalysis_wtwoepoch._plot_epoch_pair_matplotlib(
            epoch_id,
            expected_measurement_pairs_df
        )

        expected_plot_values = np.fabs(expected_measurement_pairs_df[
            ['m_peak', 'vs_peak']
        ].to_numpy())

        expected_plot_values = expected_plot_values[
            np.argsort(expected_plot_values[:,0])
        ]

        plot_values = result.axes[0].collections[0].get_offsets()
        plot_values = plot_values[
            np.argsort(plot_values[:,0])
        ]

        assert np.all(plot_values == expected_plot_values)

    def test__plot_epoch_pair_matplotlib_styleb(
        self,
        mocked_PipeAnalysis_wtwoepoch,
        gen_measurement_pairs_df,
        mocker
    ):
        epoch_id = 2
        expected_measurement_pairs_df = gen_measurement_pairs_df(epoch_id)

        result = mocked_PipeAnalysis_wtwoepoch._plot_epoch_pair_matplotlib(
            epoch_id,
            expected_measurement_pairs_df,
            plot_style='b'
        )

        expected_plot_values = expected_measurement_pairs_df[
            ['vs_peak', 'm_peak']
        ].to_numpy()

        expected_plot_values = expected_plot_values[
            np.argsort(expected_plot_values[:,0])
        ]

        plot_values = result.axes[0].collections[2].get_offsets()
        plot_values = plot_values[
            np.argsort(plot_values[:,0])
        ]

        print(plot_values)
        print(expected_plot_values)

        assert np.all(plot_values == expected_plot_values)

    def test__plot_epoch_pair_matplotlib_int(
        self,
        mocked_PipeAnalysis_wtwoepoch,
        gen_measurement_pairs_df,
        mocker
    ):
        epoch_id = 2
        expected_measurement_pairs_df = gen_measurement_pairs_df(epoch_id)

        result = mocked_PipeAnalysis_wtwoepoch._plot_epoch_pair_matplotlib(
            epoch_id,
            expected_measurement_pairs_df,
            use_int_flux=True
        )

        expected_plot_values = np.fabs(expected_measurement_pairs_df[
            ['m_int', 'vs_int']
        ].to_numpy())

        expected_plot_values = expected_plot_values[
            np.argsort(expected_plot_values[:,0])
        ]

        plot_values = result.axes[0].collections[0].get_offsets()
        plot_values = plot_values[
            np.argsort(plot_values[:,0])
        ]

        assert np.all(plot_values == expected_plot_values)

    def test__plot_epoch_pair_bokeh(
        self,
        mocked_PipeAnalysis_wtwoepoch,
        gen_measurement_pairs_df,
        mocker
    ):
        epoch_id = 2
        expected_measurement_pairs_df = gen_measurement_pairs_df(epoch_id)

        result = mocked_PipeAnalysis_wtwoepoch._plot_epoch_pair_bokeh(
            epoch_id,
            expected_measurement_pairs_df
        )

        # expected_plot_values = np.fabs(measurement_pairs_df[
        #     ['m_peak', 'vs_peak']
        # ].to_numpy())
        #
        # expected_plot_values = expected_plot_values[
        #     np.argsort(expected_plot_values[:,0])
        # ]

        # smoke test as I'm not sure at the moment how to test with bokeh.

    def test__plot_epoch_pair_bokeh_styleb(
        self,
        mocked_PipeAnalysis_wtwoepoch,
        gen_measurement_pairs_df,
        mocker
    ):
        epoch_id = 2
        expected_measurement_pairs_df = gen_measurement_pairs_df(epoch_id)

        result = mocked_PipeAnalysis_wtwoepoch._plot_epoch_pair_bokeh(
            epoch_id,
            expected_measurement_pairs_df,
            plot_style='b'
        )

        # expected_plot_values = np.fabs(measurement_pairs_df[
        #     ['m_peak', 'vs_peak']
        # ].to_numpy())
        #
        # expected_plot_values = expected_plot_values[
        #     np.argsort(expected_plot_values[:,0])
        # ]

        # smoke test as I'm not sure at the moment how to test with bokeh.

    def test_plot_two_epoch_pairs_matplotlib(
        self,
        mocked_PipeAnalysis_wtwoepoch,
        gen_measurement_pairs_df,
        mocker
    ):
        plot_epoch_pair_mocker = mocker.patch(
            'vasttools.pipeline.PipeAnalysis._plot_epoch_pair_matplotlib',
            return_value = -99
        )

        epoch_id = 2
        expected_measurement_pairs_df = gen_measurement_pairs_df(epoch_id)

        result = mocked_PipeAnalysis_wtwoepoch.plot_two_epoch_pairs(
            epoch_id,
            plot_type='matplotlib'
        )

        plot_epoch_pair_mocker.assert_called_once()
        assert plot_epoch_pair_mocker.call_args.args[1].equals(
            expected_measurement_pairs_df
        )
        assert result == -99

    def test_plot_two_epoch_pairs_bokeh(
        self,
        mocked_PipeAnalysis_wtwoepoch,
        gen_measurement_pairs_df,
        mocker
    ):
        plot_epoch_pair_mocker = mocker.patch(
            'vasttools.pipeline.PipeAnalysis._plot_epoch_pair_bokeh',
            return_value = -99
        )

        epoch_id = 2
        expected_measurement_pairs_df = gen_measurement_pairs_df(epoch_id)

        result = mocked_PipeAnalysis_wtwoepoch.plot_two_epoch_pairs(epoch_id)

        plot_epoch_pair_mocker.assert_called_once()
        assert plot_epoch_pair_mocker.call_args.args[1].equals(
            expected_measurement_pairs_df
        )
        assert result == -99

    def test_run_two_epoch_analysis(
        self,
        mocked_PipeAnalysis_wtwoepoch,
    ):
        result_sources, result_pairs = (
            mocked_PipeAnalysis_wtwoepoch.run_two_epoch_analysis(
                4.3,
                0.26
            )
        )

        assert result_sources.shape[0] == 1
        assert result_sources.iloc[0].name == 2251

    def test_run_two_epoch_analysis_int_values(
        self,
        mocked_PipeAnalysis_wtwoepoch,
    ):
        result_sources, result_pairs = (
            mocked_PipeAnalysis_wtwoepoch.run_two_epoch_analysis(
                10.0,
                0.26,
                use_int_flux=True
            )
        )

        assert result_sources.shape[0] == 2
        assert np.all(result_sources.index == [729, 2251])

    def test__fit_eta_v(
        self,
        mocked_PipeAnalysis,
        gen_sources_metrics_df
    ):
        expected_values = [0.5, 1., 1., 2.]

        result = mocked_PipeAnalysis._fit_eta_v(gen_sources_metrics_df)

        assert result == pytest.approx(expected_values, rel=1e-1)

    def test__fit_eta_v_int_flux(
        self,
        mocked_PipeAnalysis,
        gen_sources_metrics_df
    ):
        expected_values = [1.5, 3., 2., 4.]

        result = mocked_PipeAnalysis._fit_eta_v(
            gen_sources_metrics_df,
            use_int_flux=True
        )

        assert result == pytest.approx(expected_values, rel=1e-1)

    def test__gaussian_fit(
        self,
        mocked_PipeAnalysis,
        mocker
    ):
        norm_pdf_mocker = mocker.patch(
            'vasttools.pipeline.norm.pdf',
            return_value = -99
        )

        test_data = pd.Series([0.03, 38., 84., 1090.])
        test_mean = 1.
        test_sigma = 2.

        result = mocked_PipeAnalysis._gaussian_fit(
            test_data,
            test_mean,
            test_sigma
        )

        norm_pdf_args = norm_pdf_mocker.call_args

        assert norm_pdf_args.args[0][0] == np.min(test_data)
        assert norm_pdf_args.args[0][-1] == np.max(test_data)
        assert norm_pdf_args.kwargs['loc'] == test_mean
        assert norm_pdf_args.kwargs['scale'] == test_sigma

    def test__make_bins(
        self,
        mocked_PipeAnalysis,
        gen_sources_metrics_df
    ):
        result_bins = mocked_PipeAnalysis._make_bins(
            np.log10(gen_sources_metrics_df['eta_peak'])
        )

        assert len(result_bins) == pytest.approx(18, rel=1)

    def test_eta_v_diagnostic_plot(
        self,
        mocked_PipeAnalysis,
        gen_sources_metrics_df
    ):
        """
        Smoke test
        """
        plot = mocked_PipeAnalysis.eta_v_diagnostic_plot(
            1.5,
            1.5,
            df=gen_sources_metrics_df
        )

    def test__plot_eta_v_matplotlib(
        self,
        mocked_PipeAnalysis,
        gen_sources_metrics_df,
        mocker
    ):
        """
        Smoke test.
        """
        make_bins_mocker = mocker.patch(
            'vasttools.pipeline.PipeAnalysis._make_bins',
            return_value = [0, 1, 2, 3, 4]
        )

        plot = mocked_PipeAnalysis._plot_eta_v_matplotlib(
            gen_sources_metrics_df,
            0.5,
            1.,
            1.,
            2.,
            1.5,
            1.5,
        )

    def test__plot_eta_v_bokeh(
        self,
        mocked_PipeAnalysis,
        gen_sources_metrics_df,
        mocker
    ):
        """
        Smoke test.
        """
        make_bins_mocker = mocker.patch(
            'vasttools.pipeline.PipeAnalysis._make_bins',
            return_value = [0, 1, 2, 3, 4]
        )

        plot = mocked_PipeAnalysis._plot_eta_v_bokeh(
            gen_sources_metrics_df,
            0.5,
            1.,
            1.,
            2.,
            1.5,
            1.5,
        )

    # No test for run_eta_v_analysis as it uses all the functions tested
    # above.