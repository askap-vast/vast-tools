import astropy.units as u
import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pytest
import vaex
import dask.dataframe as dd

from astropy.coordinates import SkyCoord
from mocpy import MOC
from pathlib import Path
from pytest_mock import mocker, MockerFixture  # noqa: F401
from typing import Dict, List, Union

import vasttools.pipeline as vtp


TEST_DATA_DIR = Path(__file__).resolve().parent / 'data'


@pytest.fixture
def dummy_pipeline_object(mocker: MockerFixture) -> vtp.Pipeline:
    """
    A dummy Pipeline instance for testing.

    Args:
        mocker: Pytest mock mocker object.

    Returns:
        The dummy Pipeline instance.
    """
    expected_path = '/path/to/pipelineruns/'
    mocker_getenv = mocker.patch(
        'os.getenv', return_value=expected_path
    )
    mock_isdir = mocker.patch('os.path.isdir', return_value=True)

    pipe = vtp.Pipeline()

    return pipe


def dummy_pipeline_images() -> pd.DataFrame:
    """
    A dummy pipeline images dataframe.

    Loaded from the test data directory.

    Returns:
        The dummy pipeline images dataframe.
    """
    filepath = TEST_DATA_DIR / 'test_images.csv'

    images_df = pd.read_csv(filepath)
    images_df['datetime'] = pd.to_datetime(images_df['datetime'])

    return images_df


def dummy_pipeline_bands() -> pd.DataFrame:
    """
    A dummy pipeline bands dataframe.

    Returns:
        The dummy pipeline bands dataframe.
    """
    bands_df = pd.DataFrame(
        data={
            'id': {0: 1},
            'name': {0: '887'},
            'frequency': {0: 887},
            'bandwidth': {0: 0}
        }
    )

    return bands_df


def dummy_pipeline_skyregions() -> pd.DataFrame:
    """
    A dummy pipeline sky regions dataframe.

    Returns:
        The dummy pipeline sky regions dataframe.
    """
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


def dummy_pipeline_relations() -> pd.DataFrame:
    """
    A dummy pipeline relations dataframe.

    Returns:
        The dummy pipeline relations dataframe.
    """
    relations_df = pd.DataFrame(
        data={
            'from_source_id': {6: 729, 74: 2251},
            'to_source_id': {6: 2251, 74: 729}
        }
    )

    return relations_df


def dummy_pipeline_sources() -> pd.DataFrame:
    """
    A dummy pipeline sources dataframe.

    Loaded from the test data directory.

    Returns:
        The dummy pipeline sources dataframe.
    """
    filepath = TEST_DATA_DIR / 'test_sources.csv'

    sources_df = pd.read_csv(filepath, index_col='id')

    return sources_df


def dummy_pipeline_associations() -> pd.DataFrame:
    """
    A dummy pipeline associations dataframe.

    Returns:
        The dummy pipeline associations dataframe.
    """
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


def dummy_pipeline_measurements_dask() -> dd.DataFrame:
    """
    A dummy pipeline measurements dataframe, as a dask dataframe.

    Loaded from the test data directory.

    Returns:
        The dummy pipeline measurements dask dataframe.
    """
    filepath = TEST_DATA_DIR / 'test_measurements_dask.csv'
    measurements_df = dd.read_csv(filepath).set_index('source')

    return measurements_df


def dummy_pipeline_measurements() -> pd.DataFrame:
    """
    A dummy pipeline measurements dataframe, as a pandas dataframe.

    Loaded from the test data directory.

    Returns:
        The dummy pipeline measurements pandas dataframe.
    """
    filepath = TEST_DATA_DIR / 'test_measurements.csv'
    measurements_df = pd.read_csv(filepath, index_col='Unnamed: 0')

    return measurements_df


def dummy_pipeline_measurement_pairs(*args, **kwargs) -> pd.DataFrame:
    """
    A dummy pipeline measurements pairs dataframe, as a pandas dataframe.

    Loaded from the test data directory.

    Args:
        args: Provided arguments.
        kwargs: Keyword arguments.

    Returns:
        The dummy pipeline measurements pairs pandas dataframe.
    """
    filepath = TEST_DATA_DIR / 'test_measurement_pairs.csv'
    measurement_pairs_df = pd.read_csv(filepath)

    return measurement_pairs_df


def dummy_pipeline_measurement_pairs_vaex(
    *args, **kwargs
) -> vaex.dataframe.DataFrame:
    """
    A dummy pipeline measurements pairs dataframe, as a vaex dataframe.

    Loaded from the test data directory.

    Args:
        args: Provided arguments.
        kwargs: Keyword arguments.

    Returns:
        The dummy pipeline measurements pairs vaex dataframe.
    """
    filepath = TEST_DATA_DIR / 'test_measurement_pairs.csv'
    measurements_pairs_df = pd.read_csv(filepath)
    measurements_pairs_df = vaex.from_pandas(measurements_pairs_df)

    return measurements_pairs_df


@pytest.fixture
def dummy_pipeline_pairs_df() -> pd.DataFrame:
    """
    A dummy pipeline pairs dataframe.

    Loaded from the test data directory.

    Returns:
        The dummy pipeline pairs pandas dataframe.
    """
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
    value: str,
    **kwargs
) -> pd.DataFrame:
    """
    A side effect function for the loading of the parquet files during the
    loading of a pipeline run.

    The function is called in the mocked pd.read_parquet function.

    Args:
        value: The string value being passed to pd.read_parquet in
            pipeline.py.
        kwargs: Keyword arguments.

    Returns:
        The relevant pandas dataframe which is returned from the function.
    """
    if 'bands.parquet' in value:
        return dummy_pipeline_bands()
    elif 'associations.parquet' in value:
        return dummy_pipeline_associations()
    elif 'images.parquet' in value:
        return dummy_pipeline_images()
    elif 'relations.parquet' in value:
        return dummy_pipeline_relations()
    elif 'skyregions.parquet' in value:
        return dummy_pipeline_skyregions()
    elif 'sources.parquet' in value:
        return dummy_pipeline_sources()
    else:
        raise ValueError(f'{value} file not recognised.')


@pytest.fixture
def dummy_PipeAnalysis_base(
    dummy_pipeline_object: vtp.Pipeline,
    mocker: MockerFixture
) -> vtp.PipeAnalysis:
    """
    The base dummy PipeAnalysis object used in testing.

    Because the raw pipeline outputs are processed in the load run function
    it is easier to test while creating the object each time. It is a little
    inefficient and really the pipeline process should be refactored slightly
    to support better testing.

    Args:
        dummy_pipeline_object: The dummy vtp.Pipeline fixture that is used
            to load the run.
        mocker: The pytest mock mocker object.

    Returns:
        The vtp.PipeAnalysis instance.
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
        dummy_pipeline_measurements()
    )

    pipe = dummy_pipeline_object
    run_name = 'test_run'
    run = pipe.load_run(run_name)

    return run


@pytest.fixture
def dummy_PipeAnalysis(
    dummy_PipeAnalysis_base: vtp.PipeAnalysis,
    mocker: MockerFixture
) -> vtp.PipeAnalysis:
    """
    The dummy PipeAnalysis object used for most testing.

    Because the raw pipeline outputs are processed in the load run function
    it is easier to test while creating the object each time. It is a little
    inefficient and really the pipeline process should be refactored slightly
    to support better testing.

    Args:
        dummy_PipeAnalysis_base: The base vtp.PipeAnalysis fixture
        mocker: The pytest mock mocker object.

    Returns:
        The vtp.PipeAnalysis instance.
    """

    measurement_pairs_existence_mocker = mocker.patch(
        'vasttools.pipeline.PipeRun._check_measurement_pairs_file',
        return_value=True
    )

    dummy_PipeAnalysis_base._measurement_pairs_exists = True

    return dummy_PipeAnalysis_base


@pytest.fixture
def dummy_PipeAnalysis_wtwoepoch(
    dummy_PipeAnalysis: vtp.PipeAnalysis,
    mocker: MockerFixture
) -> vtp.PipeAnalysis:
    """
    A dummy PipeAnalysis object used in testing with the two epoch data
    pre-loaded.

    Args:
        dummy_PipeAnalysis: The dummy vtp.PipeAnalysis fixture that is used
            to load the run.
        mocker: The pytest mock mocker object.

    Returns:
        The vtp.PipeAnalysis instance with two epoch data attached.
    """
    pandas_read_parquet_mocker = mocker.patch(
        'vasttools.pipeline.pd.read_parquet',
        side_effect=dummy_pipeline_measurement_pairs
    )

    dummy_PipeAnalysis.load_two_epoch_metrics()

    return dummy_PipeAnalysis


@pytest.fixture
def dummy_PipeAnalysis_dask(
    dummy_pipeline_object: vtp.Pipeline,
    mocker: MockerFixture
) -> vtp.PipeAnalysis:
    """
    A dummy PipeAnalysis object used in testing, a dask version.

    Because the raw pipeline outputs are processed in the load run function
    it is easier to test while creating the object each time. It is a little
    inefficient and really the pipeline process should be refactored slightly
    to support better testing.

    Args:
        dummy_pipeline_object: The dummy vtp.Pipeline fixture that is used
            to load the run.
        mocker: The pytest mock mocker object.

    Returns:
        The vtp.PipeAnalysis instance.
    """
    mock_isdir = mocker.patch('os.path.isdir', return_value=True)
    mock_isfile = mocker.patch('os.path.isfile', return_value=True)
    pandas_read_parquet_mocker = mocker.patch(
        'vasttools.pipeline.pd.read_parquet',
        side_effect=load_parquet_side_effect
    )
    dask_open_mocker = mocker.patch(
        'vasttools.pipeline.dd.read_parquet',
        return_value=dummy_pipeline_measurements_dask()
    )

    pipe = dummy_pipeline_object
    run_name = 'test_run'
    run = pipe.load_run(run_name)

    return run


@pytest.fixture
def dummy_PipeAnalysis_vaex_wtwoepoch(
    dummy_PipeAnalysis_vaex: vtp.PipeAnalysis,
    mocker: MockerFixture
) -> vtp.PipeAnalysis:
    """
    A dummy PipeAnalysis object used in testing with the two epoch data
    pre-loaded. Vaex version.

    Args:
        dummy_PipeAnalysis_vaex: The dummy vtp.PipeAnalysis fixture that is
            used to load the run, vaex version.
        mocker: The pytest mock mocker object.

    Returns:
        The vtp.PipeAnalysis instance with two epoch data attached.
    """
    vaex_open_mocker = mocker.patch(
        'vasttools.pipeline.vaex.open',
        side_effect=dummy_pipeline_measurement_pairs_vaex
    )

    dummy_PipeAnalysis_vaex.load_two_epoch_metrics()

    return dummy_PipeAnalysis_vaex


@pytest.fixture
def expected_sources_skycoord() -> SkyCoord:
    """
    Returns the expected SkyCoord object from the dummy sources dataframe.

    Returns:
        The SkyCoord object generated from the sources dataframe.
    """
    sources = dummy_pipeline_sources()
    sources_sc = SkyCoord(
        ra=sources['wavg_ra'],
        dec=sources['wavg_dec'],
        unit=(u.deg, u.deg)
    )

    return sources_sc


@pytest.fixture
def expected_source_measurements_pd(
    dummy_PipeAnalysis: vtp.PipeAnalysis
) -> pd.DataFrame:
    """
    A fixture function to produce the measurements dataframe for a single
    source.

    Args:
        dummy_PipeAnalysis: The dummy PipeAnalysis instance.

    Returns:
        The measurements dataframe for a single source.
    """
    def _filter_source(id: int):
        """
        A function that allows for the source id to be put as an argument.

        Args:
            id: The source id.

        Returns:
            The measurements dataframe for a requested source.
        """
        meas = dummy_PipeAnalysis.measurements
        meas = meas.loc[meas['source'] == id]

        return meas
    return _filter_source


@pytest.fixture
def filter_moc() -> MOC:
    """
    A dummy MOC to use in the filtering test.

    Returns:
        A MOC to use as a filter.
    """
    coords = SkyCoord(
        ra=[321.0, 322.0, 322.0, 321.0],
        dec=[-7., -7., -6., -6.],
        unit=(u.deg, u.deg)
    )

    moc = MOC.from_polygon_skycoord(coords, 9)

    return moc


@pytest.fixture
def gen_measurement_pairs_df(
    dummy_PipeAnalysis_wtwoepoch: vtp.PipeAnalysis
) -> pd.DataFrame:
    """
    Generates a measurement pairs dataframe for a specific 'pair epoch'.

    Args:
        dummy_PipeAnalysis_wtwoepoch: The dummy PipeAnalysis object with the
            two epoch metrics loaded.

    Returns:
        The measurement pairs df filtered for a pair epoch.
    """
    def _gen_df(epoch_id: int = 2) -> pd.DataFrame:
        """
        Filters a measurement pairs dataframe for a specific 'pair epoch'.

        Args:
            epoch_id: The id of the measurement pair epoch.

        Returns:
            The measurement pairs df filtered for a pair epoch.
        """
        epoch_key = (
            dummy_PipeAnalysis_wtwoepoch
            .pairs_df.loc[epoch_id]['pair_epoch_key']
        )

        measurement_pairs_df = (
            dummy_PipeAnalysis_wtwoepoch.measurement_pairs_df.loc[
                dummy_PipeAnalysis_wtwoepoch.measurement_pairs_df[
                    'pair_epoch_key'
                ] == epoch_key
            ]
        ).copy()

        return measurement_pairs_df
    return _gen_df


@pytest.fixture
def gen_sources_metrics_df() -> pd.DataFrame:
    """
    Generates a dataframe containing random source metrics used for testing.

    Returns:
        Source metrics dataframe.
    """
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
    """
    Class that contains all the tests for the Pipeline object from
    vasttools.Pipeline.
    """

    def test_init(self, mocker: MockerFixture) -> None:
        """
        Tests the initialisation of a Pipeline instance.

        The directory checked is mocked to be true.

        Args:
            mocker: The pytest mocker mock object.

        Returns:
            None
        """
        expected_path = '/path/to/pipelineruns/'
        mocker_getenv = mocker.patch(
            'os.getenv', return_value=expected_path
        )
        mock_isdir = mocker.patch('os.path.isdir', return_value=True)

        pipe = vtp.Pipeline()

        assert pipe.project_dir == expected_path

    def test_init_projectdir(self, mocker: MockerFixture) -> None:
        """
        Tests the initialisation of a Pipeline instance, when the project
        dir is stated.

        The directory checked is mocked to be true.

        Args:
            mocker: The pytest mocker mock object.

        Returns:
            None
        """
        expected_path = '/path/to/projectdir/'
        mocker_abspath = mocker.patch(
            'os.path.abspath', return_value=expected_path
        )
        mock_isdir = mocker.patch('os.path.isdir', return_value=True)

        pipe = vtp.Pipeline(project_dir=expected_path)

        assert pipe.project_dir == expected_path

    def test_init_env_fail(self, mocker: MockerFixture) -> None:
        """
        Tests the initialisation failure of a Pipeline instance.

        Specifically no project dir has been stated or found in the env.

        Args:
            mocker: The pytest mocker mock object.

        Returns:
            None
        """
        expected_path = '/path/to/pipelineruns/'
        mocker_getenv = mocker.patch(
            'os.getenv', return_value=None
        )

        with pytest.raises(vtp.PipelineDirectoryError) as excinfo:
            pipe = vtp.Pipeline()

        assert str(excinfo.value).startswith(
            "The pipeline run directory could not be determined!"
        )

    def test_init_project_dir_fail(self, mocker: MockerFixture) -> None:
        """
        Tests the initialisation failure of a Pipeline instance.

        The directory checked is mocked to be false.

        Args:
            mocker: The pytest mocker mock object.

        Returns:
            None
        """
        expected_path = '/path/to/projectdir/'
        mocker_abspath = mocker.patch(
            'os.path.abspath', return_value=expected_path
        )
        mock_isdir = mocker.patch('os.path.isdir', return_value=False)

        with pytest.raises(vtp.PipelineDirectoryError) as excinfo:
            pipe = vtp.Pipeline(project_dir=expected_path)

        assert str(excinfo.value).startswith("Pipeline run directory")

    def test_list_piperuns(
        self,
        dummy_pipeline_object: vtp.Pipeline,
        mocker: MockerFixture
    ) -> None:
        """
        Tests the list piperuns method.

        Args:
            dummy_pipeline_object: The dummy Pipeline object that is used
                for testing.
            mocker: The pytest mocker mock object.

        Returns:
            None
        """
        expected_path = '/path/to/pipelineruns'

        expected_result = ['job1', 'job2']

        mocker_glob = mocker.patch(
            'glob.glob', return_value=[
                os.path.join(expected_path, 'job1'),
                os.path.join(expected_path, 'job2'),
                os.path.join(expected_path, 'images')
            ]
        )

        result = dummy_pipeline_object.list_piperuns()

        mocker_glob.assert_called_once_with(os.path.join(expected_path, '*'))
        assert result == expected_result

    def test_list_images(
        self,
        dummy_pipeline_object: vtp.Pipeline,
        mocker: MockerFixture
    ) -> None:
        """
        Tests the list images method.

        Assumes pipeline run directory has 'images'. Mocks the glob call.

        Args:
            dummy_pipeline_object: The dummy Pipeline object that is used
                for testing.
            mocker: The pytest mocker mock object.

        Returns:
            None
        """
        expected_path = '/path/to/pipelineruns'

        expected_result = ['image1', 'image2']

        mocker_glob = mocker.patch(
            'glob.glob', return_value=[
                os.path.join(expected_path, 'images', 'image1'),
                os.path.join(expected_path, 'images', 'image2'),
            ]
        )

        result = dummy_pipeline_object.list_images()

        mocker_glob.assert_called_once_with(os.path.join(
            expected_path, 'images', '*'
        ))
        assert result == expected_result

    def test_load_run_dir_fail(
        self,
        dummy_pipeline_object: vtp.Pipeline,
        mocker: MockerFixture
    ) -> None:
        """
        Tests the failure of the load run method.

        Specifically when the run directory is not found. Is dir check is
        mocked with a False return.

        Args:
            dummy_pipeline_object: The dummy Pipeline object that is used
                for testing.
            mocker: The pytest mocker mock object.

        Returns:
            None
        """
        mock_isdir = mocker.patch('os.path.isdir', return_value=False)

        pipe = dummy_pipeline_object

        with pytest.raises(OSError) as excinfo:
            pipe.load_run('test')

    def test_load_run_no_dask(
        self,
        dummy_pipeline_object: vtp.Pipeline,
        mocker: MockerFixture
    ) -> None:
        """
        Tests the load run method.

        Specifically when the arrow files are not present so dask is not used.
        The usual mocks are in place, including using the read parquet side
        effect.

        Args:
            dummy_pipeline_object: The dummy Pipeline object that is used
                for testing.
            mocker: The pytest mocker mock object.

        Returns:
            None
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
            dummy_pipeline_measurements()
        )

        pipe = dummy_pipeline_object
        run_name = 'test_run'
        run = pipe.load_run(run_name)

        assert run.name == run_name
        assert run._dask_meas is False

    def test_load_run_no_dask_check_columns(
        self,
        dummy_pipeline_object: vtp.Pipeline,
        mocker: MockerFixture
    ) -> None:
        """
        Tests the load run method.

        Specifically checks that the dataframes have been constructed
        correctly in the non dask case.

        Args:
            dummy_pipeline_object: The dummy Pipeline object that is used
                for testing.
            mocker: The pytest mocker mock object.

        Returns:
            None
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
            dummy_pipeline_measurements()
        )

        pipe = dummy_pipeline_object
        run_name = 'test_run'
        run = pipe.load_run(run_name)

        assert 'centre_ra' in run.images.columns
        assert run.images.shape[1] == 29
        assert run.measurements.shape[1] == 42

    def test_load_run_dask(
        self,
        dummy_pipeline_object: vtp.Pipeline,
        mocker: MockerFixture
    ) -> None:
        """
        Tests the load run method.

        Specifically when the arrow files are present so dask is used.
        The usual mocks are in place, including using the read parquet side
        effect.

        Args:
            dummy_pipeline_object: The dummy Pipeline object that is used
                for testing.
            mocker: The pytest mocker mock object.

        Returns:
            None
        """
        mock_isdir = mocker.patch('os.path.isdir', return_value=True)
        mock_isfile = mocker.patch('os.path.isfile', return_value=True)
        pandas_read_parquet_mocker = mocker.patch(
            'vasttools.pipeline.pd.read_parquet',
            side_effect=load_parquet_side_effect
        )
        dask_open_mocker = mocker.patch(
            'vasttools.pipeline.dd.read_parquet',
            return_value=dummy_pipeline_measurements_dask()
        )

        pipe = dummy_pipeline_object
        run_name = 'test_run'
        run = pipe.load_run(run_name)

        assert run.name == run_name
        assert run._dask_meas is True


class TestPipeAnalysis:
    """
    This class contains tests for both the PipeRun and PipeAnalysis objects in
    the pipeline component.
    """

    @pytest.mark.parametrize(
        "pairs_existence",
        [
            [True],
            [False],
            [True,True],
            [False,False],
            [True,False],
        ],
        ids=("single-exists",
             "single-no-exists",
             "multiple-all-exists",
             "multiple-no-exists",
             "multiple-some-exists",
             )
     )
    def test__check_measurement_pairs_file(self,
        pairs_existence: List[bool],
        dummy_PipeAnalysis_base: vtp.PipeAnalysis,
        mocker: MockerFixture
        ) -> None:
        """
        Tests the _check_measurement_pairs_file method.
        
        Args:
            pairs_existence: A list of booleans corresponding to whether a
                pairs file exists.
            dummy_PipeAnalysis_base: The base dummy PipeAnalysis object.
            mocker: The pytest-mock mocker object.
        
        Returns:
            None
        """
        mocker_isfile = mocker.patch(
            "os.path.isfile",
            side_effect=pairs_existence
        )
        
        fake_pairs_file = [""]*len(pairs_existence)
        
        dummy_PipeAnalysis_base.measurement_pairs_file = fake_pairs_file
        
        returned_val = dummy_PipeAnalysis_base._check_measurement_pairs_file()
        
        all_exist = sum(pairs_existence) == len(pairs_existence)
        
        assert returned_val == all_exist
        
    
    def test_combine_with_run(
        self,
        dummy_PipeAnalysis: vtp.PipeAnalysis
    ) -> None:
        """
        Tests the combine with other run method.

        To mimic a second run the PipeAnalysis object is copied and the source
        id's are changed to be different. Both are pandas based.

        Args:
            dummy_PipeAnalysis: The dummy PipeAnalysis object that is used
                for testing.

        Returns:
            None
        """
        run_2 = copy.deepcopy(dummy_PipeAnalysis)
        run_2.sources.index = [100, 200, 300]
        new_run = dummy_PipeAnalysis.combine_with_run(run_2)

        assert new_run.sources.shape[0] == 6

    def test_combine_with_run_pandas_dask(
        self,
        dummy_PipeAnalysis: vtp.PipeAnalysis,
        dummy_PipeAnalysis_dask: vtp.PipeAnalysis
    ) -> None:
        """
        Tests the combine with other run method.

        The original run is pandas based and the second run is dask based.

        Args:
            dummy_PipeAnalysis: The dummy PipeAnalysis object that is used
                for testing.
            dummy_PipeAnalysis_dask: The dummy PipeAnalysis object that is
                used for testing that has dask loaded measurements.

        Returns:
            None
        """
        dummy_PipeAnalysis.sources.index = [100, 200, 300]
        new_run = dummy_PipeAnalysis.combine_with_run(
            dummy_PipeAnalysis_dask
        )

        assert new_run.sources.shape[0] == 6

    def test_combine_with_run_both_dask(
        self,
        dummy_PipeAnalysis_dask: vtp.PipeAnalysis
    ) -> None:
        """
        Tests the combine with other run method.

        To mimic a second run the PipeAnalysis object is copied and the source
        id's are changed to be different. Both are pandas based. Both runs
        are dask based.

        Args:
            dummy_PipeAnalysis_dask: The dummy PipeAnalysis object that is
                used for testing that has dask loaded measurements.

        Returns:
            None
        """
        run_2 = copy.deepcopy(dummy_PipeAnalysis_dask)
        run_2.sources.index = [100, 200, 300]
        new_run = dummy_PipeAnalysis_dask.combine_with_run(run_2)

        assert new_run.sources.shape[0] == 6

    def test_pipeanalysis_get_sources_skycoord(
        self,
        dummy_PipeAnalysis: vtp.PipeAnalysis,
        mocker: MockerFixture
    ) -> None:
        """
        Tests the get sources sky coord method.

        Args:
            dummy_PipeAnalysis: The dummy PipeAnalysis object that is used
                for testing.
            mocker: The pytest mocker mock object.

        Returns:
            None
        """
        pipe = dummy_PipeAnalysis

        expected = dummy_pipeline_sources()
        expected = SkyCoord(
            ra=expected['wavg_ra'],
            dec=expected['wavg_dec'],
            unit=(u.deg, u.deg)
        )

        result = pipe.get_sources_skycoord()

        assert np.all(result == expected)

    def test_pipeanalysis_get_sources_skycoord_user_sources(
        self,
        dummy_PipeAnalysis: vtp.PipeAnalysis,
        mocker: MockerFixture
    ) -> None:
        """
        Tests the get sources sky coord method for a custom user defined
        dataframe.

        Args:
            dummy_PipeAnalysis: The dummy PipeAnalysis object that is used
                for testing.
            mocker: The pytest mocker mock object.

        Returns:
            None
        """
        pipe = dummy_PipeAnalysis

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
            user_sources=test_sources_df,
            ra_col='ra',
            dec_col='dec'
        )

        assert np.all(result == expected)

    def test_pipeanalysis_get_sources_skycoord_user_sources_hms(
        self,
        dummy_PipeAnalysis: vtp.PipeAnalysis,
        mocker: MockerFixture
    ) -> None:
        """
        Tests the get sources sky coord method for a custom dataframe where
        the coordinates are in hms format.

        Args:
            dummy_PipeAnalysis: The dummy PipeAnalysis object that is used
                for testing.
            mocker: The pytest mocker mock object.

        Returns:
            None
        """
        pipe = dummy_PipeAnalysis

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
            user_sources=test_sources_df,
            ra_unit=u.hourangle
        )

        assert np.all(result == expected)

    def test_pipeanalysis_get_source(
        self,
        dummy_PipeAnalysis: vtp.PipeAnalysis,
        expected_sources_skycoord: SkyCoord,
        expected_source_measurements_pd: pd.DataFrame,
        mocker: MockerFixture
    ) -> None:
        """
        Tests the get source method that would normally return a
        vasttools.source.Source instance.

        The Source instance call is mocked with the call args checked.

        Args:
            dummy_PipeAnalysis: The dummy PipeAnalysis object that is used
                for testing.
            expected_sources_skycoord: The expected SkyCoord object generated
                from the sources.
            expected_source_measurements_pd: The expected source measurements
                dataframe.
            mocker: The pytest mocker mock object.

        Returns:
            None
        """
        mocker_source = mocker.patch(
            'vasttools.pipeline.Source',
            return_value=-99
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

        the_source = dummy_PipeAnalysis.get_source(729)

        mocker_source.assert_called_once()

        mocker_calls = mocker_source.call_args
        assert mocker_calls.kwargs == expected_call.kwargs
        assert mocker_calls.args[7].shape[1] == 51
        assert mocker_calls.args[0] == expected_sources_skycoord[0]
        assert the_source == -99

    def test_pipeanalysis_get_source_dask(
        self,
        dummy_PipeAnalysis_dask: vtp.PipeAnalysis,
        expected_sources_skycoord: SkyCoord,
        expected_source_measurements_pd: pd.DataFrame,
        mocker: MockerFixture
    ) -> None:
        """
        Tests the get source method that would normally return a
        vasttools.source.Source instance.

        The Source instance call is mocked with the call args checked. This
        test performs the check on a dask loaded run.

        Args:
            dummy_PipeAnalysis: The dummy PipeAnalysis object that is used
                for testing.
            expected_sources_skycoord: The expected SkyCoord object generated
                from the sources.
            expected_source_measurements_pd: The expected source measurements
                dataframe.
            mocker: The pytest mocker mock object.

        Returns:
            None
        """
        mocker_source = mocker.patch(
            'vasttools.pipeline.Source',
            return_value=-99
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

        the_source = dummy_PipeAnalysis_dask.get_source(729)

        mocker_source.assert_called_once()

        mocker_calls = mocker_source.call_args
        assert mocker_calls.kwargs == expected_call.kwargs
        assert mocker_calls.args[7].shape[1] == 51
        assert mocker_calls.args[0] == expected_sources_skycoord[0]
        assert the_source == -99

    def test_pipeanalysis_load_two_epoch_metrics_pandas(
        self,
        dummy_PipeAnalysis: vtp.PipeAnalysis,
        dummy_pipeline_pairs_df: pd.DataFrame,
        mocker: MockerFixture
    ) -> None:
        """
        Tests the method that loads the two epoch metrics.

        This test is for pandas loaded dataframes.

        Args:
            dummy_PipeAnalysis: The dummy PipeAnalysis object that is used
                for testing.
            dummy_pipeline_pairs_df: The dummy pairs dataframe.
            mocker: The pytest mocker mock object.

        Returns:
            None
        """
        pandas_read_parquet_mocker = mocker.patch(
            'vasttools.pipeline.pd.read_parquet',
            side_effect=dummy_pipeline_measurement_pairs
        )

        dummy_PipeAnalysis.load_two_epoch_metrics()

        assert dummy_PipeAnalysis.pairs_df.equals(dummy_pipeline_pairs_df)
        assert dummy_PipeAnalysis.measurement_pairs_df.shape[0] == 30

    def test_pipeanalysis_load_two_epoch_metrics_vaex(
        self,
        dummy_PipeAnalysis_vaex: vtp.PipeAnalysis,
        dummy_pipeline_pairs_df: pd.DataFrame,
        mocker: MockerFixture
    ) -> None:
        """
        Tests the method that loads the two epoch metrics.

        This test is for vaex loaded dataframes.

        Args:
            dummy_PipeAnalysis: The dummy PipeAnalysis object that is used
                for testing.
            dummy_pipeline_pairs_df: The dummy pairs dataframe.
            mocker: The pytest mocker mock object.

        Returns:
            None
        """
        vaex_open_mocker = mocker.patch(
            'vasttools.pipeline.vaex.open',
            side_effect=dummy_pipeline_measurement_pairs_vaex
        )

        dummy_PipeAnalysis_vaex.load_two_epoch_metrics()

        assert dummy_PipeAnalysis_vaex.pairs_df.equals(
            dummy_pipeline_pairs_df
        )
        assert dummy_PipeAnalysis_vaex.measurement_pairs_df.shape[0] == 30

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
        row: pd.Series,
        kwargs: Dict[str, bool],
        expected: List[pd.Timestamp],
        dummy_PipeAnalysis: vtp.PipeAnalysis
    ) -> None:
        """
        Tests the method that adds times to be searched for planet matches.

        The test is parametrised to run for every hour is False and True.

        Args:
            row: The pandas series containing the DATE-OBS and duration.
            kwargs: Contains the 'every_hour' argument to pass.
            expected: The expected pandas series result.
            dummy_PipeAnalysis: The dummy PipeAnalysis object that is used
                for testing.

        Returns:
            None
        """
        result = dummy_PipeAnalysis._add_times(row, **kwargs)

        assert result == expected

    def test_check_for_planets(
        self,
        dummy_PipeAnalysis: vtp.PipeAnalysis,
        mocker: MockerFixture
    ) -> None:
        """
        Tests the method checks the pipeline run for planets.

        The dask call is mocked as the actual method that finds the planets
        is tested in the utils tests.

        Args:
            dummy_PipeAnalysis: The dummy PipeAnalysis object that is used
                for testing.
            mocker: The pytest mock mocker object.

        Returns:
            None
        """
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
        ) = pd.DataFrame(data={'planet_test': [True]})

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

        result = dummy_PipeAnalysis.check_for_planets()
        call = dask_from_pandas_mocker.call_args.args[0]

        dask_from_pandas_mocker.assert_called_once()
        assert call.isnull().values.any() == False
        assert call.planet.value_counts().equals(expected_planet_counts)

    def test_filter_by_moc(
        self,
        dummy_PipeAnalysis: vtp.PipeAnalysis,
        filter_moc: MOC,
    ) -> None:
        """
        Tests the filter by moc function.

        The filter moc has been designed to leave two sources
        (10 measurements).

        Args:
            dummy_PipeAnalysis: The dummy PipeAnalysis object that is used
                for testing.
            filter_moc: The MOC used for filtering.

        Returns:
            None
        """
        result = dummy_PipeAnalysis.filter_by_moc(filter_moc)

        assert result.sources.shape[0] == 2
        assert result.measurements.shape[0] == 10

    def test_create_moc(
        self,
        dummy_PipeAnalysis: vtp.PipeAnalysis,
        mocker: MockerFixture
    ) -> None:
        """
        Tests the front facing create moc method.

        Asserts that the above tested function is called once.

        Args:
            dummy_PipeAnalysis: The dummy PipeAnalysis object that is used
                for testing.
            mocker: The pytest mock mocker object.

        Returns:
            None
        """
        create_moc_from_fits_mocker = mocker.patch(
            'vasttools.pipeline.create_moc_from_fits'
        )

        result = dummy_PipeAnalysis.create_moc()
        create_moc_from_fits_mocker.assert_called_once()

    def test_create_moc_multiple_regions(
        self,
        dummy_PipeAnalysis: vtp.PipeAnalysis,
        mocker: MockerFixture
    ) -> None:
        """
        Tests the create moc method for multiple regions present.

        Asserts that the union MOC method is called.

        Args:
            dummy_PipeAnalysis: The dummy PipeAnalysis object that is used
                for testing.
            mocker: The pytest mock mocker object.

        Returns:
            None
        """
        create_moc_from_fits_mocker = mocker.patch(
            'vasttools.pipeline.create_moc_from_fits'
        )

        moc_union_mocker = create_moc_from_fits_mocker.return_value.union

        new_image_row = dummy_PipeAnalysis.images.iloc[0]
        new_image_row.name = 10

        dummy_PipeAnalysis.images = pd.concat(
            [
                dummy_PipeAnalysis.images,
                # need to transpose the series to a dataframe to concat
                new_image_row.to_frame().T.set_index('name')
            ]
        )

        dummy_PipeAnalysis.images.loc[10, 'skyreg_id'] = 4

        result = dummy_PipeAnalysis.create_moc()

        create_calls = create_moc_from_fits_mocker.call_args_list
        union_calls = moc_union_mocker.call_args_list

        assert len(create_calls) == 2
        moc_union_mocker.assert_called_once()

    @pytest.mark.parametrize(
        'fixture_name',
        ['dummy_PipeAnalysis_wtwoepoch', 'dummy_PipeAnalysis_vaex_wtwoepoch']
    )
    def test__filter_meas_pairs_df(self, fixture_name: str, request) -> None:
        """
        Tests the utility method that filters the measurement pairs dataframe.

        Args:
            fixture_name: The name of the fixture to load.
            request: The pytest request fixture, used to load the
                relevant fixture.

        Returns:
            None
        """
        the_fixture = request.getfixturevalue(fixture_name)
        # remove measurements from image id 2
        mask = the_fixture.measurements['image_id'] != 2

        new_measurements = the_fixture.measurements[
            mask
        ].copy()

        # get IDs of those removed
        meas_ids = (
            the_fixture.measurements[~mask]['id'].to_numpy()
        )

        result = the_fixture._filter_meas_pairs_df(
            new_measurements
        )

        assert result.shape[0] == 18
        assert np.any(result['meas_id_a'].isin(meas_ids).to_numpy()) == False
        assert np.any(result['meas_id_b'].isin(meas_ids).to_numpy()) == False

    @pytest.mark.parametrize(
        'fixture_name',
        ['dummy_PipeAnalysis_wtwoepoch', 'dummy_PipeAnalysis_vaex_wtwoepoch']
    )
    def test_recalc_measurement_pairs_df(
        self,
        fixture_name: str,
        request
    ) -> None:
        """
        Tests the method that recalculates the measurement pairs dataframe.

        Args:
            fixture_name: The name of the fixture to load.
            request: The pytest request fixture, used to load the
                relevant fixture.

        Returns:
            None
        """
        the_fixture = request.getfixturevalue(fixture_name)

        # mulitply fluxes by 100 as new meas
        new_meas = the_fixture.measurements.copy()
        new_meas['flux_int'] = new_meas['flux_int'] * 100.
        new_meas['flux_peak'] = new_meas['flux_peak'] * 100.

        result = the_fixture.recalc_measurement_pairs_df(new_meas)

        expected_vs_peak = the_fixture.measurement_pairs_df['vs_peak'] * 100.
        expected_vs_int = the_fixture.measurement_pairs_df['vs_int'] * 100.

        expected_m_peak = the_fixture.measurement_pairs_df['m_peak']
        expected_m_int = the_fixture.measurement_pairs_df['m_int']

        assert result['vs_peak'].to_numpy() == pytest.approx(
            expected_vs_peak.to_numpy()
        )
        assert result['vs_int'].to_numpy() == pytest.approx(
            expected_vs_int.to_numpy()
        )

        assert result['m_peak'].to_numpy() == pytest.approx(
            expected_m_peak.to_numpy()
        )
        assert result['m_int'].to_numpy() == pytest.approx(
            expected_m_int.to_numpy()
        )

    def test_recalc_sources_df(
        self,
        dummy_PipeAnalysis: vtp.PipeAnalysis,
        mocker: MockerFixture
    ) -> None:
        """
        Tests the method that recalculates the source statistics.

        Args:
            dummy_PipeAnalysis: The dummy PipeAnalysis object that is used
                for testing.
            mocker: The pytest mock mocker object.

        Returns:
            None
        """
        
        pandas_read_parquet_mocker = mocker.patch(
            'vasttools.pipeline.pd.read_parquet',
            side_effect=dummy_pipeline_measurement_pairs
        )

        # define this to speed up the test to avoid dask
        """dask_from_pandas_mocker = mocker.patch(
            'vasttools.pipeline.dd.from_pandas'
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
        ) = metrics_return_value"""

        dummy_PipeAnalysis.load_two_epoch_metrics()

        # remove measurements from image id 2
        new_measurements = dummy_PipeAnalysis.measurements[
            dummy_PipeAnalysis.measurements.image_id != 2
        ].copy()

        result = dummy_PipeAnalysis.recalc_sources_df(new_measurements)
        
        print(result)
        print(dummy_PipeAnalysis.sources)
        
        
        assert result['n_selavy'].to_list() == [4, 4, 4]
        assert result.shape[1] == dummy_PipeAnalysis.sources.shape[1]
        pd.testing.assert_frame_equal(result, dummy_PipeAnalysis.sources)
        assert 1==0

    def test__get_epoch_pair_plotting_df(
        self,
        dummy_PipeAnalysis_wtwoepoch: vtp.PipeAnalysis,
        dummy_pipeline_pairs_df: pd.DataFrame,
        mocker: MockerFixture
    ) -> None:
        """
        Tests the method that generates a dataframe and other metrics used
        in plotting two epoch metric values.

        Args:
            dummy_PipeAnalysis_wtwoepoch: The dummy PipeAnalysis object that
                is used for testing that includes the two epoch metrics.
            dummy_pipeline_pairs_df: The dummy pipelien pairs dataframe.
            mocker: The pytest mock mocker object.

        Returns:
            None
        """
        epoch_id = 2

        df_filter, num_pairs, num_candidates, td_days = (
            dummy_PipeAnalysis_wtwoepoch._get_epoch_pair_plotting_df(
                dummy_PipeAnalysis_wtwoepoch.measurement_pairs_df,
                epoch_id,
                'vs_peak',
                'm_peak',
                4.3,
                0.26
            )
        )

        expected_td_days = (
            dummy_pipeline_pairs_df.loc[2]['td'].total_seconds() / 86400.
        )

        assert num_pairs == 30
        assert num_candidates == 4
        assert td_days == expected_td_days

    def test__plot_epoch_pair_matplotlib(
        self,
        dummy_PipeAnalysis_wtwoepoch: vtp.PipeAnalysis,
        gen_measurement_pairs_df: pd.DataFrame,
        mocker: MockerFixture
    ) -> None:
        """
        Tests the matplotlib method of plotting the epoch pair metrics.

        Args:
            dummy_PipeAnalysis_wtwoepoch: The dummy PipeAnalysis object that
                is used for testing that includes the two epoch metrics.
            gen_measurement_pairs_df: Measurement pairs df for a specific
                epoch.
            mocker: The pytest mock mocker object.

        Returns:
            None
        """
        epoch_id = 2
        expected_measurement_pairs_df = gen_measurement_pairs_df(epoch_id)

        result = dummy_PipeAnalysis_wtwoepoch._plot_epoch_pair_matplotlib(
            epoch_id,
            expected_measurement_pairs_df
        )

        expected_plot_values = np.fabs(expected_measurement_pairs_df[
            ['m_peak', 'vs_peak']
        ].to_numpy())

        expected_plot_values = expected_plot_values[
            np.argsort(expected_plot_values[:, 0])
        ]

        plot_values = result.axes[0].collections[0].get_offsets()
        plot_values = plot_values[
            np.argsort(plot_values[:, 0])
        ]

        assert np.all(plot_values == expected_plot_values)

        plt.close(result)

    def test__plot_epoch_pair_matplotlib_styleb(
        self,
        dummy_PipeAnalysis_wtwoepoch: vtp.PipeAnalysis,
        gen_measurement_pairs_df: pd.DataFrame,
        mocker: MockerFixture
    ) -> None:
        """
        Tests the matplotlib style b method of plotting the epoch pair
        metrics.

        Args:
            dummy_PipeAnalysis_wtwoepoch: The dummy PipeAnalysis object that
                is used for testing that includes the two epoch metrics.
            gen_measurement_pairs_df: Measurement pairs df for a specific
                epoch.
            mocker: The pytest mock mocker object.

        Returns:
            None
        """
        epoch_id = 2
        expected_measurement_pairs_df = gen_measurement_pairs_df(epoch_id)

        result = dummy_PipeAnalysis_wtwoepoch._plot_epoch_pair_matplotlib(
            epoch_id,
            expected_measurement_pairs_df,
            plot_style='b'
        )

        expected_plot_values = expected_measurement_pairs_df[
            ['vs_peak', 'm_peak']
        ].to_numpy()

        expected_plot_values = expected_plot_values[
            np.argsort(expected_plot_values[:, 0])
        ]

        plot_values = result.axes[0].collections[2].get_offsets()
        plot_values = plot_values[
            np.argsort(plot_values[:, 0])
        ]

        assert np.all(plot_values == expected_plot_values)

        plt.close(result)

    def test__plot_epoch_pair_matplotlib_int_flux(
        self,
        dummy_PipeAnalysis_wtwoepoch: vtp.PipeAnalysis,
        gen_measurement_pairs_df: pd.DataFrame,
        mocker: MockerFixture
    ) -> None:
        """
        Tests the matplotlib method of plotting the epoch pair metrics using
        the integrated fluxes.

        Args:
            dummy_PipeAnalysis_wtwoepoch: The dummy PipeAnalysis object that
                is used for testing that includes the two epoch metrics.
            gen_measurement_pairs_df: Measurement pairs df for a specific
                epoch.
            mocker: The pytest mock mocker object.

        Returns:
            None
        """
        epoch_id = 2
        expected_measurement_pairs_df = gen_measurement_pairs_df(epoch_id)

        result = dummy_PipeAnalysis_wtwoepoch._plot_epoch_pair_matplotlib(
            epoch_id,
            expected_measurement_pairs_df,
            use_int_flux=True
        )

        expected_plot_values = np.fabs(expected_measurement_pairs_df[
            ['m_int', 'vs_int']
        ].to_numpy())

        expected_plot_values = expected_plot_values[
            np.argsort(expected_plot_values[:, 0])
        ]

        plot_values = result.axes[0].collections[0].get_offsets()
        plot_values = plot_values[
            np.argsort(plot_values[:, 0])
        ]

        assert np.all(plot_values == expected_plot_values)

        plt.close(result)

    def test__plot_epoch_pair_bokeh(
        self,
        dummy_PipeAnalysis_wtwoepoch: vtp.PipeAnalysis,
        gen_measurement_pairs_df: pd.DataFrame,
        mocker: MockerFixture
    ) -> None:
        """
        Smoke tests the bokeh method of plotting the epoch pair metrics.

        Args:
            dummy_PipeAnalysis_wtwoepoch: The dummy PipeAnalysis object that
                is used for testing that includes the two epoch metrics.
            gen_measurement_pairs_df: Measurement pairs df for a specific
                epoch.
            mocker: The pytest mock mocker object.

        Returns:
            None
        """
        epoch_id = 2
        expected_measurement_pairs_df = gen_measurement_pairs_df(epoch_id)

        result = dummy_PipeAnalysis_wtwoepoch._plot_epoch_pair_bokeh(
            epoch_id,
            expected_measurement_pairs_df
        )

        # smoke test as I'm not sure at the moment how to test with bokeh.
        assert True

    def test__plot_epoch_pair_bokeh_styleb(
        self,
        dummy_PipeAnalysis_wtwoepoch: vtp.PipeAnalysis,
        gen_measurement_pairs_df: pd.DataFrame,
        mocker: MockerFixture
    ) -> None:
        """
        Smoke tests the bokeh style b method of plotting the epoch pair
        metrics.

        Args:
            dummy_PipeAnalysis_wtwoepoch: The dummy PipeAnalysis object that
                is used for testing that includes the two epoch metrics.
            gen_measurement_pairs_df: Measurement pairs df for a specific
                epoch.
            mocker: The pytest mock mocker object.

        Returns:
            None
        """
        epoch_id = 2
        expected_measurement_pairs_df = gen_measurement_pairs_df(epoch_id)

        result = dummy_PipeAnalysis_wtwoepoch._plot_epoch_pair_bokeh(
            epoch_id,
            expected_measurement_pairs_df,
            plot_style='b'
        )

        # smoke test as I'm not sure at the moment how to test with bokeh.
        assert True

    def test_plot_two_epoch_pairs_matplotlib(
        self,
        dummy_PipeAnalysis_wtwoepoch: vtp.PipeAnalysis,
        gen_measurement_pairs_df: pd.DataFrame,
        mocker: MockerFixture
    ) -> None:
        """
        Tests the front facing method to plot the two epoch metrics
        (matplotlib). Asserts the right functions are called and the return
        results passed through.

        Args:
            dummy_PipeAnalysis_wtwoepoch: The dummy PipeAnalysis object that
                is used for testing that includes the two epoch metrics.
            gen_measurement_pairs_df: Measurement pairs df for a specific
                epoch.
            mocker: The pytest mock mocker object.

        Returns:
            None
        """
        plot_epoch_pair_mocker = mocker.patch(
            'vasttools.pipeline.PipeAnalysis._plot_epoch_pair_matplotlib',
            return_value=-99
        )

        epoch_id = 2
        expected_measurement_pairs_df = gen_measurement_pairs_df(epoch_id)

        result = dummy_PipeAnalysis_wtwoepoch.plot_two_epoch_pairs(
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
        dummy_PipeAnalysis_wtwoepoch: vtp.Pipeline,
        gen_measurement_pairs_df: pd.DataFrame,
        mocker: MockerFixture
    ) -> None:
        """
        Tests the front facing method to plot the two epoch metrics
        (matplotlib). Asserts the right functions are called and the return
        results passed through.

        Args:
            dummy_PipeAnalysis_wtwoepoch: The dummy PipeAnalysis object that
                is used for testing that includes the two epoch metrics.
            gen_measurement_pairs_df: Measurement pairs df for a specific
                epoch.
            mocker: The pytest mock mocker object.

        Returns:
            None
        """
        plot_epoch_pair_mocker = mocker.patch(
            'vasttools.pipeline.PipeAnalysis._plot_epoch_pair_bokeh',
            return_value=-99
        )

        epoch_id = 2
        expected_measurement_pairs_df = gen_measurement_pairs_df(epoch_id)

        result = dummy_PipeAnalysis_wtwoepoch.plot_two_epoch_pairs(epoch_id)

        plot_epoch_pair_mocker.assert_called_once()
        assert plot_epoch_pair_mocker.call_args.args[1].equals(
            expected_measurement_pairs_df
        )
        assert result == -99

    @pytest.mark.parametrize(
        "vs_thresh,use_int_flux,expected_shape,expected_ids",
        [
            (4.3, False, 1, [2251]),
            (10.0, True, 2, [729, 2251])
        ]
    )
    def test_run_two_epoch_analysis(
        self,
        vs_thresh: float,
        use_int_flux: bool,
        expected_shape: int,
        expected_ids: List[int],
        dummy_PipeAnalysis_wtwoepoch: vtp.PipeAnalysis,
    ) -> None:
        """
        Tests the main run two epoch analysis function. Tests using int flux
        and passing custom threshold.

        Args:
            vs_thresh: The threshold to use for the vs metric.
            use_int_flux: The True, False flag from the parametrize for using
                int fluxes.
            expected_shape: Expected shape of the result.
            expected_ids: The expected ids of the sources returned.
            dummy_PipeAnalysis_wtwoepoch: The dummy PipeAnalysis object that
                is used for testing that includes the two epoch metrics.

        Returns:
            None
        """
        result_sources, result_pairs = (
            dummy_PipeAnalysis_wtwoepoch.run_two_epoch_analysis(
                vs_thresh,
                0.26,
                use_int_flux=use_int_flux
            )
        )

        assert result_sources.shape[0] == expected_shape
        assert np.all(result_sources.index == expected_ids)

    @pytest.mark.parametrize(
        'use_int_flux,expected_values',
        [
            (False, [0.5, 1., 1., 2.]),
            (True, [1.5, 3., 2., 4.])
        ]
    )
    def test__fit_eta_v(
        self,
        use_int_flux: bool,
        expected_values: List[float],
        dummy_PipeAnalysis: vtp.PipeAnalysis,
        gen_sources_metrics_df: pd.DataFrame
    ) -> None:
        """
        Tests fitting the values to the eta and v distributions.
        Integrated flux is also tested through the parametrisation.

        Args:
            use_int_flux: The True, False flag from the parametrize for using
                int fluxes.
            expected_shape: Expected fit values.
            dummy_PipeAnalysis: The dummy PipeAnalysis object that is used
                for testing.
            gen_sources_metrics_df: A generated sources dataframe containing
                randomly generated metric values (pytest fixture).

        Returns:
            None
        """
        result = dummy_PipeAnalysis._fit_eta_v(
            gen_sources_metrics_df,
            use_int_flux=use_int_flux
        )

        assert result == pytest.approx(expected_values, rel=1e-1)

    def test__gaussian_fit(
        self,
        dummy_PipeAnalysis: vtp.PipeAnalysis,
        mocker: MockerFixture
    ) -> None:
        """
        Tests performing the gaussian fits to the source metrics.
        The actual norm.pdf function is not tested, just the calls to the
        function are asserted on.

        Args:
            dummy_PipeAnalysis: The dummy PipeAnalysis object that is used
                for testing.

        Returns:
            None
        """
        norm_pdf_mocker = mocker.patch(
            'vasttools.pipeline.norm.pdf',
            return_value=-99
        )

        test_data = pd.Series([0.03, 38., 84., 1090.])
        test_mean = 1.
        test_sigma = 2.

        result = dummy_PipeAnalysis._gaussian_fit(
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
        dummy_PipeAnalysis: vtp.PipeAnalysis,
        gen_sources_metrics_df: pd.DataFrame
    ) -> None:
        """
        Tests the bin generation. Because the values in gen_sources_metrics_df
        are random there can be a slight change in the number of bins. It
        should be 18 +/- 1.

        Args:
            dummy_PipeAnalysis: The dummy PipeAnalysis object that is used
                for testing.
            gen_sources_metrics_df: A generated sources dataframe containing
                randomly generated metric values (pytest fixture).

        Returns:
            None
        """
        result_bins = dummy_PipeAnalysis._make_bins(
            np.log10(gen_sources_metrics_df['eta_peak'])
        )

        assert len(result_bins) == pytest.approx(18, rel=1)

    def test_eta_v_diagnostic_plot(
        self,
        dummy_PipeAnalysis: vtp.PipeAnalysis,
        gen_sources_metrics_df: pd.DataFrame
    ) -> None:
        """
        Smoke test for the eta v diagnostic plot.

        Args:
            dummy_PipeAnalysis: The dummy PipeAnalysis object that is used
                for testing.
            gen_sources_metrics_df: A generated sources dataframe containing
                randomly generated metric values (pytest fixture).

        Returns:
            None
        """
        plot = dummy_PipeAnalysis.eta_v_diagnostic_plot(
            1.5,
            1.5,
            df=gen_sources_metrics_df
        )

        # smoke test
        assert True

    def test__plot_eta_v_matplotlib(
        self,
        dummy_PipeAnalysis: vtp.PipeAnalysis,
        gen_sources_metrics_df: pd.DataFrame,
        mocker: MockerFixture
    ) -> None:
        """
        Smoke test for the eta v plot (matplotlib).

        Args:
            dummy_PipeAnalysis: The dummy PipeAnalysis object that is used
                for testing.
            gen_sources_metrics_df: A generated sources dataframe containing
                randomly generated metric values (pytest fixture).

        Returns:
            None
        """
        make_bins_mocker = mocker.patch(
            'vasttools.pipeline.PipeAnalysis._make_bins',
            return_value=[0, 1, 2, 3, 4]
        )

        plot = dummy_PipeAnalysis._plot_eta_v_matplotlib(
            gen_sources_metrics_df,
            0.5,
            1.,
            1.,
            2.,
            1.5,
            1.5,
        )

        # smoke test
        assert True

    def test__plot_eta_v_bokeh(
        self,
        dummy_PipeAnalysis: vtp.PipeAnalysis,
        gen_sources_metrics_df: pd.DataFrame,
        mocker: MockerFixture
    ) -> None:
        """
        Smoke test for the eta v plot (bokeh).

        Args:
            dummy_PipeAnalysis: The dummy PipeAnalysis object that is used
                for testing.
            gen_sources_metrics_df: A generated sources dataframe containing
                randomly generated metric values (pytest fixture).

        Returns:
            None
        """
        make_bins_mocker = mocker.patch(
            'vasttools.pipeline.PipeAnalysis._make_bins',
            return_value=[0, 1, 2, 3, 4]
        )

        plot = dummy_PipeAnalysis._plot_eta_v_bokeh(
            gen_sources_metrics_df,
            0.5,
            1.,
            1.,
            2.,
            1.5,
            1.5,
        )

        # smoke test
        assert True

    # No test for run_eta_v_analysis as it uses all the functions tested
    # above.
