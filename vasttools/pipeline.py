import numexpr
import os
import warnings
import glob
import gc
import vaex
import dask.dataframe as dd
from typing import Dict, List, Tuple
import bokeh.colors.named as colors
from bokeh.models import (
    ColumnDataSource,
    Span,
    BoxAnnotation,
    Model,
    DataRange1d,
    Range1d,
    Whisker,
    LabelSet,
    Circle,
    HoverTool,
    Slider
)
from bokeh.layouts import gridplot, Spacer
from bokeh.palettes import Category10_3
from bokeh.plotting import figure, from_networkx
from bokeh.transform import linear_cmap, factor_cmap
import colorcet as cc
import numpy as np
import pandas as pd
from scipy.stats import norm
import scipy.ndimage as ndi
from astropy.stats import sigma_clip, mad_std
from astropy.coordinates import SkyCoord
from astropy import units as u
from multiprocessing import cpu_count
from mocpy import MOC
from vasttools.source import Source
from vasttools.utils import (
    match_planet_to_field,
    pipeline_get_variable_metrics,
    pipeline_get_eta_metric,
    gen_skycoord_from_df
)
from vasttools.survey import Image
from datetime import timedelta
from itertools import combinations
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from matplotlib.font_manager import FontProperties
from astroML import density_estimation


HOST_NCPU = cpu_count()
numexpr.set_num_threads(int(HOST_NCPU / 4))
matplotlib.pyplot.switch_backend('Agg')


class Pipeline(object):
    '''
    Class to interface with VAST Pipeline results.

    Attributes
    ----------

    project_dir : str
        The pipeline project directory provided by the user on
        initialisation.

    Methods
    -------

    list_piperuns()
        Lists the avaialble pipeline runs in the directory.

    list_images()
        Lists all the images that have been processed in jobs
        associated with the pipeline directory.

    load_run(run_name, n_workers=cpu_count()-1)
        Loads the pipeline run defined by run_name.
        Returns a PipeRun object.

    load_runs(run_names, name, n_workers=cpu_count()-1)
        Loads a list of run names into to one pipeline object.
        Returns a PipeRun object.
    '''

    def __init__(self, project_dir=None):
        '''
        Constructor method.

        The system variable `PIPELINE_WORKING_DIR` will be checked
        first with the project_dir inpuut the fallback option.

        :param project_dir: The directory of the pipeline results,
            only required when the system variable is not defined,
            defaults to 'None'.
        :type project_dir: str, optional
        '''
        super(Pipeline, self).__init__()

        if project_dir is None:
            try:
                pipeline_run_path = os.getenv('PIPELINE_WORKING_DIR')
            except Exception as e:
                raise Exception(
                    "The pipeline run directory could not be determined!"
                    " Either the system environment 'PIPELINE_WORKING_DIR'"
                    " must be defined or the 'project_dir' argument defined"
                    " when initialising the pipeline class object."
                )
        else:
            pipeline_run_path = os.path.abspath(str(project_dir))

        if not os.path.isdir(pipeline_run_path):
            raise Exception(
                "Pipeline run directory {} not found!".format(
                    pipeline_run_path
                )
            )

        self.project_dir = pipeline_run_path

    def list_piperuns(self):
        '''
        Lists the runs present in the pipeline directory.

        :returns: List of pipeline run names present in directory.
        :rtype: list
        '''
        jobs = sorted(glob.glob(
            os.path.join(self.project_dir, "*")
        ))

        jobs = [i.split("/")[-1] for i in jobs]
        jobs.remove('images')

        return jobs

    def list_images(self):
        '''
        Lists all images processed in the pipeline directory.

        :returns: List of images processed.
        :rtype: list
        '''
        img_list = sorted(glob.glob(
            os.path.join(self.project_dir, "images", "*")
        ))

        img_list = [i.split("/")[-1] for i in img_list]

        return img_list

    def load_runs(self, run_names, name=None, n_workers=cpu_count() - 1):
        '''
        Wrapper to load multiple runs in one command.

        :param run_names: List containing the names of the runs
            to load.
        :type run_names: list
        :param name: State a name for the pipeline run.
        :type name: str
        :param n_workers: The number of workers (cpus)
            available.
        :type run_name: int, optional

        :returns: Combined PipeAnalysis object.
        :rtype: vasttools.pipeline.PipeAnalysis
        '''
        piperun = self.load_run(
            run_names[0],
            n_workers=n_workers
        )

        if len(run_names) > 1:
            for r in run_names[1:]:
                piperun = piperun.combine_with_run(
                    self.load_run(
                        r,
                        n_workers=n_workers
                    )
                )
        if name is not None:
            piperun.name = name

        return piperun

    def load_run(
        self, run_name, n_workers=cpu_count() - 1
    ):
        '''
        Process and load a pipeline run.

        :param run_name: The name of the run to load.
        :type run_name: str
        :param n_workers: The number of workers (cpus)
            available.
        :type run_name: int, optional

        :returns: PipeAnalysis object.
        :rtype: vasttools.pipeline.PipeAnalysis
        '''

        run_dir = os.path.join(
            self.project_dir,
            run_name
        )

        if not os.path.isdir(run_dir):
            raise ValueError(
                "Run '%s' does not exist!",
                run_name
            )
            return

        images = pd.read_parquet(
            os.path.join(
                run_dir,
                'images.parquet'
            )
        )

        skyregions = pd.read_parquet(
            os.path.join(
                run_dir,
                'skyregions.parquet'
            ),
            engine='pyarrow'
        )

        bands = pd.read_parquet(
            os.path.join(
                run_dir,
                'bands.parquet'
            ),
            engine='pyarrow'
        )

        images = images.merge(
            skyregions[[
                'id',
                'centre_ra',
                'centre_dec',
                'xtr_radius'
            ]], how='left',
            left_on='skyreg_id',
            right_on='id'
        ).drop(
            'id_y', axis=1
        ).rename(
            columns={'id_x': 'id'}
        ).merge(  # second merge for band
            bands[['id', 'frequency', 'bandwidth']],
            how='left',
            left_on='band_id',
            right_on='id'
        ).drop(
            'id_y', axis=1
        ).rename(
            columns={'id_x': 'id'}
        )

        relations = pd.read_parquet(
            os.path.join(
                run_dir,
                'relations.parquet'
            ),
            engine='pyarrow'
        )

        sources = pd.read_parquet(
            os.path.join(
                run_dir,
                'sources.parquet'
            ),
            engine='pyarrow'
        )

        to_move = ['n_meas', 'n_meas_sel', 'n_meas_forced', 'n_sibl', 'n_rel']
        sources_len = sources.shape[1]
        for c in to_move:
            col = sources.pop(c)
            sources.insert(sources_len - 1, c, col)

        sources = sources.rename(
            columns={
                'n_meas_forced': 'n_forced',
                'n_meas': 'n_measurements',
                'n_meas_sel': 'n_selavy',
                'n_sibl': 'n_siblings',
                'n_rel': 'n_relations'
            }
        )

        associations = pd.read_parquet(
            os.path.join(
                run_dir,
                'associations.parquet'
            ),
            engine='pyarrow'
        )

        vaex_meas = False

        if os.path.isfile(os.path.join(
            run_dir,
            'measurements.arrow'
        )):
            measurements = vaex.open(
                os.path.join(run_dir, 'measurements.arrow')
            )

            vaex_meas = True

            warnings.warn("Measurements have been loaded with vaex.")

        else:
            m_files = images['measurements_path'].tolist()
            m_files += sorted(glob.glob(os.path.join(
                run_dir,
                "forced_measurements*.parquet"
            )))

            # use dask to open measurement parquets
            # as they are spread over many different files
            measurements = dd.read_parquet(
                m_files,
                engine='pyarrow'
            ).compute()

            measurements = measurements.loc[
                measurements['id'].isin(associations['meas_id'].values)
            ]

            measurements = (
                associations.loc[:, ['meas_id', 'source_id']]
                .set_index('meas_id')
                .merge(
                    measurements,
                    left_index=True,
                    right_on='id'
                )
                .rename(columns={'source_id': 'source'})
            )

        images = images.set_index('id')

        if os.path.isfile(os.path.join(
            run_dir,
            "measurement_pairs.arrow"
        )):
            measurement_pairs_file = [os.path.join(
                run_dir,
                "measurement_pairs.arrow"
            )]
        else:
            measurement_pairs_file = [os.path.join(
                run_dir,
                "measurement_pairs.parquet"
            )]

        piperun = PipeAnalysis(
            name=run_name,
            images=images,
            skyregions=skyregions,
            relations=relations,
            sources=sources,
            associations=associations,
            measurements=measurements,
            measurement_pairs_file=measurement_pairs_file,
            vaex_meas=vaex_meas
        )

        return piperun


class PipeRun(object):
    '''
    Class that represents a Pipeline run.

    Attributes
    ----------

    name : str
        The pipeline run name.
    images : pandas.core.frame.DataFrame
        Dataframe containing all the information on the images
        of the pipeline run.
    skyregions : pandas.core.frame.DataFrame
        Dataframe containing all the information on the skyregions
        of the pipeline run.
    sources : pandas.core.frame.DataFrame
        Dataframe containing all the information on the sources
        of the pipeline run.
    sources_skycoord : astropy.coordinates.SkyCoord
        A SkyCoord object of the default sources attribute.
    measurements : pandas.core.frame.DataFrame
        Dataframe containing all the information on the measurements
        of the pipeline run.
    measurement_pairs_file : list
        List containing the locations of the measurement_pairs.parquet (or
        .arrow) file(s).
    relations : pandas.core.frame.DataFrame
        Dataframe containing all the information on the relations
        of the pipeline run.
    n_workers : pandas.core.frame.DataFrame
        Number of workers (cpus) available.

    Methods
    -------

    get_sources_skycoord(user_sources=None, ra_col='wavg_ra',
        dec_col='wavg_dec', ra_unit=u.degree, dec_unit=u.degree)
        A convenience function to generate a SkyCoord object from the
        sources dataframe. Also has support for custom source lists.

    get_source(id, field=None, stokes='I', outdir='.')
        Creates a vasttools.source.Source object for the requested
        source.

    check_for_planets()
        Searches the pipeline run images for any planets present.
        Returns pandas dataframe with results.

    create_moc(max_depth=9)
        Create a MOC file that represents the area covered by
        the pipeline run.

    combine_with_run(other_PipeRun, new_name=None)
        Combines the output of another PipeRun object with the PipeRun
        from which this method is being called from.

        WARNING! It is assumed you are loading runs from the same Pipeline
        instance. If this is not the case then erroneous results may be
        returned.

    load_two_epoch_metrics()
        Loads the two epoch metrics dataframe, usually stored as either
        'measurement_pairs.parquet' or 'measurement_pairs.arrow', and is added
        as an attribute named 'measurement_pairs_df'. Adds an epoch
        'key' to the dataframe. Also creates a 'pairs_df' that lists all the
        possible epoch pairs.
    '''
    def __init__(
        self, name, images, skyregions, relations, sources,
        associations, measurements, measurement_pairs_file, vaex_meas=False,
        n_workers=cpu_count() - 1
    ):
        '''
        Constructor method.

        :param name: The name of the pipeline run.
        :type project_dir: str
        :param images: Images dataframe from the pipeline run
            loaded from images.parquet.
        :type images: pandas.core.frame.DataFrame
        :param skyregions: Images dataframe from the pipeline run
            loaded from skyregions.parquet.
        :type skyregions: pandas.core.frame.DataFrame
        :param sources: Sources dataframe from the pipeline run
            loaded from sources.parquet.
        :type sources: pandas.core.frame.DataFrame
        :param measurements: Measurements dataframe from the pipeline run
            loaded from measurements.parquet and the forced measurements
            parquet files.
        :type measurements: pandas.core.frame.DataFrame
        :param measurement_pairs: Two epoch pairs dataframe from the pipeline
            run loaded from measurement_pairs.parquet.
        :type measurement_pairs: pandas.core.frame.DataFrame
        :param relations: Relations dataframe from the pipeline run
            loaded from relations.parquet.
        :type relations: pandas.core.frame.DataFrame
        :param n_workers: Number of workers (cpus) available.
        :type n_workers: int
        '''
        super(PipeRun, self).__init__()
        self.name = name
        self.images = images
        self.skyregions = skyregions
        self.sources = sources
        self.sources_skycoord = self.get_sources_skycoord()
        self.associations = associations
        self.measurements = measurements
        self.measurement_pairs_file = measurement_pairs_file
        self.relations = relations
        self.n_workers = n_workers
        self._vaex_meas = vaex_meas
        self._loaded_two_epoch_metrics = False

    def combine_with_run(self, other_PipeRun, new_name=None):
        '''
        Combines the output of another PipeRun object with the PipeRun
        from which this method is being called from.

        WARNING! It is assumed you are loading runs from the same Pipeline
        instance. If this is not the case then erroneous results may be
        returned.

        :param other_PipeRun: The other pipeline run to merge.
        :type other_PipeRun: vasttools.pipeline.PipeRun
        :param new_name: If not None then the PipeRun attribute 'name'
            is changed to the given value.
        :type new_name: str, optional
        '''

        self.images = self.images.append(
            other_PipeRun.images,
        ).drop_duplicates('path')

        self.skyregions = self.skyregions.append(
            other_PipeRun.skyregions,
            ignore_index=True
        ).drop_duplicates('id')

        if self._vaex_meas and other_PipeRun._vaex_meas:
            self.measurements = self.measurements.concat(
                other_PipeRun.measurements
            )

        elif self._vaex_meas and not other_PipeRun._vaex_meas:
            self.measurements = self.measurements.concat(
                vaex.from_pandas(other_PipeRun.measurements)
            )

        elif not self._vaex_meas and other_PipeRun._vaex_meas:
            self.measurements = vaex.from_pandas(self.measurements).concat(
                other_PipeRun.measurements
            )
            self._vaex_meas = True

        else:
            self.measurements = self.measurements.append(
                other_PipeRun.measurements,
                ignore_index=True
            ).drop_duplicates(['id', 'source'])

        sources_to_add = other_PipeRun.sources.loc[
            ~(other_PipeRun.sources.index.isin(
                self.sources.index
            ))
        ]

        self.sources = self.sources.append(
            sources_to_add
        )

        # need to keep access to all the different pairs files
        # for two epoch metrics.
        for i in other_PipeRun.measurement_pairs_file:
            self.measurement_pairs_file.append(i)

        del sources_to_add

        if new_name is not None:
            self.name = new_name

        return self

    def get_sources_skycoord(
        self, user_sources=None, ra_col='wavg_ra', dec_col='wavg_dec',
        ra_unit=u.degree, dec_unit=u.degree
    ):
        '''
        A convenience function to generate a SkyCoord object from the
        sources dataframe. Also has support for custom source lists.

        :param user_sources: Provide a user generated source dataframe
            in place of using the default run sources dataframe.
        :type user_sources: pandas.core.frame.DataFrame, optional
        :param ra_col: The column to use for the Right Ascension.
        :type ra_col: str, optional
        :param dec_col: The column to use for the Declination.
        :type dec_col: str, optional
        :param ra_unit: The unit of the RA column, defaults to degrees.
            Must be an astropy.unit value.
        :type ra_unit: astropy.unit, optional
        :param dec_unit: The unit of the Dec column, defaults to degrees.
            Must be an astropy.unit value.
        :type dec_unit: astropy.unit, optional

        :returns: sources_skycoord
        :rtype: astropy.coordinates.SkyCoord
        '''
        if user_sources is None:
            the_sources = self.sources
        else:
            the_sources = user_sources

        sources_skycoord = gen_skycoord_from_df(
            the_sources, ra_col=ra_col, dec_col=dec_col, ra_unit=ra_unit,
            dec_unit=dec_unit
        )

        return sources_skycoord

    def get_source(
        self, id, field=None, stokes='I', outdir='.', user_measurements=None,
        user_sources=None
    ):
        '''
        Fetches an individual source and returns a
        vasttools.source.Source object. Users do not need
        to change the field, stokes and outdir parameters.

        :param id: The id of the source to load.
        :type run_name: int
        :param field: The field of the source being loaded, defaults
            to None. If None then the run name is used as the field.
        :type field: str, optional
        :param stokes: Stokes parameter of the source, defaults to 'I'.
        :type stokes: str, optional
        :param outdir: The output directory where generated plots will
            be saved, defauls to '.' (the current working directory).
        :type outdir: str, optional

        :returns: vast tools Source object
        :rtype: vasttools.source.Source
        '''

        if user_measurements is None:
            the_measurements = self.measurements
        else:
            the_measurements = user_measurements

        if user_sources is None:
            the_sources = self.sources
        else:
            the_sources = user_sources

        if self._vaex_meas:
            measurements = the_measurements[
                the_measurements['source'] == id
            ].to_pandas_df()

        else:
            measurements = the_measurements.loc[
                the_measurements['source'] == id
            ]

        measurements = measurements.merge(
            self.images[[
                'path',
                'noise_path',
                'measurements_path',
                'frequency'
            ]], how='left',
            left_on='image_id',
            right_index=True
        ).rename(
            columns={
                'path': 'image',
                'noise_path': 'rms',
                'measurements_path': 'selavy'
            }
        )

        measurements = measurements.rename(
            columns={
              'time': 'dateobs',
            }
        ).sort_values(
            by='dateobs'
        ).reset_index(drop=True)

        s = the_sources.loc[id]

        num_measurements = s['n_measurements']

        source_coord = SkyCoord(
            s['wavg_ra'],
            s['wavg_dec'],
            unit=(u.deg, u.deg)
        )

        source_name = "VAST {}".format(
            source_coord.to_string(
                "hmsdms", sep='', precision=1
            ).replace(
                " ", ""
            )[:15]
        )
        source_epochs = [str(i) for i in range(1, num_measurements + 1)]
        if field is None:
            field = self.name
        measurements['field'] = field
        measurements['epoch'] = source_epochs
        measurements['stokes'] = stokes
        measurements['skycoord'] = source_coord
        measurements['detection'] = measurements['forced'] == False
        source_fields = [field for i in range(num_measurements)]
        source_stokes = stokes
        source_base_folder = None
        source_crossmatch_radius = None
        source_outdir = outdir
        source_image_type = None

        thesource = Source(
            source_coord,
            source_name,
            source_epochs,
            source_fields,
            source_stokes,
            None,
            source_crossmatch_radius,
            measurements,
            source_base_folder,
            source_image_type,
            islands=False,
            outdir=source_outdir,
            pipeline=True
        )

        return thesource

    def load_two_epoch_metrics(self):
        """
        Loads the two epoch metrics dataframe, usually stored as either
        'measurement_pairs.parquet' or 'measurement_pairs.arrow'. Adds an epoch
        'key' to the dataframe. Also creates a 'pairs_df' that lists all the
        possible epoch pairs.

        :param None:

        :returns: None
        """
        image_ids = self.images.sort_values(by='datetime').index.tolist()

        pairs_df = pd.DataFrame.from_dict(
            {'pair': combinations(image_ids, 2)}
        )

        pairs_df = (
            pd.DataFrame(pairs_df['pair'].tolist())
            .rename(columns={0: 'image_id_a', 1: 'image_id_b'})
            .merge(
                self.images[['datetime', 'name']],
                left_on='image_id_a', right_index=True,
                suffixes=('_a', '_b')
            )
            .merge(
                self.images[['datetime', 'name']],
                left_on='image_id_b', right_index=True,
                suffixes=('_a', '_b')
            )
        ).reset_index().rename(
            columns={
                'index': 'id',
                'name_a': 'image_name_a',
                'name_b': 'image_name_b'
            }
        )

        pairs_df['td'] = pairs_df['datetime_b'] - pairs_df['datetime_a']

        pairs_df.drop(['datetime_a', 'datetime_b'], axis=1)

        pairs_df['pair_epoch_key'] = (
            pairs_df[['image_name_a', 'image_name_b']]
            .apply(
                lambda x: f"{x['image_name_a']}_{x['image_name_b']}", axis=1
            )
        )

        self._vaex_meas_pairs = False
        if len(self.measurement_pairs_file) > 1:
            arrow_files = (
                [i.endswith(".arrow") for i in self.measurement_pairs_file]
            )
            if np.any(arrow_files):
                measurement_pairs_df = vaex.open_many(
                    self.measurement_pairs_file[arrow_files]
                )
                for i in self.measurement_pairs_file[~arrow_files]:
                    temp = pd.read_parquet(i)
                    temp = vaex.from_pandas(temp)
                    measurement_pairs_df = measurement_pairs_df.concat(temp)
                self._vaex_meas_pairs = True
                warnings.warn("Measurement pairs have been loaded with vaex.")
            else:
                measurement_pairs_df = (
                    dd.read_parquet(self.measurement_pairs_file).compute()
                )
        else:
            if self.measurement_pairs_file[0].endswith('.arrow'):
                measurement_pairs_df = (
                    vaex.open(self.measurement_pairs_file[0])
                )
                self._vaex_meas_pairs = True
                warnings.warn("Measurement pairs have been loaded with vaex.")
            else:
                measurement_pairs_df = (
                    pd.read_parquet(self.measurement_pairs_file[0])
                )

        if self._vaex_meas_pairs:
            measurement_pairs_df['pair_epoch_key'] = (
                measurement_pairs_df['image_name_a'] + "_"
                + measurement_pairs_df['image_name_b']
            )

            pair_counts = measurement_pairs_df.groupby(
                measurement_pairs_df.pair_epoch_key, agg='count'
            )

            pair_counts = pair_counts.to_pandas_df().rename(
                columns={'count': 'total_pairs'}
            ).set_index('pair_epoch_key')
        else:
            measurement_pairs_df['pair_epoch_key'] = (
                measurement_pairs_df[['image_name_a', 'image_name_b']]
                .apply(
                    lambda x: f"{x['image_name_a']}_{x['image_name_b']}",
                    axis=1
                )
            )

            pair_counts = measurement_pairs_df[
                ['pair_epoch_key', 'image_name_a']
            ].groupby('pair_epoch_key').count().rename(
                columns={'image_name_a': 'total_pairs'}
            )

        pairs_df = pairs_df.merge(
            pair_counts, left_on='pair_epoch_key', right_index=True
        )

        del pair_counts

        pairs_df = pairs_df.dropna(subset=['total_pairs']).set_index('id')

        self.measurement_pairs_df = measurement_pairs_df
        self.pairs_df = pairs_df.sort_values(by='td')

        self._loaded_two_epoch_metrics = True

    def _add_times(self, row, duration=True, every_hour=False):
        '''
        Adds the times required for planet searching. By default it
        adds the beginning and end of the observation. The every_hour
        option adds the time every hour during the observation, which
        is required for the Sun and Moon.

        :param row: The series row containing the information.
        :type row: pandas.core.frame.Series
        :param duration: Add the times at the beginning and end of
            the observation, defaults to 'True'.
        :type duration: bool, optional
        :param every_hour: Add times to the dataframe every hour
            during the observation, defaults to 'False'.
        :type every_hour: bool, optional

        :returns: List of times to be searched for planets, in the
            format of rows.
        :rtype: list
        '''
        if row['duration'] == 0:
            return row['DATEOBS']

        elif duration:
            return [
                row['DATEOBS'],
                row['DATEOBS'] + timedelta(
                    seconds=row['duration']
                )
            ]

        elif every_hour:
            hours = int(row['duration'] / 3600.)
            times = [
                row['DATEOBS'] + timedelta(
                    seconds=row['duration'] * h
                )
                for h in range(hours + 1)
            ]
            return times

    def check_for_planets(self):
        '''
        Checks the pipeline run for any planets in the field.
        All planets are checked: Mercury, Venus, Mars, Jupiter,
        Saturn, Uranus, Neptune, Pluto, Sun, Moon.

        :returns: DataFrame with list of planet positions. Empty
            if no planets are found.
        :rtype: pandas.core.frame.DataFrame
        '''

        from vasttools.survey import ALLOWED_PLANETS
        ap = ALLOWED_PLANETS

        planets_df = self.images.loc[:, [
            'datetime',
            'duration',
            'centre_ra',
            'centre_dec',
        ]].rename(
            columns={
                'datetime': 'DATEOBS',
                'centre_ra': 'centre-ra',
                'centre_dec': 'centre-dec'
            }
        )

        # Split off a sun and moon df so we can check more times
        sun_moon_df = planets_df.copy()
        ap.remove('sun')
        ap.remove('moon')

        # check planets at start and end of observation
        planets_df['DATEOBS'] = planets_df[['DATEOBS', 'duration']].apply(
            self._add_times,
            axis=1
        )
        planets_df['planet'] = [ap for i in range(planets_df.shape[0])]

        # check sun and moon every hour
        sun_moon_df['DATEOBS'] = sun_moon_df[['DATEOBS', 'duration']].apply(
            self._add_times,
            args=(False, True),
            axis=1
        )

        sun_moon_df['planet'] = [
            ['sun', 'moon'] for i in range(sun_moon_df.shape[0])
        ]

        planets_df = planets_df.append(sun_moon_df, ignore_index=True)

        del sun_moon_df

        planets_df = planets_df.explode('planet').explode('DATEOBS').drop(
            'duration', axis=1
        )
        planets_df['planet'] = planets_df['planet'].str.capitalize()

        meta = {
            'id': 'i',
            'DATEOBS': 'datetime64[ns]',
            'centre-ra': 'f',
            'centre-dec': 'f',
            'planet': 'U',
            'ra': 'f',
            'dec': 'f',
            'sep': 'f'
        }

        result = (
            dd.from_pandas(planets_df, self.n_workers)
            .groupby('planet')
            .apply(
                match_planet_to_field,
                meta=meta
            ).compute(
                scheduler='processes',
                n_workers=self.n_workers
            )
        )

        if result.empty:
            warnings.warn("No planets found.")

        return result

    def _distance_from_edge(self, x):
        '''
        Analyses the binary array x and determines the distance from
        the edge (0).

        :param x: The binary array to analyse.
        :type x: numpy.ndarray

        :returns: Array each cell containing distance from the edge.
        :rtype: numpy.ndarray
        '''
        x = np.pad(x, 1, mode='constant')
        dist = ndi.distance_transform_cdt(x, metric='taxicab')

        return dist[1:-1, 1:-1]

    def _create_moc_from_fits(self, fits_img, max_depth=9):
        '''
        Creates a MOC from (assuming) an ASKAP fits image
        using the cheat method of analysing the edge pixels of the image.

        :param fits_img: The path of the ASKAP FITS image to
            generate the MOC from.
        :type fits_img: str
        :param max_depth: Max depth parameter passed to the
            MOC.from_polygon_skycoord() function, defaults to 9.
        :type max_depth: int, optional

        :returns: The MOC generated from the FITS file.
        :rtype: mocpy.moc.moc.MOC
        '''
        image = Image(
            'field', '1', 'I', 'None',
            path=fits_img
        )

        binary = (~np.isnan(image.data)).astype(int)
        mask = self._distance_from_edge(binary)
        x, y = np.where(mask == 1)

        array_coords = np.column_stack((x, y))
        coords = image.wcs.array_index_to_world_values(array_coords)
        # need to know when to reverse by checking axis sizes.
        coords = np.column_stack(coords)
        coords = SkyCoord(coords[0], coords[1], unit=(u.deg, u.deg))

        moc = MOC.from_polygon_skycoord(coords, max_depth=max_depth)

        del image
        del binary
        del array_coords
        gc.collect()

        return moc

    def create_moc(self, max_depth=9, ignore_large_run_warning=False):
        '''
        Create a MOC file that represents the area covered by
        the pipeline run.

        WARNING! This will take a very long time for large runs.

        :param max_depth: Max depth parameter passed to the
            MOC.from_polygon_skycoord() function, defaults to 9.
        :type max_depth: int, optional
        :param ignore_large_run_warning: Ignores the warning of
            creating a MOC on a large run.
        :type ignore_large_run_warning: bool, optional

        :returns: MOC object.
        :rtype: mocpy.moc.moc.MOC
        '''

        images_to_use = self.images.drop_duplicates(
            'skyreg_id'
        )['path'].values

        if not ignore_large_run_warning and images_to_use.shape[0] > 10:
            warnings.warn(
                "Creating a MOC for a large run will take a long time!"
                " Run again with 'ignore_large_run_warning=True` if you"
                " are sure you want to run this. A smaller `max_depth` is"
                " highly recommended."
            )
            return

        moc = self._create_moc_from_fits(
            images_to_use[0],
            max_depth=max_depth
        )

        if images_to_use.shape[0] > 1:
            for img in images_to_use[1:]:
                img_moc = self._create_moc_from_fits(
                    img,
                    max_depth
                )
                moc = moc.union(img_moc)

        return moc


class PipeAnalysis(PipeRun):
    '''
    Class that represents an Analysis instance of a Pipeline run.
    Inherits from class `PipeRun`.

    Attributes
    ----------

    name : str
        The pipeline run name.
    images : pandas.core.frame.DataFrame
        Dataframe containing all the information on the images
        of the pipeline run.
    skyregions : pandas.core.frame.DataFrame
        Dataframe containing all the information on the skyregions
        of the pipeline run.
    sources : pandas.core.frame.DataFrame
        Dataframe containing all the information on the sources
        of the pipeline run.
    sources_skycoord : astropy.coordinates.SkyCoord
        A SkyCoord object of the default sources attribute.
    measurements : pandas.core.frame.DataFrame
        Dataframe containing all the information on the measurements
        of the pipeline run.
    measurement_pairs_file : list
        List containing the locations of the measurement_pairs.parquet (or
        .arrow) file(s).
    relations : pandas.core.frame.DataFrame
        Dataframe containing all the information on the relations
        of the pipeline run.
    n_workers : pandas.core.frame.DataFrame
        Number of workers (cpus) available.

    Methods
    -------

    recalc_sources_df(measurements_df, min_vs=4.3)
        Regenreates a sources dataframe using a user provided measurements
        dataframe.

    get_sources_skycoord(user_sources=None, ra_col='wavg_ra',
        dec_col='wavg_dec', ra_unit=u.degree, dec_unit=u.degree)
        A convenience function to generate a SkyCoord object from the
        sources dataframe. Also has support for custom source lists.

    get_source(id, field=None, stokes='I', outdir='.')
        Creates a vasttools.source.Source object for the requested
        source.

    check_for_planets()
        Searches the pipeline run images for any planets present.
        Returns pandas dataframe with results.

    create_moc(max_depth=9)
        Create a MOC file that represents the area covered by
        the pipeline run.

    combine_with_run(other_PipeRun, new_name=None)
        Combines the output of another PipeRun object with the PipeRun
        from which this method is being called from.

        WARNING! It is assumed you are loading runs from the same Pipeline
        instance. If this is not the case then erroneous results may be
        returned.

    load_two_epoch_metrics()
        Loads the two epoch metrics dataframe, usually stored as either
        'measurement_pairs.parquet' or 'measurement_pairs.arrow', and is added
        as an attribute named 'measurement_pairs_df'. Adds an epoch
        'key' to the dataframe. Also creates a 'pairs_df' that lists all the
        possible epoch pairs.

    run_two_epoch_analysis(v, m, query=None, df=None, use_int_flux=False)
        Runs the two epoch variability analysis on the pipeline run. Filters
        can be applied using query argument or directly passing the filtered
        sources df. Returns two dataframes: the candidates sources given the
        input v and m values and the pair values that met the thresholds.

    plot_two_epoch_pairs(epoch_pair_id, query=None, df=None, vs_min=4.3,
        m_min=0.26, use_int_flux=False, remove_two_forced=False,
        plot_type='bokeh', plot_style='a')
        Create and returns a bokeh or matplotlib plot of the two epoch pairs of
        a defined 'pair epoch'.

    run_eta_v_analysis(eta_sigma, v_sigma, query=None, df=None,
        use_int_flux=False, plot_type='bokeh', diagnostic=False)
        Runs the analysis based on the `eta` and `V` metrics that are returned
        by the pipeline. Returns the eta and v cutoff values and the list of
        candidates based on the entered sigma values, a results plot (either
        bokeh or matplotlib based) and a matplotlib diagnostics plot if
        selected.

    eta_v_diagnostic_plot(df, eta_cutoff, v_cutoff, use_int_flux=False)
        Returns the eta and V based diagnostic plot (matplotlib). Requires
        eta and V cutoff values from `run_eta_v_analysis` and the sources
        dataframe.
    '''
    def __init__(
        self, name, images, skyregions, relations, sources, associations,
        measurements, measurement_pairs_file, vaex_meas=False,
        n_workers=cpu_count() - 1
    ):
        '''
        Constructor method.

        :param name: The name of the pipeline run.
        :type project_dir: str
        :param images: Images dataframe from the pipeline run
           loaded from images.parquet.
        :type images: pandas.core.frame.DataFrame
        :param skyregions: Images dataframe from the pipeline run
           loaded from skyregions.parquet.
        :type skyregions: pandas.core.frame.DataFrame
        :param sources: Sources dataframe from the pipeline run
           loaded from sources.parquet.
        :type sources: pandas.core.frame.DataFrame
        :param measurements: Measurements dataframe from the pipeline run
           loaded from measurements.parquet and the forced measurements
           parquet files.
        :type measurements: pandas.core.frame.DataFrame
        :param measurement_pairs: Two epoch pairs dataframe from the pipeline
            run loaded from measurement_pairs.parquet.
        :type measurement_pairs: pandas.core.frame.DataFrame
        :param relations: Relations dataframe from the pipeline run
           loaded from relations.parquet.
        :type relations: pandas.core.frame.DataFrame
        :param n_workers: Number of workers (cpus) available.
        :type n_workers: int, optional
        '''
        super().__init__(
            name, images, skyregions, relations, sources, associations,
            measurements, measurement_pairs_file, vaex_meas, n_workers
        )

    def recalc_sources_df(self, measurements_df, min_vs=4.3):
        """
        Regenreates a sources dataframe using a user provided measurements
        dataframe.

        :param measurements_df: Dataframe of measurements with default pipeline
            columns.
        :type measurements_df: pandas.core.frame.DataFrame.
        :param min_vs: Minimum value of the Vs two epoch parameter to use
            when appending the two epoch metrics maximum.
        :type min_vs: float.

        :returns: sources_df
        :rtype: pandas.core.frame.DataFrame.
        """

        if not self._vaex_meas:
            measurements_df = vaex.from_pandas(measurements_df)

        # account for RA wrapping
        ra_wrap_mask = measurements_df.ra <= 0.1
        measurements_df['ra_wrap'] = measurements_df.func.where(
            ra_wrap_mask, measurements_df[ra_wrap_mask].ra + 360.,
            measurements_df['ra']
        )

        measurements_df['interim_ew'] = (
           measurements_df['ra_wrap'] * measurements_df['weight_ew']
        )

        measurements_df['interim_ns'] = (
           measurements_df['dec'] * measurements_df['weight_ns']
        )

        for col in ['flux_int', 'flux_peak']:
            measurements_df[f'{col}_sq'] = (measurements_df[col] ** 2.)

        # most of the aggregate calculations done in vaex
        sources_df = measurements_df.groupby(
            by='source',
            agg={
                'interim_ew_sum': vaex.agg.sum(
                    'interim_ew', selection='forced==False'
                ),
                'interim_ns_sum': vaex.agg.sum(
                    'interim_ns', selection='forced==False'
                ),
                'weight_ew_sum': vaex.agg.sum(
                    'weight_ew', selection='forced==False'
                ),
                'weight_ns_sum': vaex.agg.sum(
                    'weight_ns', selection='forced==False'
                ),
                'avg_compactness': vaex.agg.mean(
                    'compactness', selection='forced==False'
                ),
                'min_snr': vaex.agg.min(
                    'snr', selection='forced==False'
                ),
                'max_snr': vaex.agg.max(
                    'snr', selection='forced==False'
                ),
                'avg_flux_int': vaex.agg.mean('flux_int'),
                'avg_flux_peak': vaex.agg.mean('flux_peak'),
                'max_flux_peak': vaex.agg.max('flux_peak'),
                'max_flux_int': vaex.agg.max('flux_int'),
                'min_flux_peak': vaex.agg.min('flux_peak'),
                'min_flux_int': vaex.agg.min('flux_int'),
                'min_flux_peak_isl_ratio': vaex.agg.min('flux_peak_isl_ratio'),
                'min_flux_int_isl_ratio': vaex.agg.min('flux_int_isl_ratio'),
                'n_measurements': vaex.agg.count('id'),
                'n_selavy': vaex.agg.count('id', selection='forced==False'),
                'n_forced': vaex.agg.count('id', selection='forced==True'),
                'n_siblings': vaex.agg.sum('has_siblings')
            }
        )

        # Drop sources which no longer have any selavy measurements
        sources_df = sources_df[sources_df.n_selavy > 0].extract()

        # Calculate average position
        sources_df['wavg_ra'] = (
            sources_df['interim_ew_sum'] / sources_df['weight_ew_sum']
        )
        sources_df['wavg_dec'] = (
            sources_df['interim_ns_sum'] / sources_df['weight_ns_sum']
        )

        sources_df['wavg_uncertainty_ew'] = (
            1. / np.sqrt(sources_df['weight_ew_sum'])
        )
        sources_df['wavg_uncertainty_ns'] = (
            1. / np.sqrt(sources_df['weight_ns_sum'])
        )

        # the RA wrapping is reverted at the end of the function when the
        # df is in pandas format.

        # TraP variability metrics, using Dask.
        measurements_df_temp = measurements_df[[
            'flux_int', 'flux_int_err', 'flux_peak', 'flux_peak_err', 'source'
        ]].extract().to_pandas_df()

        col_dtype = {
            'v_int': 'f',
            'v_peak': 'f',
            'eta_int': 'f',
            'eta_peak': 'f',
        }

        sources_df_fluxes = (
            dd.from_pandas(measurements_df_temp, HOST_NCPU)
            .groupby('source')
            .apply(
                pipeline_get_variable_metrics,
                meta=col_dtype
            )
            .compute(num_workers=HOST_NCPU - 1, scheduler='processes')
        )

        # Switch to pandas at this point to perform join
        sources_df = sources_df.to_pandas_df().set_index('source')

        sources_df = sources_df.join(sources_df_fluxes)

        sources_df = sources_df.join(
            self.sources[['new', 'new_high_sigma']],
        )

        # Two epoch metrics
        if not self._loaded_two_epoch_metrics:
            self.load_two_epoch_metrics()

        if self._vaex_meas_pairs:
            new_measurement_pairs = (
                self.measurement_pairs_df[
                    self.measurement_pairs_df.vs_int.abs() >= min_vs
                    or self.measurement_pairs_df.vs_peak.abs() >= min_vs
                ]
            )
        else:
            min_vs_mask = np.logical_or(
                (self.measurement_pairs_df.vs_int.abs() >= min_vs).values,
                (self.measurement_pairs_df.vs_peak.abs() >= min_vs).values
            )
            new_measurement_pairs = self.measurement_pairs_df.loc[min_vs_mask]

        # make masks of dropped measurements
        mask_a = new_measurement_pairs.meas_id_a.isin(
            measurements_df.id.values
        ).values

        mask_b = new_measurement_pairs.meas_id_b.isin(
            measurements_df.id.values
        ).values

        meas_mask = np.logical_and(mask_a, mask_b)

        if self._vaex_meas_pairs:
            new_measurement_pairs = new_measurement_pairs.extract()
            new_measurement_pairs['mask_a'] = mask_a
            new_measurement_pairs['mask_b'] = mask_b
            new_measurement_pairs = new_measurement_pairs[
                new_measurement_pairs.mask_a == True
                and new_measurement_pairs.mask_b == True
            ]
        else:
            new_measurement_pairs = new_measurement_pairs.loc[meas_mask]
            new_measurement_pairs = vaex.from_pandas(new_measurement_pairs)

        new_measurement_pairs['vs_int_abs'] = (
            new_measurement_pairs.vs_int.abs()
        )

        new_measurement_pairs['vs_peak_abs'] = (
            new_measurement_pairs.vs_peak.abs()
        )

        new_measurement_pairs['m_int_abs'] = (
            new_measurement_pairs.m_int.abs()
        )

        new_measurement_pairs['m_peak_abs'] = (
            new_measurement_pairs.m_peak.abs()
        )

        sources_df_two_epochs = new_measurement_pairs.groupby(
            'source_id',
            agg={
                'vs_max_int': vaex.agg.max('vs_int_abs'),
                'vs_max_peak': vaex.agg.max('vs_peak_abs'),
                'm_abs_max_int': vaex.agg.max('m_int_abs'),
                'm_abs_max_peak': vaex.agg.max('m_peak_abs'),
            }
        )

        sources_df_two_epochs = (
            sources_df_two_epochs.to_pandas_df().set_index('source_id')
        )

        # keep this here for when pipeline udpated.
        # sources_df_two_epochs = sources_df_two_epochs.fillna(value={
        #      "vs_significant_max_peak": 0.0,
        #      "m_abs_significant_max_peak": 0.0,
        #      "vs_significant_max_int": 0.0,
        #      "m_abs_significant_max_int": 0.0,
        #  })

        sources_df = sources_df.join(sources_df_two_epochs)

        del sources_df_two_epochs

        # new relation numbers
        relation_mask = np.logical_and(
            (self.relations.from_source_id.isin(sources_df.index.values)),
            (self.relations.to_source_id.isin(sources_df.index.values))
        )

        new_relations = self.relations.loc[relation_mask]

        sources_df_relations = (
            new_relations.groupby('from_source_id').agg('count')
        ).rename(columns={'to_source_id': 'n_relations'})

        sources_df = sources_df.join(sources_df_relations)

        # nearest neighbour
        sources_sky_coord = gen_skycoord_from_df(
            sources_df, ra_col='wavg_ra', dec_col='wavg_dec'
        )

        idx, d2d, _ = sources_sky_coord.match_to_catalog_sky(
            sources_sky_coord, nthneighbor=2
        )

        sources_df['n_neighbour_dist'] = d2d.degree

        # Fill the NaN values.
        sources_df = sources_df.fillna(value={
            # keep this here for when pipeline udpated.
            # "vs_significant_max_peak": 0.0,
            # "m_abs_significant_max_peak": 0.0,
            # "vs_significant_max_int": 0.0,
            # "m_abs_significant_max_int": 0.0,
            "vs_max_int": 0.0,
            "vs_max_peak": 0.0,
            "m_abs_max_int": 0.0,
            "m_abs_max_peak": 0.0,
            'n_relations': 0,
            'v_int': 0.,
            'v_peak': 0.
        }).drop([
            'interim_ew_sum', 'interim_ns_sum',
            'weight_ew_sum', 'weight_ns_sum'
        ], axis=1)

        # correct the RA wrapping
        ra_wrap_mask = sources_df['wavg_ra'] >= 360.
        sources_df.at[
            ra_wrap_mask, 'wavg_ra'
        ] = weighted_df[ra_wrap_mask].wavg_ra.values - 360.

        # Switch relations column to int
        sources_df['n_relations'] = sources_df['n_relations'].astype(int)

        return sources_df

    def _get_epoch_pair_plotting_df(
        self, df_filter, epoch_pair_id, vs_label, m_label, vs_min, m_min
    ):
        """
        Generates some standard parameters used by both two epoch plotting
        routines (bokeh and matplotlib).

        :param df_filter: Dataframe of measurement pairs with metric
            information (pre-filtered).
        :type df_filter: pandas.core.frame.DataFrame.
        :param epoch_pair_id: The epoch pair to plot.
        :type epoch_pair_id: int.
        :param vs_label: The name of the vs column to use (vs_int or vs_peak).
        :type vs_min: str
        :param m_label: The name of the m column to use (m_int or m_peak).
        :type m_min: str
        :param vs_min: The minimum Vs metric value to be considered
            a candidate.
        :type vs_min: float
        :param m_min: The minimum m metric absolute value to be
            considered a candidates.
        :type m_min: float

        :returns: Tuple of (df_filter, num_pairs, num_candidates, td_days).
        :rtype: (pd.DataFrame, int, int, float)
        """

        td_days = (
            self.pairs_df.loc[epoch_pair_id]['td'].total_seconds()
            / (3600. * 24.)
        )

        num_pairs = df_filter.shape[0]

        # convert Vs to absolute for plotting purposes.
        df_filter[vs_label] = df_filter[vs_label].abs()

        num_candidates = df_filter[
            (df_filter[vs_label] > vs_min) & (df_filter[m_label].abs() > m_min)
        ].shape[0]

        unique_meas_ids = (
            pd.unique(df_filter[['meas_id_a', 'meas_id_b']].values.ravel('K'))
        )

        temp_meas = self.measurements[
            self.measurements['id'].isin(unique_meas_ids)
        ][['id', 'forced']]

        if self._vaex_meas:
            temp_meas = temp_meas.extract().to_pandas_df()

        temp_meas = temp_meas.drop_duplicates('id').set_index('id')

        df_filter = df_filter.merge(
            temp_meas, left_on='meas_id_a', right_index=True,
            suffixes=('_a', '_b')
        )

        df_filter = df_filter.merge(
            temp_meas, left_on='meas_id_b', right_index=True,
            suffixes=('_a', '_b')
        ).rename(columns={'forced': 'forced_a'})

        df_filter['forced_sum'] = (
            df_filter[['forced_a', 'forced_b']].agg('sum', axis=1)
        ).astype(str)

        return df_filter, num_pairs, num_candidates, td_days

    def _plot_epoch_pair_bokeh(
        self,
        epoch_pair_id,
        df,
        vs_min=4.3,
        m_min=0.26,
        use_int_flux=False,
        remove_two_forced=False,
        plot_style='a'
    ) -> Model:
        '''
        Adapted from code written by Andrew O'Brien.
        Plot the results of the two epoch analysis using bokeh. Currently this
        can only plot one epoch pair at a time.

        :param epoch_pair_id: The epoch pair to plot.
        :type epoch_pair_id: int.
        :param df: Dataframe of measurement pairs with metric information.
        :type df: pandas.core.frame.DataFrame.
        :param vs_min: The minimum Vs metric value to be considered
            a candidate, defaults to 4.3.
        :type vs_min: float, optional.
        :param m_min: The minimum m metric absolute value to be
            considered a candidates, defaults to 0.26.
        :type m_min: float, optional.
        :param use_int_flux: Whether to use the integrated fluxes instead of
            the peak fluxes.
        :type use_int_flux: bool, optional.
        :param remove_two_forced: Will exclude any pairs that are both forced
            extractions if True, defaults to False.
        :type remove_two_forced: bool, optional.
        :param plot_style: Select whether to plot with style 'a' (Mooley) or
            'b' (Radcliffe). Defaults to 'a'.
        :type plot_style: str, optional.

        :returns: Bokeh figure.
        :rtype: bokeh.plotting.figure
        '''
        vs_label = 'vs_int' if use_int_flux else 'vs_peak'
        m_label = 'm_int' if use_int_flux else 'm_peak'

        df_filter, num_pairs, num_candidates, td_days = (
            self._get_epoch_pair_plotting_df(
                df, epoch_pair_id, vs_label, m_label, vs_min, m_min
            )
        )


        candidate_perc = num_candidates / num_pairs * 100.

        cmap = factor_cmap(
            'forced_sum', palette=Category10_3, factors=['0', '1', '2']
        )

        if plot_style == 'a':
            df_filter[m_label] = df_filter[m_label].abs()

            fig = figure(
                x_axis_label="m",
                y_axis_label="Vs",
                y_axis_type='log',
                title=(
                    f"{epoch_pair_id}: {td_days:.2f} days"
                    f" {num_candidates}/{num_pairs} candidates "
                    f"({candidate_perc:.2f}%)"
                ),
                tools="pan,box_select,lasso_select,box_zoom,wheel_zoom,reset",
                tooltips=[("source", "@source_id")],
            )

            range_len = 2 if remove_two_forced else 3

            for i in range(range_len):
                source = df_filter[df_filter['forced_sum'] == str(i)]
                if not source.empty:
                    fig.scatter(
                        f"{m_label}",
                        f"{vs_label}",
                        source=source,
                        color=cmap,
                        marker="circle",
                        legend_label=f"{i} forced",
                        size=2,
                        nonselection_fill_alpha=0.1,
                        nonselection_fill_color="grey",
                        nonselection_line_color=None,
                    )
            # Vertical line
            vline = Span(
                location=m_min, dimension='height', line_color='black',
                line_dash='dashed'
            )
            fig.add_layout(vline)
            # Horizontal line
            hline = Span(
                location=vs_min, dimension='width', line_color='black',
                line_dash='dashed'
            )
            fig.add_layout(hline)

            variable_region = BoxAnnotation(
                left=m_min,
                bottom=vs_min,
                fill_color="orange",
                fill_alpha=0.3,
                level="underlay",
            )
            fig.add_layout(variable_region)
            fig.legend.location = "bottom_right"

        else:

            fig = figure(
                x_axis_label="Vs",
                y_axis_label="m",
                title=(
                    f"{epoch_pair_id}: {td_days:.2f} days"
                    f" {num_candidates}/{num_pairs} candidates "
                    f"({candidate_perc:.2f}%)"
                ),
                tools="pan,box_select,lasso_select,box_zoom,wheel_zoom,reset",
                tooltips=[("source", "@source_id")],
            )

            range_len = 2 if remove_two_forced else 3

            for i in range(range_len):
                source = df_filter[df_filter['forced_sum'] == str(i)]
                if not source.empty:
                    fig.scatter(
                        f"{vs_label}",
                        f"{m_label}",
                        source=source,
                        color=cmap,
                        marker="circle",
                        legend_label=f"{i} forced",
                        size=2,
                        nonselection_fill_alpha=0.1,
                        nonselection_fill_color="grey",
                        nonselection_line_color=None,
                    )

            variable_region_1 = BoxAnnotation(
                left=vs_min, bottom=m_min,
                fill_color="orange", level="underlay"
            )
            variable_region_2 = BoxAnnotation(
                left=vs_min, top=-m_min, fill_color="orange", level="underlay"
            )
            fig.add_layout(variable_region_1)
            fig.add_layout(variable_region_2)

            fig.legend.location = "top_right"

        fig.legend.click_policy = "hide"

        return fig

    def _plot_epoch_pair_matplotlib(
        self,
        epoch_pair_id,
        df,
        vs_min=4.3,
        m_min=0.26,
        use_int_flux=False,
        remove_two_forced=False,
        plot_style='a'
    ):
        """
        Plot the results of the two epoch analysis using matplotlib. Currently
        this can only plot one epoch pair at a time.

        :param epoch_pair_id: The epoch pair to plot.
        :type epoch_pair_id: int.
        :param df: Dataframe of measurement pairs with metric information.
        :type df: pandas.core.frame.DataFrame OR vaex.dataframe.DataFrame.
        :param vs_min: The minimum Vs metric value to be considered
            a candidate, defaults to 4.3.
        :type vs_min: float, optional.
        :param m_min: The minimum m metric absolute value to be
            considered a candidates, defaults to 0.26.
        :type m_min: float, optional.
        :param use_int_flux: Whether to use the integrated fluxes instead of
            the peak fluxes.
        :type use_int_flux: bool, optional.
        :param remove_two_forced: Will exclude any pairs that are both forced
            extractions if True, defaults tto False.
        :type remove_two_forced: bool, optional.
        :param plot_style: Select whether to plot with style 'a' (Mooley) or
            'b' (Radcliffe). Defaults to 'a'.
        :type plot_style: str, optional.

        :returns: matplotlib pyplot figure.
        :rtype: matplotlib.pyplot.figure
        """
        plt.close()  # close any previous ones

        vs_label = 'vs_int' if use_int_flux else 'vs_peak'
        m_label = 'm_int' if use_int_flux else 'm_peak'

        df_filter, num_pairs, num_candidates, td_days = (
            self._get_epoch_pair_plotting_df(
                df, epoch_pair_id, vs_label, m_label, vs_min, m_min
            )
        )

        candidate_perc = num_candidates / num_pairs * 100.

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)

        colors = ["C0", "C1", "C2"]
        labels = ["0 forced", "1 forced", "2 forced"]

        range_len = 2 if remove_two_forced else 3

        if plot_style == 'a':
            for i in range(range_len):
                mask = df_filter['forced_sum'] == str(i)
                if np.any(mask):
                    ax.scatter(
                        df_filter[mask][m_label].abs(),
                        df_filter[mask][vs_label],
                        c=colors[i], label=labels[i],
                        zorder=2
                    )

            ax.axhline(vs_min, ls="--", c='k', zorder=5)
            ax.axvline(m_min, ls="--", c='k', zorder=5)
            ax.set_yscale('log')

            y_limits = ax.get_ylim()
            x_limits = ax.get_xlim()

            ax.fill_between(
                [m_min, 1e5], vs_min, 1e5,
                color='navajowhite', alpha=0.5, zorder=1
            )

            ax.set_xlim(x_limits)
            ax.set_ylim(y_limits)

            ax.set_xlabel(r"|$m$|")
            ax.set_ylabel(r"$V_{s}$")

        else:
            ax.fill_between([vs_min, 100], m_min, 4.2, color="gold", alpha=0.3)
            ax.fill_between(
                [vs_min, 100], -4.2, m_min * -1, color="gold", alpha=0.3
            )

            for i in range(range_len):
                mask = df_filter['forced_sum'] == str(i)
                if np.any(mask):
                    ax.scatter(
                        df_filter[mask][vs_label], df_filter[mask][m_label],
                        c=colors[i], label=labels[i]
                    )
            ax.set_xlim(0.5, 50)
            ax.set_ylim(-4.0, 4.0)

            ax.set_ylabel(r"$m$")
            ax.set_xlabel(r"$V_{s}$")

        date_string = "Epoch {} (Time {:.2f} days)".format(
            epoch_pair_id, td_days
        )
        number_string = "Candidates: {}/{} ({:.2f} %)".format(
            num_candidates, num_pairs, (100.*num_candidates/num_pairs)
        )
        ax.text(
            0.6, 0.05, date_string + '\n' + number_string,
            transform=ax.transAxes
        )
        ax.legend()

        return fig

    def plot_two_epoch_pairs(
        self,
        epoch_pair_id,
        query=None,
        df=None,
        vs_min=4.3,
        m_min=0.26,
        use_int_flux=False,
        remove_two_forced=False,
        plot_type='bokeh',
        plot_style='a'
    ):
        """
        Adapted from code written by Andrew O'Brien.
        Plot the results of the two epoch analysis. Currently this can only
        plot one epoch pair at a time.

        :param epoch_pair_id: The epoch pair to plot.
        :type epoch_pair_id: int.
        :param query: String query to apply to the dataframe before
            the analysis is run, defaults to None.
        :type query: str, optional.
        :param df: Dataframe of sources from the pipeline run, defaults
            to None. If None then the sources from the PipeAnalysis object
            are used.
        :type df: pandas.core.frame.DataFrame, optional.
        :param vs_min: The minimum Vs metric value to be considered
            a candidate, defaults to 4.3.
        :type vs_min: float, optional.
        :param m_min: The minimum m metric absolute value to be
            considered a candidates, defaults to 0.26.
        :type m_min: float, optional.
        :param use_int_flux: Whether to use the integrated fluxes instead of
            the peak fluxes.
        :type use_int_flux: bool, optional.
        :param remove_two_forced: Will exclude any pairs that are both forced
            extractions if True, defaults tto False.
        :type remove_two_forced: bool, optional.
        :param plot_type: Selects whether the returned plot is 'bokeh' or
            'matplotlib', defaults to 'bokeh'.
        :type plot_type: str, optional.
        :param plot_style: Select whether to plot with style 'a' (Mooley) or
            'b' (Radcliffe). Defaults to 'a'.
        :type plot_style: str, optional.

        :returns: Bokeh or matplotlib figure.
        :rtype: bokeh.plotting.figure or matplotlib.pyplot.figure
        """
        if not self._loaded_two_epoch_metrics:
            raise Exception(
                "The two epoch metrics must first be loaded to use the"
                " plotting function. Please do so with the command:\n"
                "'mypiperun.load_two_epoch_metrics()'\n"
                "and try again."
            )

        if plot_type not in ['bokeh', 'matplotlib']:
            raise Exception(
                "'plot_type' value is not recongised!"
                " Must be either 'bokeh' or 'matplotlib'."
            )

        if plot_style not in ['a', 'b']:
            raise Exception(
                "'plot_style' value is not recongised!"
                " Must be either 'a' for Mooley or 'b' for Radcliffe."
            )

        if epoch_pair_id not in self.pairs_df.index.values:
            raise Exception(f"Pair with ID '{epoch_pair_id}' does not exist!")

        if df is None:
            df = self.sources

        if query is not None:
            df = df.query(query)

        pair_epoch_key = self.pairs_df.loc[epoch_pair_id]['pair_epoch_key']

        pairs_df = (
            self.measurement_pairs_df[
                self.measurement_pairs_df.pair_epoch_key == pair_epoch_key
            ]
        )

        if self._vaex_meas_pairs:
            pairs_df = pairs_df.extract().to_pandas_df()

        pairs_df = pairs_df[pairs_df['source_id'].isin(df.index.values)]

        if plot_type == 'bokeh':
            fig = self._plot_epoch_pair_bokeh(
                epoch_pair_id,
                pairs_df,
                vs_min,
                m_min,
                use_int_flux,
                remove_two_forced,
                plot_style
            )
        else:
            fig = self._plot_epoch_pair_matplotlib(
                epoch_pair_id,
                pairs_df,
                vs_min,
                m_min,
                use_int_flux,
                remove_two_forced,
                plot_style
            )

        return fig

    def run_two_epoch_analysis(
        self, v, m, query=None, df=None, use_int_flux=False
    ):
        '''
        Run the two epoch analysis on the pipeline run, with optional
        inputs to use a query or filtered dataframe.

        :param v: The minimum Vs metric value to be considered
            a candidate.
        :type v: float.
        :param m: The minimum m metric absolute value to be
            considered a candidates.
        :type m: float.
        :param query: String query to apply to the dataframe before
            the analysis is run, defaults to None.
        :type query: str, optional.
        :param df: Dataframe of sources from the pipeline run, defaults
            to None. If None then the sources from the PipeAnalysis object
            are used.
        :type df: pandas.core.frame.DataFrame, optional.
        :param use_int_flux: Use integrated fluxes for the analysis instead of
            peak fluxes, defaults to 'False'.
        :type use_int_flux: bool, optional.

        :returns: Tuple containing two dataframes of the candidate sources
            and pairs.
        :rtype: (pandas.core.frame.DataFrame, pandas.core.frame.DataFrame)
        '''
        if not self._loaded_two_epoch_metrics:
            raise Exception(
                "The two epoch metrics must first be loaded to use the"
                " plotting function. Please do so with the command:\n"
                "'mypiperun.load_two_epoch_metrics()'\n"
                "and try again."
            )

        if df is None:
            df = self.sources

        if query is not None:
            df = df.query(query)

        allowed_sources = df.index.values

        pairs_df = self.measurement_pairs_df.copy()

        if len(allowed_sources) != self.sources.shape[0]:
            if self._vaex_meas_pairs:
                pairs_df = pairs_df[
                    pairs_df['source_id'].isin(allowed_sources)
                ]
            else:
                pairs_df = pairs_df.loc[
                    pairs_df['source_id'].isin(allowed_sources)
                ]

        vs_label = 'vs_int' if use_int_flux else 'vs_peak'
        m_abs_label = 'm_int' if use_int_flux else 'm_peak'

        pairs_df[vs_label] = pairs_df[vs_label].abs()

        # If vaex convert these to pandas
        if self._vaex_meas_pairs:
            candidate_pairs = pairs_df[
                (pairs_df[vs_label] > v) & (pairs_df[m_abs_label] > m)
            ]

            candidate_pairs = candidate_pairs.to_pandas_df()

            candidate_sources = (
                candidate_pairs[['source_id']].extract().to_pandas_df()
            )

            unique_sources = candidate_sources['source_id'].unique()
        else:
            candidate_pairs = pairs_df.loc[
                (pairs_df[vs_label] > v) & (pairs_df[m_abs_label] > m)
            ]

            unique_sources = candidate_pairs['source_id'].unique()

        candidate_sources = self.sources.loc[unique_sources]

        return candidate_sources, candidate_pairs

    def _fit_eta_v(self, df, use_int_flux=False):
        '''
        Fits the eta and v distributions with Gaussians. Used from
        within the 'run_eta_v_analysis' method.

        :param df: DataFrame containing the sources from the pipeline run.
        :type df: pandas.core.frame.DataFrame.
        :param use_int_flux: Use integrated fluxes for the analysis instead of
            peak fluxes, defaults to 'False'.
        :type pairs: bool, optional.

        :returns: tuple containing the eta_fit_mean, eta_fit_sigma, v_fit_mean
            and the v_fit_sigma
        :rtype: float, float, float, float
        '''

        if use_int_flux:
            eta_label = 'eta_int'
            v_label = 'v_int'
        else:
            eta_label = 'eta_peak'
            v_label = 'v_peak'

        eta_log = np.log10(df[eta_label])
        v_log = np.log10(df[v_label])

        eta_log_clipped = sigma_clip(
            eta_log, masked=False, stdfunc=mad_std, sigma=3
        )
        v_log_clipped = sigma_clip(
            v_log, masked=False, stdfunc=mad_std, sigma=3
        )

        eta_fit_mean, eta_fit_sigma = norm.fit(eta_log_clipped)
        v_fit_mean, v_fit_sigma = norm.fit(v_log_clipped)

        return (eta_fit_mean, eta_fit_sigma, v_fit_mean, v_fit_sigma)

    def _gaussian_fit(self, data, param_mean, param_sigma):
        '''
        Returns the Guassian to add to the matplotlib plot.

        :param data: Series object containing the log10 values of the
            distribution to plot.
        :type data: pandas.core.frame.Series.
        :param param_mean: The calculated mean of the Gaussian to fit.
        :type param_mean: float.
        :param param_sigma: The calculated sigma of the Gaussian to fit.
        :type param_sigma: float.

        :returns: tuple containing the range of the returned data and the
            Gaussian fit.
        :rtype: numpy.ndarray, scipy.stats.norm
        '''
        range_data = np.linspace(min(data), max(data), 1000)
        fit = norm.pdf(range_data, loc=param_mean, scale=param_sigma)

        return range_data, fit

    def _make_bins(self, x):
        '''
        Calculates the bins that should be used for the v, eta distribution
        using bayesian blocks.

        :param x: Series object containing the log10 values of the
            distribution to plot.
        :type data: pandas.core.frame.Series.

        :returns: bins to apply.
        :rtype: list.
        '''
        new_bins = density_estimation.bayesian_blocks(x)
        binsx = [
            new_bins[a] for a in range(
                len(new_bins) - 1
            ) if abs((new_bins[a + 1] - new_bins[a]) / new_bins[a]) > 0.05
        ]
        binsx = binsx + [new_bins[-1]]

        return binsx

    def eta_v_diagnostic_plot(
        self, df, eta_cutoff, v_cutoff, use_int_flux=False
    ):
        '''
        Adapted from code written by Antonia Rowlinson.
        Produces the eta, V 'diagnostic plot'
        (see Rowlinson et al., 2018,
        https://ui.adsabs.harvard.edu/abs/2019A%26C....27..111R/abstract).

        :param df: Dataframe containing the sources from the Pipeline run.
        :type df: pandas.core.frame.DataFrame.
        :param eta_cutoff: The log10 eta_cutoff from the analysis.
        :type eta_cutoff: float.
        :param v_cutoff: The log10 v_cutoff from the analysis.
        :type v_cutoff: float.
        :param use_int_flux: Use integrated fluxes for the analysis instead of
            peak fluxes, defaults to 'False'.
        :type pairs: bool, optional.

        :returns: matplotlib figure containing plot.
        :rtype: matplotlib.pyplot.figure.
        '''
        plt.close()  # close any previous ones

        if use_int_flux:
            eta_label = 'eta_int'
            v_label = 'v_int'
        else:
            eta_label = 'eta_peak'
            v_label = 'v_peak'

        eta_cutoff = np.log10(eta_cutoff)
        v_cutoff = np.log10(v_cutoff)

        nullfmt = NullFormatter()  # no labels

        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
        fontP = FontProperties()
        fontP.set_size('large')
        fig.subplots_adjust(hspace=.001, wspace=0.001)
        ax1.set_ylabel(r'$\eta_\nu$', fontsize=28)
        ax3.set_ylabel(r'$V_\nu$', fontsize=28)
        ax3.set_xlabel('Max Flux (Jy)', fontsize=24)
        ax4.set_xlabel('Max Flux / Median Flux', fontsize=24)

        xdata_ax3 = df['max_flux_peak']
        xdata_ax4 = df['max_flux_peak'] / df['avg_flux_peak']
        ydata_ax1 = df[eta_label]
        ydata_ax3 = df[v_label]
        ax1.scatter(xdata_ax3, ydata_ax1, s=10., zorder=5)
        ax2.scatter(xdata_ax4, ydata_ax1, s=10., zorder=6)
        ax3.scatter(xdata_ax3, ydata_ax3, s=10., zorder=7)
        ax4.scatter(xdata_ax4, ydata_ax3, s=10., zorder=8)

        Xax3 = df['max_flux_peak']
        Xax4 = df['max_flux_peak'] / df['avg_flux_peak']
        Yax1 = df[eta_label]
        Yax3 = df[v_label]

        if eta_cutoff != 0 or v_cutoff != 0:
            ax1.axhline(
                y=10.**eta_cutoff, linewidth=2, color='k', linestyle='--'
            )
            ax2.axhline(
                y=10.**eta_cutoff, linewidth=2, color='k', linestyle='--'
            )
            ax3.axhline(
                y=10.**v_cutoff, linewidth=2, color='k', linestyle='--'
            )
            ax4.axhline(
                y=10.**v_cutoff, linewidth=2, color='k', linestyle='--'
            )

        ax1.set_yscale('log')
        ax1.set_xscale('log')
        ax2.set_yscale('log')
        ax3.set_yscale('log')
        ax3.set_xscale('log')
        ax4.set_yscale('log')
        xmin_ax3 = 10.**(int(np.log10(min(Xax3)) - 1.1))
        xmax_ax3 = 10.**(int(np.log10(max(Xax3)) + 1.2))
        xmin_ax4 = 0.8
        xmax_ax4 = int(max(xdata_ax4) + 0.5)
        ymin_ax1 = 10.**(int(np.log10(min(Yax1)) - 1.1))
        ymax_ax1 = 10.**(int(np.log10(max(Yax1)) + 1.2))
        ymin_ax3 = 10.**(int(np.log10(min(Yax3)) - 1.1))
        ymax_ax3 = 10.**(int(np.log10(max(Yax3)) + 1.2))
        ax1.set_ylim(ymin_ax1, ymax_ax1)
        ax3.set_ylim(ymin_ax3, ymax_ax3)
        ax3.set_xlim(xmin_ax3, xmax_ax3)
        ax4.set_xlim(xmin_ax4, xmax_ax4)
        ax1.set_xlim(ax3.get_xlim())
        ax4.set_ylim(ax3.get_ylim())
        ax2.set_xlim(ax4.get_xlim())
        ax2.set_ylim(ax1.get_ylim())
        ax1.xaxis.set_major_formatter(nullfmt)
        ax4.yaxis.set_major_formatter(nullfmt)
        ax2.xaxis.set_major_formatter(nullfmt)
        ax2.yaxis.set_major_formatter(nullfmt)

        return fig

    def _plot_eta_v_matplotlib(
        self, df, eta_fit_mean, eta_fit_sigma,
        v_fit_mean, v_fit_sigma, eta_cutoff, v_cutoff,
        use_int_flux=False
    ):
        '''
        Adapted from code written by Antonia Rowlinson.
        Produces the eta, V candidates plot
        (see Rowlinson et al., 2018,
        https://ui.adsabs.harvard.edu/abs/2019A%26C....27..111R/abstract).
        Returns a matplotlib version.

        :param df: Dataframe containing the sources from the pipeline
            run.
        :type df: pandas.core.frame.DataFrame.
        :param eta_fit_mean: The mean of the eta fitted Gaussian.
        :type eta_fit_mean: float.
        :param eta_fit_sigma: The sigma of the eta fitted Gaussian.
        :type eta_fit_sigma: float.
        :param v_fit_mean: The mean of the v fitted Gaussian.
        :type v_fit_mean: float.
        :param v_fit_sigma: The sigma of the v fitted Gaussian.
        :type v_fit_sigma: float.
        :param eta_cutoff: The log10 eta_cutoff from the analysis.
        :type eta_cutoff: float.
        :param v_cutoff: The log10 v_cutoff from the analysis.
        :type v_cutoff: float.
        :param use_int_flux: Use integrated fluxes for the analysis instead of
            peak fluxes, defaults to 'False'.
        :type pairs: bool, optional.

        :returns: matplotlib figure containing plot.
        :rtype: matplotlib.pyplot.figure.
        '''
        plt.close()  # close any previous ones
        if use_int_flux:
            x_label = 'eta_int'
            y_label = 'v_int'
            title = "Int. Flux"
        else:
            x_label = 'eta_peak'
            y_label = 'v_peak'
            title = 'Peak Flux'

        eta_cutoff = np.log10(eta_cutoff)
        v_cutoff = np.log10(v_cutoff)

        nullfmt = NullFormatter()  # no labels
        fontP = FontProperties()
        fontP.set_size('large')
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        bottom_h = left_h = left+width+0.02
        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom_h, width, 0.2]
        rect_histy = [left_h, bottom, 0.2, height]
        fig = plt.figure(figsize=(12, 12))
        axScatter = fig.add_subplot(223, position=rect_scatter)
        plt.xlabel(r'$\eta_{\nu}$', fontsize=28)
        plt.ylabel(r'$V_{\nu}$', fontsize=28)
        axHistx = fig.add_subplot(221, position=rect_histx)
        axHisty = fig.add_subplot(224, position=rect_histy)
        axHistx.xaxis.set_major_formatter(nullfmt)
        axHisty.yaxis.set_major_formatter(nullfmt)
        axHistx.axes.yaxis.set_ticklabels([])
        axHisty.axes.xaxis.set_ticklabels([])

        xdata_var = np.log10(df[x_label])
        ydata_var = np.log10(df[y_label])
        axScatter.scatter(xdata_var, ydata_var, s=10., zorder=5, color='C0')
        axScatter.fill_between(
            [eta_cutoff, 1e4], v_cutoff, 1e4,
            color='navajowhite', alpha=0.5
        )

        x = np.log10(df[x_label])
        y = np.log10(df[y_label])

        axHistx.hist(
            x, bins=self._make_bins(x), density=1,
            histtype='stepfilled', color='C0'
        )
        axHisty.hist(
            y, bins=self._make_bins(y), density=1,
            histtype='stepfilled', orientation='horizontal', color='C0'
        )

        xmin = int(min(x) - 1.1)
        xmax = int(max(x) + 1.1)
        ymin = int(min(y) - 1.1)
        ymax = int(max(y) + 1.1)
        xvals = range(xmin, xmax)
        xtxts = [r'$10^{'+str(a)+'}$' for a in xvals]
        yvals = range(ymin, ymax)
        ytxts = [r'$10^{' + str(a) + '}$' for a in yvals]
        axScatter.set_xlim([xmin, xmax])
        axScatter.set_ylim([ymin, ymax])
        axScatter.set_xticks(xvals)
        axScatter.set_xticklabels(xtxts, fontsize=20)
        axScatter.set_yticks(yvals)
        axScatter.set_yticklabels(ytxts, fontsize=20)
        axHistx.set_xlim(axScatter.get_xlim())
        axHisty.set_ylim(axScatter.get_ylim())

        if eta_cutoff != 0 or v_cutoff != 0:
            axHistx.axvline(
                x=eta_cutoff, linewidth=2, color='k', linestyle='--'
            )
            axHisty.axhline(
                y=v_cutoff, linewidth=2, color='k', linestyle='--'
            )
            axScatter.axhline(
                y=v_cutoff, linewidth=2, color='k', linestyle='--'
            )
            axScatter.axvline(
                x=eta_cutoff, linewidth=2, color='k', linestyle='--'
            )

        range_x, fitx = self._gaussian_fit(x, eta_fit_mean, eta_fit_sigma)
        axHistx.plot(range_x, fitx, 'k:', linewidth=2)
        range_y, fity = self._gaussian_fit(y, v_fit_mean, v_fit_sigma)
        axHisty.plot(fity, range_y, 'k:', linewidth=2)

        return fig

    def _plot_eta_v_bokeh(
        self, df, eta_fit_mean, eta_fit_sigma,
        v_fit_mean, v_fit_sigma, eta_cutoff, v_cutoff,
        use_int_flux=False
    ):
        '''
        Adapted from code written by Andrew O'Brien.
        Produces the eta, V candidates plot
        (see Rowlinson et al., 2018,
        https://ui.adsabs.harvard.edu/abs/2019A%26C....27..111R/abstract).
        Returns a bokeh version.

        :param df: Dataframe containing the sources from the pipeline
            run.
        :type df: pandas.core.frame.DataFrame.
        :param eta_fit_mean: The mean of the eta fitted Gaussian.
        :type eta_fit_mean: float.
        :param eta_fit_sigma: The sigma of the eta fitted Gaussian.
        :type eta_fit_sigma: float.
        :param v_fit_mean: The mean of the v fitted Gaussian.
        :type v_fit_mean: float.
        :param v_fit_sigma: The sigma of the v fitted Gaussian.
        :type v_fit_sigma: float.
        :param eta_cutoff: The log10 eta_cutoff from the analysis.
        :type eta_cutoff: float.
        :param v_cutoff: The log10 v_cutoff from the analysis.
        :type v_cutoff: float.
        :param use_int_flux: Use integrated fluxes for the analysis instead of
            peak fluxes, defaults to 'False'.
        :type pairs: bool, optional.

        :returns: bokeh grid object containing figure.
        :rtype: bokeh.models.grids.Grid
        '''
        # generate fitted curve data for plotting
        eta_x = np.linspace(
            norm.ppf(0.001, loc=eta_fit_mean, scale=eta_fit_sigma),
            norm.ppf(0.999, loc=eta_fit_mean, scale=eta_fit_sigma),
        )
        eta_y = norm.pdf(eta_x, loc=eta_fit_mean, scale=eta_fit_sigma)

        v_x = np.linspace(
            norm.ppf(0.001, loc=v_fit_mean, scale=v_fit_sigma),
            norm.ppf(0.999, loc=v_fit_mean, scale=v_fit_sigma),
        )
        v_y = norm.pdf(v_x, loc=v_fit_mean, scale=v_fit_sigma)

        PLOT_WIDTH = 700
        PLOT_HEIGHT = PLOT_WIDTH
        fig = figure(
            plot_width=PLOT_WIDTH,
            plot_height=PLOT_HEIGHT,
            aspect_scale=1,
            x_axis_type="log",
            y_axis_type="log",
            x_axis_label="eta",
            y_axis_label="V",
            tooltips=[("source", "@id")],
        )
        cmap = linear_cmap(
            "n_selavy",
            cc.kb,
            df["n_selavy"].min(),
            df["n_selavy"].max(),
        )

        if use_int_flux:
            x_label = 'eta_int'
            y_label = 'v_int'
            title = "Int. Flux"
        else:
            x_label = 'eta_peak'
            y_label = 'v_peak'
            title = 'Peak Flux'

        fig.scatter(
            x=x_label, y=y_label, color=cmap,
            marker="circle", size=5, source=df
        )

        # axis histograms
        # filter out any forced-phot points for these
        x_hist = figure(
            plot_width=PLOT_WIDTH,
            plot_height=100,
            x_range=fig.x_range,
            y_axis_type=None,
            x_axis_type="log",
            x_axis_location="above",
            title="VAST eta-V {}".format(title),
            tools="",
        )
        x_hist_data, x_hist_edges = np.histogram(
            np.log10(df["eta_peak"]), density=True, bins=50,
        )
        x_hist.quad(
            top=x_hist_data,
            bottom=0,
            left=10 ** x_hist_edges[:-1],
            right=10 ** x_hist_edges[1:],
        )
        x_hist.line(10 ** eta_x, eta_y, color="black")
        x_hist_sigma_span = Span(
            location=eta_cutoff,
            dimension="height",
            line_color="black",
            line_dash="dashed",
        )
        x_hist.add_layout(x_hist_sigma_span)
        fig.add_layout(x_hist_sigma_span)

        y_hist = figure(
            plot_height=PLOT_HEIGHT,
            plot_width=100,
            y_range=fig.y_range,
            x_axis_type=None,
            y_axis_type="log",
            y_axis_location="right",
            tools="",
        )
        y_hist_data, y_hist_edges = np.histogram(
            np.log10(df["v_peak"]), density=True, bins=50,
        )
        y_hist.quad(
            right=y_hist_data,
            left=0,
            top=10 ** y_hist_edges[:-1],
            bottom=10 ** y_hist_edges[1:],
        )
        y_hist.line(v_y, 10 ** v_x, color="black")
        y_hist_sigma_span = Span(
            location=v_cutoff,
            dimension="width",
            line_color="black",
            line_dash="dashed",
        )
        y_hist.add_layout(y_hist_sigma_span)
        fig.add_layout(y_hist_sigma_span)

        variable_region = BoxAnnotation(
            left=eta_cutoff,
            bottom=v_cutoff,
            fill_color="orange",
            fill_alpha=0.3,
            level="underlay",
        )
        fig.add_layout(variable_region)
        grid = gridplot(
            [[x_hist, Spacer(width=100, height=100)], [fig, y_hist]]
        )
        grid.css_classes.append("mx-auto")

        return grid

    def run_eta_v_analysis(
        self, eta_sigma, v_sigma,
        query=None, df=None, use_int_flux=False,
        plot_type='bokeh', diagnostic=False
    ):
        '''
        Run the eta, v analysis on the pipeline run, with optional
        inputs to use a query or filtered dataframe (see Rowlinson
        et al., 2018,
        https://ui.adsabs.harvard.edu/abs/2019A%26C....27..111R/abstract).

        :param eta_sigma: The minimum sigma value of the eta distribution
            to be used as a threshold.
        :type eta_sigma: float.
        :param v_sigma: The minimum sigma value of the v distribution
            to be used as a threshold.
        :type v_sigma: float.
        :param query: String query to apply to the dataframe before
            the analysis is run, defaults to None.
        :type query: str, optional.
        :param df: Dataframe of sources from the pipeline run, defaults
            to None. If None then the sources from the PipeAnalysis object
            are used.
        :type df: pandas.core.frame.DataFrame, optional.
        :param use_int_flux: Use integrated fluxes for the analysis instead of
            peak fluxes, defaults to 'False'.
        :type pairs: bool, optional.
        :param plot_type: Select which format the candidates plot should be
            returned in. Either 'bokeh' or 'matplotlib', defaults to 'bokeh'.
        :type plot_type: str, optional.
        :param diagnostic: When 'True' the diagnostic plot is also returned,
            defaults to 'False'.
        :type diagnostic: bool, optional.

        :returns: tuple containing the eta cutoff value, the v cutoff value,
            dataframe of candidates, candidates plot and, if selected, the
            diagnostic plot.
        :rtype: float, float, pandas.core.frame.DataFrame,
            (bokeh.models.grids.Grid or matplotlib.pyplot.figure),
            matplotlib.pyplot.figure

        '''
        plot_types = ['bokeh', 'matplotlib']

        if plot_type not in plot_types:
            raise Exception(
                "Not a valid plot type!"
                " Must be 'bokeh' or 'matplotlib'."
            )

        if df is None:
            df = self.sources

        if query is not None:
            df = df.query(query)

        (
            eta_fit_mean, eta_fit_sigma,
            v_fit_mean, v_fit_sigma
        ) = self._fit_eta_v(df, use_int_flux=use_int_flux)

        v_cutoff = 10 ** (v_fit_mean + v_sigma * v_fit_sigma)
        eta_cutoff = 10 ** (eta_fit_mean + eta_sigma * eta_fit_sigma)

        if plot_type == 'bokeh':
            plot = self._plot_eta_v_bokeh(
                df, eta_fit_mean, eta_fit_sigma,
                v_fit_mean, v_fit_sigma, eta_cutoff, v_cutoff,
                use_int_flux=use_int_flux
            )
        else:
            plot = self._plot_eta_v_matplotlib(
                df, eta_fit_mean, eta_fit_sigma,
                v_fit_mean, v_fit_sigma, eta_cutoff, v_cutoff,
                use_int_flux=use_int_flux
            )

        if use_int_flux:
            label = 'int'
        else:
            label = 'peak'

        candidates = df.query(
            "v_{0} > {1} "
            "& eta_{0} > {2}".format(
                label,
                v_cutoff,
                eta_cutoff
            )
        )

        if diagnostic:
            diag = self.eta_v_diagnostic_plot(
                df, eta_cutoff, v_cutoff
            )
            return eta_cutoff, v_cutoff, candidates, plot, diag
        else:
            return eta_cutoff, v_cutoff, candidates, plot
