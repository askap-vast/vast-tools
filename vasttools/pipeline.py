import pandas as pd
import os
import warnings
import glob
import gc
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
import numpy as np
import pandas as pd
from scipy.stats import norm
import scipy.ndimage as ndi
from astropy.stats import sigma_clip, mad_std
from astropy.coordinates import SkyCoord
from astropy import units as u
from mocpy import MOC
from vasttools.source import Source
from vasttools.utils import match_planet_to_field
from vasttools.survey import Image
from multiprocessing import cpu_count
from datetime import timedelta
from itertools import combinations
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from matplotlib.font_manager import FontProperties
from astroML import density_estimation


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

        try:
            pipeline_run_path = os.getenv(
                'PIPELINE_WORKING_DIR',
                os.path.abspath(str(project_dir))
            )
        except Exception as e:
            raise Exception(
                "The pipeline run directory could not be determined!"
                " Either the system environment 'PIPELINE_WORKING_DIR'"
                " must be defined or the 'project_dir' argument defined"
                " when initialising the pipeline class object."
            )

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

        associations = pd.read_parquet(
            os.path.join(
                run_dir,
                'associations.parquet'
            ),
            engine='pyarrow'
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
        ).set_index('id')

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

        measurements = measurements.merge(
            associations, left_on='id', right_on='meas_id',
            how='left'
        ).drop([
            'meas_id',
        ], axis=1).rename(
            columns={
                'source_id': 'source',
            }
        ).dropna(subset=['source'])

        measurements = measurements.merge(
            images[[
                'id',
                'path',
                'noise_path',
                'measurements_path',
                'frequency'
            ]], how='left',
            left_on='image_id',
            right_on='id'
        ).drop(
            'id_y',
            axis=1
        ).rename(
            columns={
                'id_x': 'id',
                'path': 'image',
                'noise_path': 'rms',
                'measurements_path': 'selavy'
            }
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

        images = images.set_index('id')

        piperun = PipeAnalysis(
            name=run_name,
            images=images,
            skyregions=skyregions,
            relations=relations,
            sources=sources,
            measurements=measurements,
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
    measurements : pandas.core.frame.DataFrame
        Dataframe containing all the information on the measurements
        of the pipeline run.
    relations : pandas.core.frame.DataFrame
        Dataframe containing all the information on the relations
        of the pipeline run.
    n_workers : pandas.core.frame.DataFrame
        Number of workers (cpus) available.


    Methods
    -------

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
    '''
    def __init__(
        self, name, images,
        skyregions, relations, sources,
        measurements, n_workers=cpu_count() - 1
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
        self.measurements = measurements
        self.relations = relations
        self.n_workers = n_workers

    def get_source(self, id, field=None, stokes='I', outdir='.'):
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

        measurements = self.measurements.groupby(
            'source'
        ).get_group(id)

        measurements = measurements.rename(
            columns={
              'time': 'dateobs',
            }
        ).sort_values(
            by='dateobs'
        ).reset_index(drop=True)

        s = self.sources.loc[id]

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
        source_epochs = range(
            1, num_measurements + 1
        )
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

        if not ignore_large_run_warning and images_to_use.shape[0] > 20:
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

        del sources_to_add

        if new_name is not None:
            self.name = new_name

        return self


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
    measurements : pandas.core.frame.DataFrame
        Dataframe containing all the information on the measurements
        of the pipeline run.
    relations : pandas.core.frame.DataFrame
        Dataframe containing all the information on the relations
        of the pipeline run.
    n_workers : pandas.core.frame.DataFrame
        Number of workers (cpus) available.


    Methods
    -------

    get_source(id, field=None, stokes='I', outdir='.')
        Creates a vasttools.source.Source object for the requested
        source.

    check_for_planets()
        Searches the pipeline run images for any planets present.
        Returns pandas dataframe with results.

    run_two_epoch_analysis(v, m, query=None, df=None, use_int_flux=False)
        Runs the two epoch variability analysis on the pipeline run. Filters
        can be applied using query argument or directly passing the filtered
        sources df. Returns pairs dataframe, two epoch metrics dataframe, a
        dataframe of candidates given the input v and m values, and a bokeh
        plot of the metrics.

    plot_epoch_pairs_bokeh(df, pairs, vs_min=4.3, m_min=0.26)
        Creates the bokeh plot of the two epoch analysis. It is run as part
        of `run_two_epoch_analysis` or can be called separately.

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
        self, name, images,
        skyregions, relations, sources,
        measurements, n_workers=cpu_count() - 1
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
        :param relations: Relations dataframe from the pipeline run
           loaded from relations.parquet.
        :type relations: pandas.core.frame.DataFrame
        :param n_workers: Number of workers (cpus) available.
        :type n_workers: int, optional
        '''
        super().__init__(
            name, images,
            skyregions, relations, sources,
            measurements, n_workers
        )

    def _get_two_epoch_df(self, allowed_sources=[]):
        '''
        Workhorse function of the two epoch analysis which
        calculates the unique pairs of images and the metrics
        between each source datapoint for each pair.

        :param allowed_sources: List of the allowed source ids in
            the analysis.
        :type allowed_sources: list, optional

        :returns: Tuple containing the pairs dataframe and the
            two epoch dataframe containg the results of each pair
            of source measurements.
        :rtype: pandas.core.frame.DataFrame, pandas.core.frame.DataFrame
        '''
        image_ids = self.images.sort_values(by='datetime').index.tolist()

        if len(allowed_sources) > 0:
            measurements = self.measurements.loc[
                self.measurements['source'].isin(
                    allowed_sources
                )
            ]
        else:
            measurements = self.measurements

        pairs_df = pd.DataFrame.from_dict(
            {'pair': combinations(image_ids, 2)}
        )

        pairs_df = (
            pd.DataFrame(pairs_df['pair'].tolist())
            .rename(columns={0: 'image_id_x', 1: 'image_id_y'})
            .merge(
                self.images[['datetime']],
                left_on='image_id_x', right_index=True
            )
            .merge(
                self.images[['datetime']],
                left_on='image_id_y', right_index=True
            )
        ).reset_index().rename(columns={'index': 'id'})

        pairs_df['td'] = pairs_df['datetime_y'] - pairs_df['datetime_x']

        pairs_df.drop(['datetime_x', 'datetime_y'], axis=1)

        pairs_df['pair_key'] = pairs_df[['image_id_x', 'image_id_y']].apply(
            lambda x: "{}_{}".format(x['image_id_x'], x['image_id_y']), axis=1
        )

        two_epoch_df = (
            measurements.sort_values(by='time')
            .groupby("source")["id"]
            .apply(lambda x: pd.DataFrame(list(combinations(x, 2))))
            .reset_index(level=1, drop=True)
            .rename(
                columns={0: "meas_id_a", 1: "meas_id_b"}
            ).astype(int).reset_index()
        )

        measurements = measurements.set_index(["source", "id"])[[
            'image_id',
            'flux_peak',
            'flux_peak_err',
            'flux_int',
            'flux_int_err',
            'has_siblings',
            'forced'
        ]]

        two_epoch_df = two_epoch_df.join(
            measurements,
            on=["source", "meas_id_a"],
        ).join(
            measurements,
            on=["source", "meas_id_b"],
            lsuffix="_x",
            rsuffix="_y",
        )

        two_epoch_df['forced_count'] = two_epoch_df[
            ['forced_x', 'forced_y']
        ].sum(axis=1)

        two_epoch_df['siblings_count'] = two_epoch_df[
            ['has_siblings_x', 'has_siblings_y']
        ].sum(axis=1)

        two_epoch_df['pair_key'] = two_epoch_df[
            ['image_id_x', 'image_id_y']
        ].apply(
            lambda x: "{}_{}".format(x['image_id_x'], x['image_id_y']), axis=1
        )

        two_epoch_df = two_epoch_df.merge(
            pairs_df[['id', 'pair_key']].rename(columns={'id': 'pair_id'}),
            left_on='pair_key',
            right_on='pair_key'
        )

        two_epoch_df['source'] = two_epoch_df['source'].astype(int)

        pair_counts = two_epoch_df[
            ['pair_key', 'image_id_x']
        ].groupby('pair_key').count().rename(
            columns={'image_id_x': 'total_pairs'}
        )

        pairs_df = pairs_df.merge(
            pair_counts, left_on='pair_key', right_index=True
        )

        pairs_df = pairs_df.dropna(subset=['total_pairs']).set_index('id')

        return pairs_df, two_epoch_df

    def _calculate_metrics(self, df, use_int_flux=False):
        '''
        Calcualtes the V and m two epoch metrics.

        :param df: Dataframe of source pairs with flux information.
        :type df: pandas.core.frame.DataFrame
        :param use_int_flux: When 'True', integrated flux is used
            instead of peak flux, defaults to 'False'.
        :type use_int_flux: bool, optional.

        :returns: Dataframe with the metrics added.
        :rtype: pandas.core.frame.DataFrame
        '''

        if use_int_flux:
            flux = 'int'
        else:
            flux = 'peak'

        flux_x_label = "flux_{}_x".format(flux)
        flux_err_x_label = "flux_{}_err_x".format(flux)
        flux_y_label = "flux_{}_y".format(flux)
        flux_err_y_label = "flux_{}_err_y".format(flux)

        df["Vs"] = np.abs(
            (
                df[flux_x_label]
                - df[flux_y_label]
            )
            / np.hypot(
                df[flux_err_x_label],
                df[flux_err_y_label]
            )
        )

        df["m"] = (
            (
                df[flux_x_label]
                - df[flux_y_label]
            )
            / ((
                df[flux_x_label]
                + df[flux_y_label]
            ) / 2.)
        )

        return df

    def plot_epoch_pairs_bokeh(
        self,
        df: pd.DataFrame,
        pairs: pd.DataFrame,
        vs_min=4.3,
        m_min=0.26,
    ) -> Model:
        '''
        Adapted from code written by Andrew O'Brien.
        Plot the results of the two epoch analysis. It is run in
        'run_two_epoch_analysis' but can also be run separately.
        Returns a bokeh plot.

        :param df: Dataframe of source pairs with metric information.
        :type df: pandas.core.frame.DataFrame.
        :param pairs: Dataframe containing the pairs information.
        :type pairs: pandas.core.frame.DataFrame.
        :param vs_min: The minimum Vs metric value to be considered
            a candidate, defaults to 4.3.
        :type vs_min: float, optional.
        :param m_min: The minimum m metric absolute value to be
            considered a candidates, defaults to 0.26.
        :type m_min: float, optional.

        :returns: Bokeh grid containing plots.
        :rtype: bokeh.models.grids.Grid
        '''

        light_curve_pairs = pairs.index.values

        GRID_WIDTH = 3
        PLOT_WIDTH = 500
        PLOT_HEIGHT = 300
        x_range = DataRange1d(start=0.5)
        m_max_abs = df.query("Vs >= @vs_min")["m"].abs().max()
        y_range = Range1d(start=-m_max_abs, end=m_max_abs)
        epoch_pair_figs = []
        for epoch_pair in light_curve_pairs:
            td_days = pairs.loc[epoch_pair]['td'].days
            df_filter = df.query("pair_id == @epoch_pair")
            fig = figure(
                plot_width=PLOT_WIDTH,
                plot_height=PLOT_HEIGHT,
                x_axis_type="log",
                x_range=x_range,
                y_range=y_range,
                x_axis_label="Vs",
                y_axis_label="m",
                title=f"{epoch_pair}: {td_days:.2f} days",
                tools="pan,box_select,lasso_select,box_zoom,wheel_zoom,reset",
                tooltips=[("source", "@source")],
            )
            fig.scatter(
                f"Vs",
                f"m",
                source=df_filter,
                marker="circle",
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
            epoch_pair_figs.append(fig)

        # reshape fig list for grid layout
        epoch_pair_figs = [
            epoch_pair_figs[i: i + GRID_WIDTH]
            for i in range(0, len(epoch_pair_figs), GRID_WIDTH)
        ]
        grid = gridplot(
            epoch_pair_figs, plot_width=PLOT_WIDTH, plot_height=PLOT_HEIGHT
        )
        grid.css_classes.append("mx-auto")

        return grid

    def run_two_epoch_analysis(
        self, v, m, query=None,
        df=None, use_int_flux=False
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
        :type pairs: bool, optional.

        :returns: tuple containing the pairs dataframe, the two epoch dataframe
            containing the calculated metrics, the dataframe of candidates and
            the bokeh plot.
        :rtype: pandas.core.frame.DataFrame, pandas.core.frame.DataFrame,
            pandas.core.frame.DataFrame, bokeh.models.grids.Grid
        '''
        if df is None:
            df = self.sources

        if query is not None:
            df = df.query(query)

        allowed_sources = df.index.tolist()

        pairs, df = self._get_two_epoch_df(
            allowed_sources=allowed_sources
        )

        df = self._calculate_metrics(df, use_int_flux=use_int_flux)

        candidates = df.loc[(df['Vs'] > v) & (df['m'].abs() > m)]

        plot = self.plot_epoch_pairs_bokeh(
            df, pairs, vs_min=v, m_min=m
        )

        return pairs, df, candidates, plot

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
            'Viridis256',
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
