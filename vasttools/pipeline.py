import pandas as pd
import os
import warnings
import glob
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
from astropy.stats import sigma_clip, mad_std
from astropy.coordinates import SkyCoord
from astropy import units as u
from vasttools.source import Source
from vasttools.utils import match_planet_to_field
from multiprocessing import cpu_count
from datetime import timedelta
from itertools import combinations


class Pipeline(object):
    """Class to interface with VAST Pipeline results"""
    def __init__(self, project_dir):
        super(Pipeline, self).__init__()

        self.project_dir = os.path.abspath(project_dir)

    def list_piperuns(self):
        jobs = sorted(glob.glob(
            os.path.join(self.project_dir, "*")
        ))

        jobs = [i.split("/")[-1] for i in jobs]
        jobs.remove('images')

        return jobs

    def list_images(self):
        img_list = sorted(glob.glob(
            os.path.join(self.project_dir, "images", "*")
        ))

        img_list = [i.split("/")[-1] for i in img_list]

        return img_list

    def load_run(
        self, runname, n_workers=cpu_count() - 1
    ):
        """
        Load a pipeline run.
        If use_dask is True used then the data is loaded into
        dask dataframes.
        """

        run_dir = os.path.join(
            self.project_dir,
            runname
        )

        if not os.path.isdir(run_dir):
            raise ValueError(
                "Run '%s' does not exist!",
                runname
            )
            return

        images = pd.read_parquet(
            os.path.join(
                run_dir,
                'images.parquet'
            )
        )

        m_files = images['measurements_path'].tolist()
        m_files += sorted(glob.glob(os.path.join(
            run_dir,
            "forced_measurements*.parquet"
        )))

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
        ).merge( # second merge for band
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

        measurements = pd.read_parquet(
            m_files[0],
            engine='pyarrow'
        )

        for m in m_files[1:]:
            measurements = measurements.append(
                pd.read_parquet(m)
            )

        measurements = measurements.merge(
            associations, left_on='id', right_on='meas_id',
            how='left'
        ).drop([
            'meas_id',
        ], axis=1).rename(
            columns={
                'source_id': 'source',
            }
        )

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

        sources = sources.join(
            measurements[[
                'source',
                'forced',
            ]].groupby('source').count(),
            how='left'
        ).rename(columns={
            'forced': 'n_datapoints'
        })
        sources = sources.merge(
            measurements[[
                'source',
                'forced',
                'has_siblings'
            ]].groupby('source').agg(
                {
                    'forced': sum,
                    # below works in pandas but not Dask
                    'has_siblings': any
                }
            ),
            how='left', right_index=True, left_index=True
        ).rename(columns={
            'forced': 'n_forced',
        })

        sources['n_forced'] = sources['n_forced'].astype(int)
        sources['n_datapoints'] = sources['n_datapoints'].astype(int)

        sources['n_selavy'] = (
            sources['n_datapoints'] - sources['n_forced']
        )

        relations = relations[relations['related_with'] != -1]

        sources = sources.merge(
            relations.groupby('id').count(),
            how='left', left_index=True, right_index=True
        ).fillna(0).rename(
            columns={
                'related_with': 'n_relations'
            }
        )

        sources['n_relations'] = sources[
            'n_relations'
        ].astype(int)

        piperun = PipeAnalysis(
            test="no",
            name=runname,
            images=images,
            skyregions=skyregions,
            relations=relations,
            sources=sources,
            measurements=measurements,
        )

        return piperun


class PipeRun(object):
    """An individual pipeline run"""
    def __init__(
        self, name, images,
        skyregions, relations, sources,
        measurements, n_workers=cpu_count() - 1
    ):
        super(PipeRun, self).__init__()
        self.name = name
        self.images = images
        self.skyregions = skyregions
        self.sources = sources
        self.measurements = measurements
        self.relations = relations
        self.n_workers = n_workers

    def get_source(self, id, field=None, stokes='I', outdir='.'):

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

        if 'n_datapoints' in measurements.columns:
            num_measurements = s['n_datapoints']
        else:
            num_measurements = measurements.shape[0]

        source_coord = SkyCoord(
            s['wavg_ra'],
            s['wavg_dec'],
            unit=(u.deg, u.deg)
        )

        source_name = "VAST {}".format(
            source_coord.to_string("hmsdms")
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

        from vasttools.survey import ALLOWED_PLANETS
        ap = ALLOWED_PLANETS

        planets_df = self.images.loc[:, [
            'id',
            'datetime',
            'duration',
            'centre_ra',
            'centre_dec',
            'xtr_radius'
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


class PipeAnalysis(PipeRun):
    """docstring for PipeAnalysis"""
    def __init__(
        self, test, name, images,
        skyregions, relations, sources,
        measurements, n_workers=cpu_count() - 1
    ):
        super().__init__(
            name, images,
            skyregions, relations, sources,
            measurements, n_workers
        )
        self.test = test

    def _get_source_two_epochs(self, group):
        image_ids = self.images['id'].tolist()

        combs = combinations(
            image_ids, 2
        )

        measurements_images = {}
        for i in image_ids:
            measurements_images[i] = self.measurements[[
                'image_id',
                'source',
                'id',
                'flux_peak',
                'flux_peak_err',
                'flux_int',
                'flux_int_err',
                'has_siblings',
                'forced'
            ]].loc[
                self.measurements['image_id'] == i
            ]

        pairs = {}

        for i, c in enumerate(combs):
            img_1 = c[0]
            img_2 = c[1]

            pair_key = i+1

            pairs["{}_{}".format(
                img_1, img_2
            )] = pair_key

            first_set = measurements_images[img_1]
            second_set = measurements_images[img_2]

            first_set = first_set.merge(second_set, on='source')

            first_set['pair'] = pair_key
            first_set['forced_count'] = first_set[
                ['forced_x', 'forced_y']
            ].sum(axis=1)
            first_set['siblings_count'] = first_set[
                ['has_siblings_x', 'has_siblings_y']
            ].sum(axis=1)

            if i == 0:
                self.two_epoch_df = first_set
            else:
                self.two_epoch_df = self.two_epoch_df.append(first_set)

        return pairs

    def _calculate_metrics(self, use_int_flux=False):

        if use_int_flux:
            flux = 'int'
        else:
            flux = 'peak'

        flux_x_label = "flux_{}_x".format(flux)
        flux_err_x_label = "flux_{}_err_x".format(flux)
        flux_y_label = "flux_{}_y".format(flux)
        flux_err_y_label = "flux_{}_err_y".format(flux)

        self.two_epoch_df["Vs"] = np.abs(
            (self.two_epoch_df[flux_x_label] - self.two_epoch_df[flux_y_label])
            / np.hypot(
                self.two_epoch_df[flux_err_x_label],
                self.two_epoch_df[flux_err_y_label]
            )
        )

        self.two_epoch_df["m"] = np.abs(
            (self.two_epoch_df[flux_x_label] - self.two_epoch_df[flux_y_label]) /
            ((self.two_epoch_df[flux_x_label] + self.two_epoch_df[flux_y_label]) / 2.)
        )

    def two_epoch_search(self, v, m, use_int_flux=False):

        self.pairs = self._get_two_epoch_df()

        self._calculate_metrics(use_int_flux=use_int_flux)

        return pairs


def plot_eta_v(
    df: pd.DataFrame,
    v_sigma: float = 2.0,
    eta_sigma: float = 2.0
):
    v_sigma = v_sigma
    eta_sigma = eta_sigma

    # source = df['id']
    # source_selavy = ColumnDataSource(data)

    eta_peak_log = np.log10(df['eta_peak'])
    v_peak_log = np.log10(df['v_peak'])

    eta_peak_log_clipped = sigma_clip(
        eta_peak_log, masked=False, stdfunc=mad_std, sigma=2
    )
    v_peak_log_clipped = sigma_clip(
        v_peak_log, masked=False, stdfunc=mad_std, sigma=2
    )

    eta_peak_fit_mean, eta_peak_fit_sigma = norm.fit(eta_peak_log_clipped)
    v_peak_fit_mean, v_peak_fit_sigma = norm.fit(v_peak_log_clipped)

    v_cutoff = v_peak_fit_mean + v_sigma * v_peak_fit_sigma
    eta_cutoff = eta_peak_fit_mean + eta_sigma * eta_peak_fit_sigma

    # generate fitted curve data for plotting
    eta_x = np.linspace(
        norm.ppf(0.001, loc=eta_peak_fit_mean, scale=eta_peak_fit_sigma),
        norm.ppf(0.999, loc=eta_peak_fit_mean, scale=eta_peak_fit_sigma),
    )
    eta_y = norm.pdf(eta_x, loc=eta_peak_fit_mean, scale=eta_peak_fit_sigma)

    v_x = np.linspace(
        norm.ppf(0.001, loc=v_peak_fit_mean, scale=v_peak_fit_sigma),
        norm.ppf(0.999, loc=v_peak_fit_mean, scale=v_peak_fit_sigma),
    )
    v_y = norm.pdf(v_x, loc=v_peak_fit_mean, scale=v_peak_fit_sigma)

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
    fig.scatter(
        x="eta_peak", y="v_peak", color=cmap,
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
        title="VAST eta-V",
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
        location=10 ** eta_cutoff,
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
        location=10 ** v_cutoff,
        dimension="width",
        line_color="black",
        line_dash="dashed",
    )
    y_hist.add_layout(y_hist_sigma_span)
    fig.add_layout(y_hist_sigma_span)

    variable_region = BoxAnnotation(
        left=10 ** eta_cutoff,
        bottom=10 ** v_cutoff,
        fill_color="orange",
        level="underlay",
    )
    fig.add_layout(variable_region)
    # slider = Slider(start=0, end=7., step=0.5, value=2)
    # slider.js_link('value', r.left, 'radius')
    grid = gridplot([[x_hist, Spacer(width=100, height=100)], [fig, y_hist]])
    grid.css_classes.append("mx-auto")

    return grid, 10 ** eta_cutoff, 10 ** v_cutoff
