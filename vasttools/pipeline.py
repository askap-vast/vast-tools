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


class Pipeline(object):
    """Class to interface with VAST Pipeline results"""
    def __init__(self, project_dir):
        super(Pipeline, self).__init__()

        self.project_dir = os.path.abspath(project_dir)

    def _get_source_counts(self, df):

        d = {}

        d['n_detections'] = df.shape[0]
        force_mask = (df['forced'] == True)
        d['n_selavy'] = df[~force_mask].shape[0]
        d['n_forced'] = df.shape[0] - d['n_selavy']
        d['has_siblings'] = df['has_siblings'].any()

        return pd.Series(d)

    def load_run(self, runname, use_dask=False):
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

        images = dd.read_parquet(
            os.path.join(
                run_dir,
                'images.parquet'
            ),
            engine='pyarrow'
        )

        associations = dd.read_parquet(
            os.path.join(
                run_dir,
                'associations.parquet'
            ),
            engine='pyarrow'
        )

        skyregions = dd.read_parquet(
            os.path.join(
                run_dir,
                'skyregions.parquet'
            ),
            engine='pyarrow'
        )

        sources = dd.read_parquet(
            os.path.join(
                run_dir,
                'sources.parquet'
            ),
            engine='pyarrow'
        ).set_index('id')

        measurements = dd.read_parquet(
            m_files,
            engine='pyarrow'
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
                'measurements_path'
            ]], how='left',
            left_on='image_id',
            right_on='id'
        ).rename(
            columns={
                'path': 'image',
                'noise_path': 'rms',
                'measurements_path': 'selavy'
            }
        )

        meta = {
            'n_detections': 'i',
            'n_selavy': 'i',
            'n_forced': 'i',
            'has_siblings': 'b',
        }

        source_counts = (
            measurements[[
                'source',
                'forced',
                'has_siblings'
            ]].groupby('source')
            .apply(
                self._get_source_counts,
                meta=meta
            )
        )

        sources = sources.merge(
            source_counts,
            how='left',
        )

        if not use_dask:
            images = images.compute()
            associations = associations.compute()
            measurements = measurements.compute(
                scheduler='processes',
                n_workers=4
            )
            sources = sources.compute(
                scheduler='processes',
                n_workers=4
            )


        piperun = PipeRun(
            name=runname,
            associations=associations,
            images=images,
            skyregions=skyregions,
            sources=sources,
            measurements=measurements,
            dask=use_dask
        )

        return piperun


class PipeRun(object):
    """An individual pipeline run"""
    def __init__(
        self, name=None, associations=None, images=None,
        skyregions=None, sources=None, measurements=None,
        dask=None
    ):
        super(PipeRun, self).__init__()
        self.name = name
        self.associations = associations
        self.images = images
        self.skyregions = skyregions
        self.sources = sources
        self.measurements = measurements
        self.dask = dask

    def get_source(self, id, field=None, stokes='I', outdir='.'):

        measurements = self.measurements.groupby(
            'source'
        ).get_group(id)

        if self.dask:
            measurements = measurements.compute(
                scheduler='processes',
                n_workers=4
            )

        measurements = measurements.rename(
            columns={
              'time': 'dateobs',
            }
        ).sort_values(
            by='dateobs'
        ).reset_index(drop=True)

        s = self.sources.loc[id]

        num_measurements = s['n_detections']

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
