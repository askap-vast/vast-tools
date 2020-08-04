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
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from matplotlib.font_manager import FontProperties
from astroML import density_estimation


matplotlib.pyplot.switch_backend('Agg')


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

    def _get_two_epoch_df(self, allowed_sources=[]):
        image_ids = self.images.index.tolist()

        combs = combinations(
            image_ids, 2
        )

        if len(allowed_sources) > 0:
            measurements = self.measurements.loc[
                self.measurements['source'].isin(
                    allowed_sources
                )
            ]
        else:
            measurements = self.measurements

        measurements_images = {}
        for i in image_ids:
            measurements_images[i] = measurements[[
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
                measurements['image_id'] == i
            ]

        pairs = {
            'pair': [],
            'id': [],
            'td': []
        }

        for i, c in enumerate(combs):
            img_1 = c[0]
            img_2 = c[1]

            pair_key = i+1

            pairs['pair'].append("{}_{}".format(
                img_1, img_2
            ))
            pairs['id'].append(pair_key)
            pairs['td'].append(
                self.images.loc[img_2].datetime
                - self.images.loc[img_1].datetime
            )

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
                two_epoch_df = first_set
            else:
                two_epoch_df = two_epoch_df.append(first_set)

        pairs = pd.DataFrame.from_dict(pairs).set_index(
            'id'
        ).sort_values(by='td')

        return pairs, two_epoch_df

    def _calculate_metrics(self, df, use_int_flux=False):

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
            df_filter = df.query("pair == @epoch_pair")
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

    def gaussian_fit(self, data, param_mean, param_sigma):
        range_data = np.linspace(min(data), max(data), 1000)
        fit = norm.pdf(range_data, loc=param_mean, scale=param_sigma)

        return range_data, fit

    def make_bins(self, x):
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

    def plot_eta_v_matplotlib(
        self, df, eta_fit_mean, eta_fit_sigma,
        v_fit_mean, v_fit_sigma, eta_cutoff, v_cutoff,
        use_int_flux=False
    ):
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
            x, bins=self.make_bins(x), density=1,
            histtype='stepfilled', color='C0'
        )
        axHisty.hist(
            y, bins=self.make_bins(y), density=1,
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

        range_x, fitx = self.gaussian_fit(x, eta_fit_mean, eta_fit_sigma)
        axHistx.plot(range_x, fitx, 'k:', linewidth=2)
        range_y, fity = self.gaussian_fit(y, v_fit_mean, v_fit_sigma)
        axHisty.plot(fity, range_y, 'k:', linewidth=2)

        return fig

    def plot_eta_v_bokeh(
        self, df, eta_fit_mean, eta_fit_sigma,
        v_fit_mean, v_fit_sigma, eta_cutoff, v_cutoff,
        use_int_flux=False
    ):
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
            plot = self.plot_eta_v_bokeh(
                df, eta_fit_mean, eta_fit_sigma,
                v_fit_mean, v_fit_sigma, eta_cutoff, v_cutoff,
                use_int_flux=use_int_flux
            )
        else:
            plot = self.plot_eta_v_matplotlib(
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
