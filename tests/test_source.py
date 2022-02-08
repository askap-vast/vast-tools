import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from astropy.coordinates import Angle, SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from matplotlib.pyplot import Figure
from pathlib import Path
from pytest_mock import mocker
from radio_beam import Beam
from typing import Optional

import vasttools.source as vts


TEST_DATA_DIR = Path(__file__).resolve().parent / 'data'
"""
Test data directory relative to this file.
"""


@pytest.fixture
def get_measurements() -> pd.DataFrame:
    """
    Load the measurement dataframes for the dummy source.

    The dataframe is loaded from the test data directory.
    Note that all source measurements dataframes are pandas so no vaex
    testing is required.

    Returns:
        The dataframe containing the source measurements.
    """
    def _get_measurements(pipeline: bool = False,
                          multi_freq: bool = True
                          ) -> pd.DataFrame:
        """
        The workhorse function to load the measurements.

        Args:
            pipeline: If 'True' the pipeline version of measurements are
                loaded in-place of the query version.
            multi_freq: Include multifrequency data

        Returns:
            The dataframe containing the measurements.
        """
        if pipeline:
            filepath = TEST_DATA_DIR / 'psr-j2129-04-pipe-meas.csv'
        else:
            filepath = TEST_DATA_DIR / 'psr-j2129-04-query-meas.csv'
        meas_df = pd.read_csv(filepath)
        meas_df['dateobs'] = pd.to_datetime(meas_df['dateobs'])

        freq_col = 'frequency'

        if multi_freq:
            temp_df = meas_df.copy()
            temp_df[freq_col] = 1367.5
            meas_df = pd.concat([meas_df, temp_df], ignore_index=True)
            del temp_df

        if not pipeline:
            meas_df['skycoord'] = meas_df[['ra', 'dec']].apply(
                lambda row: SkyCoord(
                    [row['ra']],
                    [row['dec']],
                    unit=(u.deg, u.deg)
                ),
                axis=1
            )
        else:
            # recast pipeline epoch column as string
            meas_df['epoch'] = meas_df['epoch'].astype(str)

        return meas_df
    return _get_measurements


def dummy_filter_selavy_components(x, *args, **kwargs) -> pd.DataFrame:
    """
    A dummy filter selavy components return function that just returns
    the dataframe that was passed to it.

    Args:
        x: The dataframe argument.
        args: Positional arguments.
        kwargs: Keyword arguments.

    Returns:
        Original dataframe argument x.
    """
    return x


@pytest.fixture
def dummy_selavy_components() -> pd.DataFrame:
    """
    Provides a dummy set of selavy components containing only the columns
    required for testing.

    Returned as a pandas dataframe.

    Returns:
        The dataframe containing the dummy selavy components.
    """
    df = pd.DataFrame(data={
        'island_id': {
            0: 'SB9667_island_1000',
            1: 'SB9667_island_1001',
            2: 'SB9667_island_1002',
            3: 'SB9667_island_1003',
            4: 'SB9667_island_1004'
        },
        'ra_deg_cont': {
            0: 321.972731,
            1: 317.111595,
            2: 322.974588,
            3: 315.077869,
            4: 315.56781
        },
        'dec_deg_cont': {
            0: 0.699851,
            1: 0.53981,
            2: 1.790072,
            3: 3.011253,
            4: -0.299919
        },
        'maj_axis': {0: 15.6, 1: 18.48, 2: 21.92, 3: 16.77, 4: 14.67},
        'min_axis': {0: 14.23, 1: 16.03, 2: 16.67, 3: 12.4, 4: 13.64},
        'pos_ang': {0: 111.96, 1: 43.18, 2: 22.71, 3: 57.89, 4: 63.43}
    })

    return df


@pytest.fixture
def dummy_fits() -> fits.HDUList:
    """
    Provides a dummy FITS file to use for FITS operations and plotting.

    Pixel values are randomly generated.

    Returns:
        The HDUList object.
    """
    data = np.random.rand(100, 100)

    hdu = fits.PrimaryHDU(data=data)

    header = hdu.header

    header['BMAJ'] = 0.00493462835125746
    header['BMIN'] = 0.00300487516378073
    header['BPA'] = -71.0711523845679
    header['CDELT1'] = 1.0
    header['CDELT2'] = 1.0
    header['WCSAXES'] = 2
    header['TELESCOP'] = "ASKAP"
    header['RESTFREQ'] = 887491000.0
    header['DATE-OBS'] = "2020-01-12T05:36:03.834"
    header['MJD-OBS'] = Time(header['DATE-OBS']).mjd
    header['DATE-BEG'] = "2020-01-12T05:36:03.834"
    header['MJD-BEG'] = Time(header['DATE-BEG']).mjd
    header['DATE-END'] = "2020-01-12T05:47:50.517"
    header['MJD-END'] = Time(header['DATE-END']).mjd
    header['TIMESYS'] = "UTC"
    header['RADESYS'] = "ICRS"
    header['CTYPE1'] = "RA---SIN"
    header['CUNIT1'] = "deg"
    header['CRVAL1'] = 319.6519091667
    header['CRPIX1'] = 4059.5
    header['CD1_1'] = -0.0006944444444444
    header['CD1_2'] = 0.0
    header['CTYPE2'] = "DEC--SIN"
    header['CUNIT2'] = "deg"
    header['CRVAL2'] = -6.2985525
    header['CRPIX2'] = -2537.5
    header['CD2_1'] = 0.0
    header['CD2_2'] = 0.0006944444444444

    hdul = fits.HDUList([hdu])

    return hdul


@pytest.fixture
def source_instance(
    get_measurements: pd.DataFrame,
    dummy_selavy_components: pd.DataFrame,
    dummy_fits: fits.HDUList
) -> vts.Source:
    """
    Generates a source instance to use for testing.

    Args:
        get_measurements: The pytest fixture that loads the measurements.
        dummy_selavy_components: The pytest fixture that provides a dummy set
            of selavy components (to mimic the loading of the data selavy
            components).
        dummy_fits: The pytest fixture that provides a dummy fits file.

    Returns:
        The vast tools Source instance for testing.
    """
    def _get_source_instance(
        pipeline: bool = False,
        add_cutout_data: bool = False,
        multi_freq: bool = False
    ):
        """
        The workhorse function that actually generates the source.

        Args:
            pipeline: If 'True' the pipeline measurements are used in-place of
                the query measurements.
            add_cutout_data: If 'True' the cutout data is added to the
                source instance before returning.
            multi_freq: Include multifrequency data

        Returns:
            The vast tools Source instance for testing.
        """
        psr_coordinate = SkyCoord(
            322.4387083,
            -4.4866389,
            unit=(u.deg, u.deg)
        )
        name = 'PSR J2129-04'
        meas_df = get_measurements(pipeline=pipeline, multi_freq=multi_freq)

        if pipeline:
            epochs = ['1', '2', '3', '4', '5', '6']
            fields = ['VAST_2118-06A' for i in range(6)]
            forced_fits = False
        else:
            epochs = meas_df['epoch'].to_list()
            fields = meas_df['fields'].to_list()
            forced_fits = True

        stokes = "I"
        primary_field = "VAST_2118-06A"
        crossmatch_radius = Angle(10 * u.arcsec)
        base_folder = '/path/to/data/'

        source = vts.Source(
            psr_coordinate,
            name,
            epochs,
            fields,
            stokes,
            primary_field,
            crossmatch_radius,
            meas_df,
            base_folder,
            pipeline=pipeline,
            forced_fits=forced_fits
        )

        if add_cutout_data:
            hdul = dummy_fits
            wcs = WCS(hdul[0].header)
            selavy_components = dummy_selavy_components
            beam = Beam.from_fits_header(hdul[0].header)
            cutout_df = pd.DataFrame(
                columns=[
                    "data",
                    "wcs",
                    "header",
                    "selavy_overlay",
                    "beam"
                ]
            )

            for i in range(source.measurements.shape[0]):
                cutout_df = cutout_df.append(pd.DataFrame(
                    data={
                        "data": [hdul[0].data],
                        "wcs": [wcs],
                        "header": [hdul[0].header],
                        "selavy_overlay": [selavy_components],
                        "beam": [beam]
                    }
                ))

            source.cutout_df = cutout_df.reset_index(drop=True)

            source._cutouts_got = True
            source._size = Angle(5. * u.arcmin)

        return source
    return _get_source_instance


class TestSource:
    """
    Contains all the tests related to the Source class in vast tools.
    """
    @pytest.mark.parametrize("pipeline", [False, True])
    def test_init(
        self,
        pipeline: bool,
        source_instance: vts.Source,
    ) -> None:
        """
        Tests the initialisation of the source object.

        Parametrized for pipeline a query types.

        Args:
            pipeline: If 'True' then the Source is initialised as a
                pipeline source.
            source_instance: The pytest source_instance fixture.

        Returns:
            None
        """
        source = source_instance(pipeline=pipeline)

        assert source.pipeline == pipeline

    @pytest.mark.parametrize(
        "pipeline,simple,outfile",
        [
            (False, False, None),
            (False, True, None),
            (True, False, None),
            (True, True, None),
            (True, True, 'test_name.csv')
        ]
    )
    def test_write_measurements(
        self,
        pipeline: bool,
        simple: bool,
        outfile: Optional[str],
        source_instance: vts.Source,
        mocker
    ) -> None:
        """
        Tests the initialisation of the source object.

        Parametrized for pipeline a query types, in addition to the simple
        flag and outfile name.

        Args:
            pipeline: If 'True' then the Source is initialised as a
                pipeline source.
            simple: Used for the simple flag in the write_measurements method.
            outfile: Used as the outfile argument in the write_measurements
                method.
            source_instance: The pytest source_instance fixture.

        Returns:
            None
        """
        mocker_to_csv = mocker.patch('vasttools.source.pd.DataFrame.to_csv')

        source = source_instance(pipeline=pipeline)
        source.write_measurements(simple=simple, outfile=outfile)

        if outfile is None:
            expected_call = 'PSR_J2129-04_measurements.csv'
        else:
            outfile = outfile

        mocker_to_csv.call_args.assert_called_once_with(outfile)

    @pytest.mark.parametrize(
        "min_points,min_detections,use_forced_for_all,mjd,start_date",
        [
            (10, 2, False, False, None),
            (2, 10, False, False, None),
            (2, 2, True, False, None),
            (2, 2, False, True, pd.Timestamp('2019-10-30T10:11:56.9130')),
            (2, 2, True, False, None),
            (2, 2, True, False, None),
        ]
    )
    def test_plot_lightcurve_errors(
        self,
        min_points: int,
        min_detections: int,
        use_forced_for_all: bool,
        mjd: bool,
        start_date: Optional[pd.Timestamp],
        source_instance: vts.Source
    ) -> None:
        """
        Tests the failures when using plot_lightcurve.

        Parametrized for min_points, min_detections, forced fits, mjd and
        start date.

        Args:
            min_points: The min_point argument to be passed to the
                plot_lightcurve method.
            min_detections: The min_detections argument to be passed to the
                plot_lightcurve method.
            used_forced_for_all: The used_forced_for_all argument to be passed
                to the plot_lightcurve method.
            mjd: The mjd argument to be passed to the plot_lightcurve method.
            start_date: The start_date argument to be passed to the
                plot_lightcurve method.
            source_instance: The pytest source_instance fixture.

        Returns:
            None
        """
        source = source_instance()

        if use_forced_for_all is True:
            source.forced_fits = False

        with pytest.raises(vts.SourcePlottingError) as excinfo:
            lightcurve = source.plot_lightcurve(
                min_points=min_points,
                min_detections=min_detections,
                use_forced_for_all=use_forced_for_all,
                mjd=mjd,
                start_date=start_date
            )

        if min_points > min_detections:
            assert str(excinfo.value).startswith("Number of datapoints")
        elif min_points < min_detections:
            assert str(excinfo.value).startswith("Number of detections")
        elif use_forced_for_all:
            assert str(excinfo.value) == (
                "Source does not have any forced fits points to plot."
            )
        else:
            assert str(excinfo.value) == (
                "The 'mjd' and 'start date' options "
                "cannot be used at the same time!"
            )

    @pytest.mark.parametrize(
        "pipeline,mjd,peak_flux,start_date,multi_freq",
        [
            (False, False, True, None, False),
            (False, True, True, None, False),
            (False, False, False, None, False),
            (False, False, True, pd.Timestamp(
                '2019-10-30T10:11:56.9130'
            ), False),
            (True, False, True, None, False),
            (True, True, True, None, False),
            (True, False, False, None, False),
            (True, False, True, pd.Timestamp(
                '2019-10-30T10:11:56.9130', tz='utc'
            ), False),
            (False, False, True, None, True),
            (True, False, True, None, True),
        ]
    )
    def test_plot_lightcurve(
        self,
        pipeline: bool,
        mjd: bool,
        peak_flux: bool,
        start_date: Optional[pd.Timestamp],
        multi_freq: bool,
        source_instance: vts.Source,
        get_measurements: pd.DataFrame
    ) -> None:
        """
        Tests the plot_lightcurve method.

        Parametrized for pipeline a query types, mjd, peak_flux and start
        date. The data in the plot is asserted against.

        Args:
            pipeline: If 'True' then the Source is initialised as a
                pipeline source.
            mjd: The mjd argument to be passed to the plot_lightcurve method.
            peak_flux: The peak_flux argument to be passed to the
                plot_lightcurve method.
            start_date: The start_date argument to be passed to the
                plot_lightcurve method.
            multi_freq: If `True` then multiple frequencies are plotted.
            source_instance: The pytest source_instance fixture.
            get_measurements: The pytest fixture that loads the measurements.

        Returns:
            None
        """
        source = source_instance(pipeline=pipeline, multi_freq=multi_freq)

        lightcurve = source.plot_lightcurve(
            mjd=mjd,
            peak_flux=peak_flux,
            start_date=start_date
        )

        measurements_df = get_measurements(pipeline=pipeline,
                                           multi_freq=multi_freq
                                           )
        freq_col = 'frequency'

        grouped_df = measurements_df.groupby(freq_col)
        freqs = list(grouped_df.groups.keys())

        expected_values = {}

        if peak_flux:
            flux_col = 'flux_peak'
        else:
            flux_col = 'flux_int'

        for i, (freq, meas_df) in enumerate(grouped_df):
            expected_values[freq] = {}
            if pipeline:
                temp_df = meas_df[meas_df['forced'] == True]
                expected_values[freq]['0_x'] = temp_df['dateobs'].to_numpy()
                expected_values[freq]['0_y'] = temp_df[flux_col].to_numpy()

                temp_df = meas_df[meas_df['forced'] == False]
                expected_values[freq]['1_x'] = temp_df['dateobs'].to_numpy()
                expected_values[freq]['1_y'] = temp_df[flux_col].to_numpy()
            else:
                temp_df = meas_df[meas_df['detection'] == False]
                expected_values[freq]['0_x'] = temp_df['dateobs'].to_numpy()
                upper_lims = temp_df['rms_image'].to_numpy() * 5.
                expected_values[freq]['0_y'] = upper_lims

                temp_df = meas_df[meas_df['detection'] == True]
                expected_values[freq]['2_x'] = temp_df['dateobs'].to_numpy()
                expected_values[freq]['2_y'] = temp_df[flux_col].to_numpy()

        freq_counter = 0
        line_counter = 0
        num_lines = len(lightcurve.axes[0].lines)
        for i, line in enumerate(lightcurve.axes[0].lines):
            # skip the dummy points
            if i == num_lines - len(freqs):
                break

            # skip the extra upper limit symbol on the lines
            if not pipeline and line_counter == 1:
                line_counter += 1
                continue

            x_data = line.get_xdata()
            y_data = line.get_ydata()

            freq = freqs[freq_counter]

            expected_x = expected_values[freq][f'{line_counter}_x']
            expected_y = expected_values[freq][f'{line_counter}_y']

            if mjd:
                expected_x = Time(expected_x).mjd
            elif start_date is not None:
                # change to series as then the subtraction works
                expected_x = pd.Series(expected_x)
                expected_x = (
                    (expected_x - start_date)
                    / pd.Timedelta(1, unit='d')
                )
            assert np.all(expected_x == x_data)
            assert np.all(expected_y == y_data)

            line_counter += 1
            if line_counter >= 2:
                line_counter = 0
                freq_counter += 1

        plt.close(lightcurve)

    @pytest.mark.parametrize(
        "use_forced_for_limits,use_forced_for_all",
        [(True, False), (False, True)]
    )
    def test_plot_lightcurve_forced_options(
        self,
        use_forced_for_limits: bool,
        use_forced_for_all: bool,
        source_instance: vts.Source,
        get_measurements: pd.DataFrame
    ) -> None:
        """
        Tests the plot_lightcurve method, specifically the forced options.

        Parametrized for forced all and force limits. The data in the plot
        is asserted against.

        Args:
            used_forced_for_limits: The used_forced_for_limits argument to be
                passed to the plot_lightcurve method.
            used_forced_for_all: The used_forced_for_all argument to be passed
                to the plot_lightcurve method.
            source_instance: The pytest source_instance fixture.
            get_measurements: The pytest fixture that loads the measurements.

        Returns:
            None
        """
        source = source_instance(multi_freq=False)

        lightcurve = source.plot_lightcurve(
            use_forced_for_limits=use_forced_for_limits,
            use_forced_for_all=use_forced_for_all
        )

        meas_df = get_measurements(multi_freq=False)
        expected_values = {}

        if use_forced_for_limits:
            temp_df = meas_df[meas_df['detection'] == False]
            expected_values['0_x'] = temp_df['dateobs'].to_numpy()
            expected_values['0_y'] = temp_df['f_flux_peak'].to_numpy()

            temp_df = meas_df[meas_df['detection'] == True]
            expected_values['2_x'] = temp_df['dateobs'].to_numpy()
            expected_values['2_y'] = temp_df['flux_peak'].to_numpy()
        else:
            expected_values['0_x'] = meas_df['dateobs'].to_numpy()
            expected_values['0_y'] = meas_df['f_flux_peak'].to_numpy()

        for i, line in enumerate(lightcurve.axes[0].lines):
            # skip dummy points
            if i > 0:
                continue
            x_data = line.get_xdata()
            y_data = line.get_ydata()

            expected_x = expected_values[f'{i}_x']
            expected_y = expected_values[f'{i}_y']

            assert np.all(expected_x == x_data)
            assert np.all(expected_y == y_data)

        plt.close(lightcurve)

    @pytest.mark.parametrize("pipeline", [False, True])
    def test__get_cutout(
        self,
        pipeline: bool,
        source_instance: vts.Source,
        dummy_selavy_components: pd.DataFrame,
        mocker
    ) -> None:
        """
        Tests the get_cutout method on the Source, which fetches the cutout
        data for each measurement.

        The

        Parametrized for pipeline and query type sources.

        Args:
            pipeline: If 'True' then the Source is initialised as a
                pipeline source.
            source_instance: The pytest source_instance fixture.
            dummy_selavy_components: The pytest fixture that provides a dummy
                set of selavy components (to mimic the loading of the data
                selavy components).
            mocker: The pytest-mock mocker object.

        Returns:
            None
        """
        source = source_instance(pipeline=pipeline)

        image_mocker = mocker.patch(
            'vasttools.source.Image'
        )

        cutout2d_mocker = mocker.patch(
            'vasttools.source.Cutout2D'
        )

        pandas_read_selavy_mocker = mocker.patch(
            'vasttools.source.read_selavy',
            return_value=dummy_selavy_components
        )

        pandas_read_parquet_mocker = mocker.patch(
            'vasttools.source.pd.read_parquet',
            return_value=dummy_selavy_components
        )

        filter_selavy_components_mocker = mocker.patch(
            'vasttools.source.filter_selavy_components',
            side_effect=dummy_filter_selavy_components
        )

        test_row = source.measurements.iloc[0]

        result = source._get_cutout(test_row)

        assert result[3].shape[0] == 5

    @pytest.mark.parametrize(
        "pipeline,expected",
        [
            (False, 'PSR_J2129-04_EPOCH01.fits'),
            (True, 'PSR_J2129-04_EPOCH1.fits')
        ]
    )
    def test__get_save_name(
        self,
        pipeline: bool,
        expected: str,
        source_instance: vts.Source
    ) -> None:
        """
        Test the get_save_name method of the source.

        Args:
            pipeline: If 'True' then the Source is initialised as a
                pipeline source.
            expected: The expected save name from the parametrisation.

        Returns:
            None
        """
        source = source_instance(pipeline=pipeline)

        outname = source._get_save_name('1', '.fits')

        assert outname == expected

    @pytest.mark.parametrize(
        "pipeline,selavy,no_islands,no_colorbar,"
        "title,crossmatch_overlay,hide_beam",
        [
            (False, True, True, False, None, False, False),
            (False, False, False, True, "New tile", True, True),
            (True, True, True, False, None, False, False),
            (True, False, False, True, "New tile", True, True),
        ]
    )
    def test_make_png_flag_options(
        self,
        pipeline: bool,
        selavy: bool,
        no_islands: bool,
        no_colorbar: bool,
        title: Optional[str],
        crossmatch_overlay: bool,
        hide_beam: bool,
        source_instance: vts.Source,
        mocker
    ) -> None:
        """
        Tests the make_png method, specifically with the options that are
        bool options.

        Parametrized for pipeline and selavy, no_islands, no_colourbar,
        title, crossmatch_overlay and hide_beam.

        Args:
            pipeline: If 'True' then the Source is initialised as a
                pipeline source.
            selavy: The selavy option that is passed to make_png. Passed from
                the parametrisation.
            no_islands: The no_islands option that is passed to make_png.
                Passed from the parametrisation.
            no_colourbar: The no_colourbar option that is passed to make_png.
                Passed from the parametrisation.
            title: The title option that is passed to make_png. Passed from
                the parametrisation.
            crossmatch_overlay: The crossmatch_overlay option that is passed
                to make_png. Passed from the parametrisation.
            hide_beam: The hide_beam option that is passed to make_png.
                Passed from the parametrisation.
            source_instance: The pytest source_instance fixture.
            mocker: The pytest-mock mocker object.

        Returns:
            None
        """
        source = source_instance(pipeline=pipeline, add_cutout_data=True)

        png_plot = source.make_png(
            '1',
            selavy=selavy,
            no_islands=no_islands,
            no_colorbar=no_colorbar,
            title=title,
            crossmatch_overlay=crossmatch_overlay,
            hide_beam=hide_beam
        )

        result_data = png_plot.axes[0].get_images()[0].get_array()
        result_title = png_plot.axes[0].get_title()

        if title is None:
            if pipeline:
                title = 'PSR J2129-04 Epoch 1 2019-08-27 13:38:38'
            else:
                title = 'PSR J2129-04 Epoch 1 2019-08-27 18:52:00'

        assert isinstance(png_plot, Figure)
        assert result_title == title
        assert np.all(result_data == source.cutout_df['data'].iloc[0] * 1.e3)
        plt.close(png_plot)

    @pytest.mark.parametrize(
        "pipeline,percentile,zscale,contrast",
        [
            (False, 95.0, False, 0.2),
            (False, 99.0, True, 0.25),
            (True, 95.0, False, 0.2),
            (True, 99.0, True, 0.25),
        ]
    )
    def test_make_png_scale_options(
        self,
        pipeline: bool,
        percentile: float,
        zscale: bool,
        contrast: float,
        source_instance: vts.Source,
        mocker
    ) -> None:
        """
        Tests the make_png method, specifically with the options that affect
        the scaling.

        Parametrized for pipeline, zscale and contrast.

        Args:
            pipeline: If 'True' then the Source is initialised as a
                pipeline source.
            percentile: The percentile option that is passed to make_png.
                Passed from the parametrisation.
            zscale: The zscale option that is passed to make_png.
                Passed from the parametrisation.
            contrast: The contrast option that is passed to make_png.
                Passed from the parametrisation.
            source_instance: The pytest source_instance fixture.
            mocker: The pytest-mock mocker object.

        Returns:
            None
        """
        source = source_instance(pipeline=pipeline, add_cutout_data=True)

        png_plot = source.make_png(
            '1',
            percentile=percentile,
            zscale=zscale,
            contrast=contrast
        )

        assert isinstance(png_plot, Figure)
        plt.close(png_plot)

    @pytest.mark.parametrize("pipeline", [False, True])
    def test_skyview_contour_plot(
        self,
        pipeline: bool,
        source_instance: vts.Source,
        dummy_fits: fits.HDUList,
        mocker
    ) -> None:
        """
        Tests the skyview_contour_plot method.

        Parametrized for pipeline and query source.

        Args:
            pipeline: If 'True' then the Source is initialised as a
                pipeline source.
            source_instance: The pytest source_instance fixture.
            dummy_fits: The pytest fixture dummy fits.
            mocker: The pytest-mock mocker object.

        Returns:
            None
        """
        source = source_instance(pipeline=pipeline, add_cutout_data=True)

        mocker_skyview = mocker.patch(
            'vasttools.source.SkyView.get_images',
            return_value=[dummy_fits]
        )

        result = source.skyview_contour_plot('1', 'suveycode')

        assert isinstance(result, Figure)
        plt.close(result)

    @pytest.mark.parametrize("pipeline", [False, True])
    def test_write_ann(
        self,
        pipeline: bool,
        source_instance: vts.Source,
        mocker
    ) -> None:
        """
        Tests the write ann method.

        The expected ann file is declared in the test.

        Parametrized for pipeline and query sources.

        Args:
            pipeline: If 'True' then the Source is initialised as a
                pipeline source.
            source_instance: The pytest source_instance fixture.
            mocker: The pytest-mock mocker object.

        Returns:
            None
        """
        source = source_instance(pipeline=pipeline, add_cutout_data=True)

        mocker_file_open = mocker.patch(
            'builtins.open',
            new_callable=mocker.mock_open()
        )

        if pipeline:
            filename = 'PSR_J2129-04_EPOCH1.ann'
            expected = (
                "COORD W\n"
                "PA SKY\n"
                "FONT hershey14\n"
                "COLOR BLUE\n"
                "CROSS 322.43744332078165 -4.48494112597578 "
                "0.0008333333333333334 0.0008333333333333334\n"
                "COLOR GREEN\n"
                "ELLIPSE 321.972731 0.699851 0.0021666666666666666 "
                "0.001976388888888889 111.96\n"
                "TEXT 321.972731 0.699851 island_1000\n"
                "ELLIPSE 317.111595 0.53981 0.0025666666666666667 "
                "0.002226388888888889 43.18\n"
                "TEXT 317.111595 0.53981 island_1001\n"
                "ELLIPSE 322.974588 1.790072 0.0030444444444444447 "
                "0.002315277777777778 22.71\n"
                "TEXT 322.974588 1.790072 island_1002\n"
                "ELLIPSE 315.077869 3.011253 0.0023291666666666665 "
                "0.0017222222222222222 57.89\n"
                "TEXT 315.077869 3.011253 island_1003\n"
                "ELLIPSE 315.56781 -0.299919 0.0020375 "
                "0.0018944444444444445 63.43\n"
                "TEXT 315.56781 -0.299919 island_1004\n"
            )
        else:
            filename = 'PSR_J2129-04_EPOCH01.ann'
            expected = (
                "COORD W\n"
                "PA SKY\n"
                "FONT hershey14\n"
                "COLOR BLUE\n"
                "CROSS 322.4387083333333 -4.486638888888889 "
                "0.0008333333333333334 0.0008333333333333334\n"
                "COLOR GREEN\n"
                "ELLIPSE 321.972731 0.699851 0.0021666666666666666 "
                "0.001976388888888889 111.96\n"
                "TEXT 321.972731 0.699851 island_1000\n"
                "ELLIPSE 317.111595 0.53981 0.0025666666666666667 "
                "0.002226388888888889 43.18\n"
                "TEXT 317.111595 0.53981 island_1001\n"
                "ELLIPSE 322.974588 1.790072 0.0030444444444444447 "
                "0.002315277777777778 22.71\n"
                "TEXT 322.974588 1.790072 island_1002\n"
                "ELLIPSE 315.077869 3.011253 0.0023291666666666665 "
                "0.0017222222222222222 57.89\n"
                "TEXT 315.077869 3.011253 island_1003\n"
                "ELLIPSE 315.56781 -0.299919 0.0020375 0.0018944444444444445 "
                "63.43\n"
                "TEXT 315.56781 -0.299919 island_1004\n"
            )

        source.write_ann('1')

        write_calls = (
            mocker_file_open.return_value.__enter__().write.call_args_list
        )
        file_string = ""
        for call in write_calls:
            file_string += call.args[0]

        mocker_file_open.assert_called_once_with(
            filename, 'w'
        )
        assert expected == file_string

    @pytest.mark.parametrize("pipeline", [False, True])
    def test_write_reg(
        self,
        pipeline: bool,
        source_instance: vts.Source,
        mocker
    ) -> None:
        """
        Tests the write reg method.

        The expected region file is declared in the test.

        Parametrized for pipeline and query sources.

        Args:
            pipeline: If 'True' then the Source is initialised as a
                pipeline source.
            source_instance: The pytest source_instance fixture.
            mocker: The pytest-mock mocker object.

        Returns:
            None
        """
        source = source_instance(pipeline=pipeline, add_cutout_data=True)

        mocker_file_open = mocker.patch(
            'builtins.open',
            new_callable=mocker.mock_open()
        )

        if pipeline:
            filename = 'PSR_J2129-04_EPOCH1.reg'
            expected = (
                "# Region file format: DS9 version 4.0\n"
                "global color=green font=\"helvetica 10 normal\" "
                "select=1 highlite=1 edit=1 move=1 delete=1 include=1 "
                "fixed=0 source=1\n"
                "fk5\n"
                "point(322.43744332078165 -4.48494112597578) "
                "# point=x color=blue\n"
                "ellipse(321.972731 0.699851 0.0021666666666666666 "
                "0.001976388888888889 201.95999999999998) # color=green\n"
                "text(321.9699532222222 0.699851 \"island_1000\") "
                "# color=green\n"
                "ellipse(317.111595 0.53981 0.0025666666666666667 "
                "0.002226388888888889 133.18) # color=green\n"
                "text(317.1088172222222 0.53981 \"island_1001\") "
                "# color=green\n"
                "ellipse(322.974588 1.790072 0.0030444444444444447 "
                "0.002315277777777778 112.71000000000001) # color=green\n"
                "text(322.9718102222222 1.790072 \"island_1002\") "
                "# color=green\n"
                "ellipse(315.077869 3.011253 0.0023291666666666665 "
                "0.0017222222222222222 147.89) # color=green\n"
                "text(315.0750912222222 3.011253 \"island_1003\") "
                "# color=green\n"
                "ellipse(315.56781 -0.299919 0.0020375 0.0018944444444444445 "
                "153.43) # color=green\n"
                "text(315.5650322222222 -0.299919 \"island_1004\") "
                "# color=green\n"
            )
        else:
            filename = 'PSR_J2129-04_EPOCH01.reg'
            expected = (
                "# Region file format: DS9 version 4.0\n"
                "global color=green font=\"helvetica 10 normal\" select=1 "
                "highlite=1 edit=1 move=1 delete=1 include=1 fixed=0 "
                "source=1\n"
                "fk5\n"
                "point(322.4387083333333 -4.486638888888889) # point=x "
                "color=blue\n"
                "ellipse(321.972731 0.699851 0.0021666666666666666 "
                "0.001976388888888889 201.95999999999998) # color=green\n"
                "text(321.9699532222222 0.699851 \"island_1000\") "
                "# color=green\n"
                "ellipse(317.111595 0.53981 0.0025666666666666667 "
                "0.002226388888888889 133.18) # color=green\n"
                "text(317.1088172222222 0.53981 \"island_1001\") "
                "# color=green\n"
                "ellipse(322.974588 1.790072 0.0030444444444444447 "
                "0.002315277777777778 112.71000000000001) # color=green\n"
                "text(322.9718102222222 1.790072 \"island_1002\") "
                "# color=green\n"
                "ellipse(315.077869 3.011253 0.0023291666666666665 "
                "0.0017222222222222222 147.89) # color=green\n"
                "text(315.0750912222222 3.011253 \"island_1003\") "
                "# color=green\n"
                "ellipse(315.56781 -0.299919 0.0020375 0.0018944444444444445 "
                "153.43) # color=green\n"
                "text(315.5650322222222 -0.299919 \"island_1004\") "
                "# color=green\n"
            )

        source.write_reg('1')

        write_calls = (
            mocker_file_open.return_value.__enter__().write.call_args_list
        )
        file_string = ""
        for call in write_calls:
            file_string += call.args[0]

        mocker_file_open.assert_called_once_with(
            filename, 'w'
        )
        assert expected == file_string

    @pytest.mark.parametrize(
        "input,expected",
        [
            ("SB11169_island_3616", "island_3616"),
            ("nSB11169_island_3616", "nisland_3616")
        ]
    )
    def test__remove_sbid(
        self,
        input: str,
        expected: str,
        source_instance: vts.Source
    ) -> None:
        """
        Tests the remove SBID method.

        Parametrized for negative island.

        Args:
            input: The input island name string.
            expected: The expected island name output.
            source_instance: The pytest source_instance fixture.

        Returns:
            None
        """
        source = source_instance()
        assert expected == source._remove_sbid(input)

    def test_simbad_search(self, source_instance: vts.Source, mocker) -> None:
        """
        Tests the simbad search method.

        The SIMBAD service is not queried, the call is mocked and asserted
        against along with the return value.

        Args:
            source_instance: The pytest source_instance fixture.
            mocker: The pytest-mock mocker object.

        Returns:
            None
        """
        source = source_instance()

        mocker_simbad = mocker.patch(
            'vasttools.source.Simbad.query_region',
            return_value=-99
        )

        test_radius = Angle(30. * u.arcsec)
        result = source.simbad_search(radius=test_radius)

        mocker_simbad.assert_called_once_with(
            source.coord, radius=test_radius
        )
        assert result == -99

    def test_ned_search(self, source_instance: vts.Source, mocker) -> None:
        """
        Tests the NED search method.

        The NED service is not queried, the call is mocked and asserted
        against along with the return value.

        Args:
            source_instance: The pytest source_instance fixture.
            mocker: The pytest-mock mocker object.

        Returns:
            None
        """
        source = source_instance()

        mocker_simbad = mocker.patch(
            'vasttools.source.Ned.query_region',
            return_value=-99
        )

        test_radius = Angle(30. * u.arcsec)
        result = source.ned_search(radius=test_radius)

        mocker_simbad.assert_called_once_with(
            source.coord, radius=test_radius
        )
        assert result == -99

    def test_casda_search(self, source_instance: vts.Source, mocker) -> None:
        """
        Tests the casda search method.

        The CASDA service is not queried, the call is mocked and asserted
        against along with the return value.

        Args:
            source_instance: The pytest source_instance fixture.
            mocker: The pytest-mock mocker object.

        Returns:
            None
        """
        # TODO: Should create a dummy response and test the other method
        #       options.
        source = source_instance()

        mocker_simbad = mocker.patch(
            'vasttools.source.Casda.query_region',
            return_value=-99
        )

        test_radius = Angle(30. * u.arcsec)
        result = source.casda_search(radius=test_radius, show_all=True)

        mocker_simbad.assert_called_once_with(
            source.coord, radius=test_radius
        )
        assert result == -99

    @pytest.mark.parametrize(
        "pipeline,suffix,forced_fits",
        [
            (False, 'peak', False),
            (False, 'peak', True),
            (False, 'int', False),
            (False, 'int', True),
            (True, 'peak', False),
            (True, 'int', False),
        ]
    )
    def test__get_fluxes_and_errors(
        self,
        pipeline: bool,
        suffix: str,
        forced_fits: bool,
        source_instance: vts.Source
    ) -> None:
        """
        Tests the get fluxes and errors method.

        The method gathers the detections and upper limits/forced fluxes
        depending on what is requested. Parametrized for pipeline and query
        sources.

        Args:
            pipeline: If 'True' then the Source is initialised as a
                pipeline source.
            suffix: Type of flux suffix, either 'int' or 'peak'. Passed to
                the get_fluxes_and_errors function.
            forced_fits: Whether to use forced fits for the upper limits.
                Passed to the get_fluxes_and_errors function.
            source_instance: The pytest source_instance fixture.

        Returns:
            None
        """
        source = source_instance(pipeline=pipeline)

        fluxes, errors = source._get_fluxes_and_errors(
            suffix=suffix,
            forced_fits=forced_fits
        )

        if pipeline:
            detection_label = 'forced'

            expected_fluxes = source.measurements[
                source.measurements[detection_label] == False
            ][f'flux_{suffix}'].append(source.measurements[
                source.measurements[detection_label] == True
            ][f'flux_{suffix}'])

            expected_errors = source.measurements[
                source.measurements[detection_label] == False
            ][f'flux_{suffix}_err'].append(source.measurements[
                source.measurements[detection_label] == True
            ][f'flux_{suffix}_err'])

        else:
            detection_label = 'detection'

            expected_fluxes = source.measurements[
                source.measurements[detection_label] == True
            ][f'flux_{suffix}']

            expected_errors = source.measurements[
                source.measurements[detection_label] == True
            ][f'flux_{suffix}_err']

            if forced_fits:
                expected_fluxes = expected_fluxes.append(
                    source.measurements[
                        source.measurements[detection_label] == False
                    ][f'f_flux_{suffix}']
                )

                expected_errors = expected_errors.append(
                    source.measurements[
                        source.measurements[detection_label] == False
                    ][f'f_flux_{suffix}_err']
                )
            else:
                expected_fluxes = expected_fluxes.append(
                    source.measurements[
                        source.measurements[detection_label] == False
                    ][f'rms_image'] * 5.
                )

                expected_errors = expected_errors.append(
                    source.measurements[
                        source.measurements[detection_label] == False
                    ][f'rms_image']
                )

        assert fluxes.equals(expected_fluxes)
        assert errors.equals(expected_errors)

    @pytest.mark.parametrize(
        "pipeline,use_int,forced_fits,expected",
        [
            (False, False, False, 790.66219),
            (False, False, True, 899.707594),
            (False, True, False, 488.674997),
            (False, True, True, 624.654021),
            (True, False, False, 121.89841),
            (True, True, False, 59.805720),
        ]
    )
    def test_calc_eta_metric(
        self,
        pipeline: bool,
        use_int: bool,
        forced_fits: bool,
        expected: float,
        source_instance: vts.Source
    ) -> None:
        """
        Tests the calculation of the eta metric.

        Parametrized for pipeline and query sources, along with using the
        integrated flux and using forced fits.

        Args:
            pipeline: If 'True' then the Source is initialised as a
                pipeline source.
            use_int: Whether to use integrated fluxes instead of peak fluxes.
                Passed to the calc_eta method.
            forced_fits: Whether to use forced fits for the calculation.
                Passed to the calc_eta function.
            expected: The expected eta value for the parametrization.
            source_instance: The pytest source_instance fixture.

        Returns:
            None
        """
        source = source_instance(pipeline=pipeline)

        eta_result = source.calc_eta_metric(
            use_int=use_int,
            forced_fits=forced_fits
        )

        assert eta_result == pytest.approx(expected)

    @pytest.mark.parametrize(
        "pipeline,use_int,forced_fits,expected",
        [
            (False, False, False, 0.940100),
            (False, False, True, 1.611681),
            (False, True, False, 1.021770),
            (False, True, True, 1.582938),
            (True, False, False, 1.765981),
            (True, True, False, 1.740059),
        ]
    )
    def test_calc_v_metric(
        self,
        pipeline: bool,
        use_int: bool,
        forced_fits: bool,
        expected: float,
        source_instance: vts.Source
    ) -> None:
        """
        Tests the calculation of the V metric.

        Parametrized for pipeline and query sources, along with using the
        integrated flux and using forced fits.

        Args:
            pipeline: If 'True' then the Source is initialised as a
                pipeline source.
            use_int: Whether to use integrated fluxes instead of peak fluxes.
                Passed to the calc_v method.
            forced_fits: Whether to use forced fits for the calculation.
                Passed to the calc_v function.
            expected: The expected v value for the parametrization.
            source_instance: The pytest source_instance fixture.

        Returns:
            None
        """
        source = source_instance(pipeline=pipeline)

        v_result = source.calc_v_metric(
            use_int=use_int,
            forced_fits=forced_fits
        )

        assert v_result == pytest.approx(expected)
