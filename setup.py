from setuptools import setup, find_packages
import vasttools

setup(
    name='vasttools',
    url='https://github.com/askap-vast/vasttools/',
    author='Dougal Dobie',
    author_email='ddob1600@uni.sydney.edu.au',
    packages=find_packages(),
    version=vasttools.__version__,
    license='MIT',
    description=('Python module to interact with'
                 ' ASKAP VAST data.'),
    install_requires=[
        "astroML==0.4.1",
        "astropy==4.2",
        "astroquery==0.4.1",
        "bokeh==2.1.1",
        "colorcet==2.0.2",
        "colorlog==4.1.0",
        "dask[dataframe]==2.30.0",
        "forced-phot @ git+https://github.com/askap-vast/forced_phot.git@v0.1.0",
        "matplotlib==3.2.1",
        # TODO Update mocpy when they update release
        "mocpy==0.8.4",
        "multiprocessing_logging==0.3.1",
        "numexpr==2.7.1",
        "numpy==1.19.2",
        "pandas==1.1.4",
        "pyarrow==2.0.0",
        "radio_beam==0.3.2",
        "scipy==1.4.1",
        "tables==3.6.1",
        "tabulate==0.8.7",
        "vaex==3.0.0",
        ],
    scripts=[
        "bin/find_sources.py",
        "bin/build_lightcurves.py",
        "bin/pilot_fields_info.py"
        ],
    include_package_data=True
)
