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
        "astroML>=0.4.1,<0.5",
        "astropy>=3.2,<4.2,!=4.0",
        "astroquery<=0.4.1",
        "bokeh>=2.1.1,<2.2",
        "colorlog>=4.0.2,<5.0",
        "dask[dataframe]<=2.20.0",
        "matplotlib>=3.1.2,<3.3.0",
        "mocpy>=0.8.4",
        "multiprocessing_logging<=0.3.1",
        "numexpr>=2.7.1,<2.8",
        "numpy>=1.17.4,<1.18",
        "pandas<1.1,>=1.0",
        "pyarrow<=0.17.1",
        "radio_beam>=0.3.1,<0.4",
        "scipy>=1.4.0,<1.5.0",
        "tables>=3.6.1,<3.7.0",
        "tabulate>=0.8.6,<0.9",
        ],
    scripts=[
        "bin/find_sources.py",
        "bin/build_lightcurves.py",
        "bin/pilot_fields_info.py"
        ],
    include_package_data=True
)
