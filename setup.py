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
        "pandas>=0.25.3,<0.26",
        "numpy>=1.17.4,<1.18",
        "scipy>=1.4.0,<1.5.0",
        "astropy>=3.2,<4.2,!=4.0",
        "matplotlib>=3.1.2,<4.0",
        "colorlog>=4.0.2,<5.0",
        "dropbox>=9.4.0,<9.5.0",
        "tables>=3.6.1,<3.7.0",
        "radio_beam>=0.3.1,<0.4",
        "tabulate>=0.8.6,<0.9"
        ],
    scripts=[
        "bin/find_sources.py",
        "bin/get_vast_pilot_dbx.py",
        "bin/build_lightcurves.py",
        "bin/pilot_fields_info.py"
        ],
    include_package_data=True
)
