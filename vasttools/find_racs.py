#!/usr/bin/env python

#Example command:

# ./find_racs.py "16:16:00.22 +22:16:04.83" --create-png --imsize 5.0 --png-zscale-contrast 0.1 --png-selavy-overlay --use-combined

import argparse, sys
import numpy as np
import os
import datetime
import pandas as pd
from astropy.coordinates import Angle
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.nddata.utils import Cutout2D
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from astropy.utils.exceptions import AstropyWarning, AstropyDeprecationWarning
import warnings
warnings.filterwarnings('ignore', category=AstropyWarning, append=True)
warnings.filterwarnings('ignore', category=AstropyDeprecationWarning, append=True)

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection
from astropy.visualization import ZScaleInterval, ImageNormalize, PercentileInterval, AsymmetricPercentileInterval
from astropy.visualization import LinearStretch
import matplotlib.axes as maxes
from mpl_toolkits.axes_grid1 import make_axes_locatable

import logging
import logging.handlers
import logging.config

try:
    import colorlog
    use_colorlog=True
except ImportError:
    use_colorlog=False

class Fields:
    def __init__(self, fname):
        self.fields = pd.read_csv(fname)
        self.direction = SkyCoord(Angle(self.fields["RA_HMS"], unit=u.hourangle), Angle(self.fields["DEC_DMS"], unit=u.deg))

    def find(self, src_dir, max_sep, catalog):
        # if len(src_dir) > 1:
        nearest_beams, seps, _d3d = src_dir.match_to_catalog_sky(self.direction)
        within_beam = seps.deg < max_sep
        catalog["sbid"]=self.fields["SBID"].iloc[nearest_beams].values
        catalog["field_name"]=self.fields["FIELD_NAME"].iloc[nearest_beams].values
        catalog["original_index"]=catalog.index.values
        new_catalog = catalog[within_beam].reset_index(drop=True)
        logger.info("RACS field match found for {}/{} sources.".format(len(new_catalog.index),len(nearest_beams)))
        if len(new_catalog.index)-len(nearest_beams) != 0:
            logger.warning("No RACS field matches found for sources with index (or name):")
            for i in range(0, len(catalog.index)):
                if i not in new_catalog["original_index"]:
                    if "name" in catalog.columns:
                        logger.warning(catalog["name"].iloc[i])
                    else:
                        logger.warning("{:03d}".format(i+1))
        else:
            logger.info("All sources found!")
            
        return new_catalog, within_beam
        

class Image:
    def __init__(self, sbid, field, tiles=False):
        self.sbid = sbid
        self.field = field
        
        if tiles:
            self.imgname = 'image.i.SB%s.cont.%s.linmos.taylor.0.restored.fits'%(sbid, field)            
        else:
            self.imgname = '%s.fits'%(field)

        self.imgpath = os.path.join(IMAGE_FOLDER, self.imgname)
        
        self.hdu = fits.open(self.imgpath)[0]
        self.wcs = WCS(self.hdu.header, naxis=2)
        
        try:
            self.data = self.hdu.data[0,0,:,:]
        except:
            self.data = self.hdu.data
            
    def get_rms_img(self):
        self.rmsname = self.imgname.replace('.fits','_rms.fits')

        self.rmspath = os.path.join(BANE_FOLDER, self.rmsname)
        
        self.rms_hdu = fits.open(self.rmspath)[0]
        self.rms_wcs = WCS(self.rms_hdu.header, naxis=2)
        
        try:
            self.rms_data = self.rms_hdu.data[0,0,:,:]
        except:
            self.rms_data = self.rms_hdu.data

class Source:
    def __init__(self, field, sbid, tiles=False, stokesv=False):
        self.field = field
        self.sbid = sbid
        
        if tiles:
            self.selavyname = 'selavy-image.i.SB%s.cont.%s.linmos.taylor.0.restored.components.txt'%(self.sbid, self.field)
        else:
            self.selavyname = '%s-selavy.components.txt'%(self.field)
            if args.stokesv:
                self.nselavyname = 'n%s-selavy.components.txt'%(self.field)
        self.selavypath = os.path.join(SELAVY_FOLDER, self.selavyname)
        if args.stokesv:
            self.nselavypath = os.path.join(SELAVY_FOLDER, self.nselavyname)
        
    
    def make_postagestamp(self, img_data, hdu, wcs, src_coord, size, outfile):

        
        self.cutout = Cutout2D(img_data, position=src_coord, size=size, wcs=wcs)
        
        # Put the cutout image in the FITS HDU
        
        hdu_stamp = fits.PrimaryHDU(data=self.cutout.data)
        
        hdu_stamp.header = hdu.header
        # Update the FITS header with the cutout WCS
        hdu_stamp.header.update(self.cutout.wcs.to_header())

        # Write the cutout to a new FITS file
        hdu_stamp.writeto(outfile, overwrite=True)
        
    def _empty_selavy(self):
        columns = ['island_id', 'component_id', 'component_name', 'ra_hms_cont',
               'dec_dms_cont', 'ra_deg_cont', 'dec_deg_cont', 'ra_err', 'dec_err',
               'freq', 'flux_peak', 'flux_peak_err', 'flux_int', 'flux_int_err',
               'maj_axis', 'min_axis', 'pos_ang', 'maj_axis_err', 'min_axis_err',
               'pos_ang_err', 'maj_axis_deconv', 'min_axis_deconv', 'pos_ang_deconv',
               'maj_axis_deconv_err', 'min_axis_deconv_err', 'pos_ang_deconv_err',
               'chi_squared_fit', 'rms_fit_gauss', 'spectral_index',
               'spectral_curvature', 'spectral_index_err', 'spectral_curvature_err',
               'rms_image', 'has_siblings', 'fit_is_estimate',
               'spectral_index_from_TT', 'flag_c4']
        return pd.DataFrame(np.array([[np.nan for i in range(len(columns))]]), columns=columns)
    
    def extract_source(self, src_coord, crossmatch_radius, stokesv):
        try:
            self.selavy_cat=pd.read_fwf(self.selavypath, skiprows=[1,])
            
            if stokesv:                
                nselavy_cat=pd.read_fwf(self.nselavypath, skiprows=[1,])
                
                nselavy_cat["island_id"]=["n{}".format(i) for i in nselavy_cat["island_id"]]
                nselavy_cat["component_id"]=["n{}".format(i) for i in nselavy_cat["component_id"]]

                self.selavy_cat = self.selavy_cat.append(nselavy_cat, ignore_index=True)
                
        except:
            if not QUIET:
                logger.warning('Selavy image does not exist')
            self.selavy_fail = True
            self.selavy_info = self._empty_selavy()
            return
        
        self.selavy_sc = SkyCoord(self.selavy_cat['ra_deg_cont'], self.selavy_cat['dec_deg_cont'], unit=(u.deg, u.deg))
        
        match_id, match_sep, _dist = src_coord.match_to_catalog_sky(self.selavy_sc)
        
        if match_sep < crossmatch_radius:
            self.selavy_info = self.selavy_cat[self.selavy_cat.index.isin([match_id])]
            
            selavy_ra = self.selavy_info['ra_hms_cont'].iloc[0]
            selavy_dec = self.selavy_info['dec_dms_cont'].iloc[0]
            
            selavy_iflux = self.selavy_info['flux_int'].iloc[0]
            selavy_iflux_err = self.selavy_info['flux_int_err'].iloc[0]
            if not QUIET:
                logger.info("Source in selavy catalogue {} {}, {:.3f}+/-{:.3f} mJy ({:.3f} arcsec offset)".format(selavy_ra, selavy_dec, selavy_iflux, selavy_iflux_err, match_sep[0].arcsec))
        else:
            if not QUIET:
                logger.info("No selavy catalogue match. Nearest source %.0f arcsec away."%(match_sep.arcsec))
            self.selavy_info = self._empty_selavy()
        self.selavy_fail = False
            
        
    def write_ann(self, outfile):
        outfile=outfile.replace(".fits", ".ann")
        neg = False
        with open(outfile, 'w') as f:
            f.write("COORD W\n")
            f.write("PA SKY\n")
            f.write("COLOR GREEN\n")
            f.write("FONT hershey14\n")
            for i,row in self.selavy_cat_cut.iterrows():
                if row["island_id"].startswith("n"):
                    neg = True
                    f.write("COLOR RED\n")
                ra = row["ra_deg_cont"]
                dec = row["dec_deg_cont"]
                f.write("ELLIPSE {} {} {} {} {}\n".format(ra, dec,
                float(row["maj_axis"])/3600./2., float(row["min_axis"])/3600./2., float(row["pos_ang"])))
                f.write("TEXT {} {} {}\n".format(ra, dec, self._remove_sbid(row["island_id"])))
                if neg:
                    f.write("COLOR GREEN\n")
                    neg = False
                
        if not QUIET:
            logger.info("Wrote annotation file {}.".format(outfile))
        
    def write_reg(self, outfile):
        outfile=outfile.replace(".fits", ".reg")
        with open(outfile, 'w') as f:
            f.write("# Region file format: DS9 version 4.0\n")
            f.write("global color=green font=\"helvetica 10 normal\" select=1 highlite=1 edit=1 move=1 delete=1 include=1 fixed=0 source=1\n")
            f.write("fk5\n")
            for i,row in self.selavy_cat_cut.iterrows():
                if row["island_id"].startswith("n"):
                    color = "red"
                else:
                    color = "green"
                ra = row["ra_deg_cont"]
                dec = row["dec_deg_cont"]
                f.write("ellipse({} {} {} {} {}) # color={}\n".format(ra, dec,
                float(row["maj_axis"])/3600./2., float(row["min_axis"])/3600./2., float(row["pos_ang"])+90., color))
                f.write("text({} {} \"{}\") # color={}\n".format(ra, dec, self._remove_sbid(row["island_id"]), color))
                
        if not QUIET:
            logger.info("Wrote region file {}.".format(outfile))
    
    def _remove_sbid(self, island):
        temp = island.split("_")
        new_val = "_".join(temp[-2:])
        if temp[0].startswith("n"):
            new_val="n"+new_val
        return new_val
    
    def filter_selavy_components(self, src_coord, imsize):
        #Filter out selavy components outside field of image
        seps = src_coord.separation(self.selavy_sc)
        mask = seps <= imsize/1.4 #I think cutout2d angle means the width of the image, not a radius hence /2
        #drop the ones we don't need
        self.selavy_cat_cut = self.selavy_cat[mask].reset_index(drop=True)
    
    def make_png(self, src_coord, imsize, selavy, percentile, zscale, contrast, outfile, pa_corr, no_islands=False, label="Source", no_colorbar=False):
        #image has already been loaded to get the fits
        outfile = outfile.replace(".fits", ".png")
        #convert data to mJy in case colorbar is used.
        cutout_data = self.cutout.data*1000.
        #create figure
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(1,1,1, projection=self.cutout.wcs)
        #Get the Image Normalisation from zscale, user contrast.
        if zscale:
            self.img_norms = ImageNormalize(cutout_data, interval=ZScaleInterval(contrast=contrast))
        else:
            self.img_norms = ImageNormalize(cutout_data, interval=PercentileInterval(percentile), stretch=LinearStretch())
        im = ax.imshow(cutout_data, norm=self.img_norms, cmap="gray_r")
        ax.scatter([src_coord.ra.deg], [src_coord.dec.deg], transform=ax.get_transform('world'), marker="x", color="r", zorder=10, label=label)
        if selavy and self.selavy_fail == False:
            ax.set_autoscale_on(False)
            #define ellipse properties for clarity, selavy cut will have already been created.
            ww = self.selavy_cat_cut["maj_axis"].astype(float)/3600.
            hh = self.selavy_cat_cut["min_axis"].astype(float)/3600.
            aa = self.selavy_cat_cut["pos_ang"].astype(float)
            x = self.selavy_cat_cut["ra_deg_cont"].astype(float)
            y = self.selavy_cat_cut["dec_deg_cont"].astype(float)
            island_names = self.selavy_cat_cut["island_id"].apply(self._remove_sbid)
            #Create ellipses, collect them, add to axis.
            #Also where correction is applied to PA to account for how selavy defines it vs matplotlib
            colors = ["C2" if c.startswith("n") else "C1" for c in island_names]
            patches = [Ellipse((x[i], y[i]), ww[i]*1.1, hh[i]*1.1, 90.+(180.-aa[i])+pa_corr) for i in range(len(x))]
            collection = PatchCollection(patches, facecolors="None", edgecolors=colors, linestyle="--", lw=2, transform=ax.get_transform('world'))
            ax.add_collection(collection, autolim=False)
            #Add island labels, haven't found a better way other than looping at the moment.
            if not no_islands:
                for i,val in enumerate(patches):
                    ax.annotate(island_names[i], val.center, xycoords=ax.get_transform('world'), annotation_clip=True, color="C0", weight="bold")
        else:
            if not QUIET:
                logger.warning("PNG: No selavy selected or selavy catalogue failed.")
        ax.legend()
        lon = ax.coords[0]
        lat = ax.coords[1]
        lon.set_axislabel("Right Ascension (J2000)")
        lat.set_axislabel("Declination (J2000)")
        if not no_colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.1, axes_class=maxes.Axes)
            cb = fig.colorbar(im, cax=cax)
            cb.set_label("mJy/beam")
        plt.savefig(outfile, bbox_inches="tight")
        if not QUIET:
            logger.info("Saved {}".format(outfile))
        plt.close()
        
    def get_background_rms(self, rms_img_data, rms_wcs, src_coord):
        pix_coord = np.rint(skycoord_to_pixel(src_coord, rms_wcs)).astype(int)
        rms_val = rms_img_data[pix_coord[0],pix_coord[1]]
        try:
          self.selavy_info['BANE_rms'] = rms_val
        except:
          self.selavy_info = self._empty_selavy()
          self.selavy_info['BANE_rms'] = rms_val

#Force nice
os.nice(5)

runstart = datetime.datetime.now()

parser=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('coords', metavar="\"HH:MM:SS [+/-]DD:MM:SS\" OR input.csv", type=str, help='Right Ascension and Declination in formnat "HH:MM:SS [+/-]DD:MM:SS", in quotes. E.g. "12:00:00 -20:00:00".\
 Degrees is also acceptable, e.g. "12.123 -20.123". Multiple coordinates are supported by separating with a comma (no space) e.g. "12.231 -56.56,123.4 +21.3. Finally you can also\
 enter coordinates using a .csv file. See example file for format.')
 
parser.add_argument('--imsize', type=float, help='Edge size of the postagestamp in arcmin', default=30.)
parser.add_argument('--maxsep', type=float, help='Maximum separation of source from beam centre in degrees.', default=1.0)
parser.add_argument('--out-folder', type=str, help='Name of the output directory to place all results in.', default="find_racs_output_{}".format(runstart.strftime("%Y%m%d_%H:%M:%S")))
parser.add_argument('--source-names', type=str, help='Only for use when entering coordaintes via the command line.\
 State the name of the source being searched. Use quote marks for names that contain a space. For multiple sources separate with a comma with no space, \
 e.g. "SN 1994N,SN 2003D,SN 2019A"', default="")
parser.add_argument('--crossmatch-radius', type=float, help='Crossmatch radius in arcseconds', default=15.0)
parser.add_argument('--use-tiles', action="store_true", help='Use the individual tiles instead of combined mosaics.')
parser.add_argument('--img-folder', type=str, help='Path to folder where images are stored')
parser.add_argument('--rms-folder', type=str, help='Path to folder where image RMS estimates are stored')
parser.add_argument('--cat-folder', type=str, help='Path to folder where selavy catalogues are stored')
parser.add_argument('--create-png', action="store_true", help='Create a png of the fits cutout.')
parser.add_argument('--png-selavy-overlay', action="store_true", help='Overlay selavy components onto the png image.')
parser.add_argument('--png-linear-percentile', type=float, default=99.9, help='Choose the percentile level for the png normalisation.')
parser.add_argument('--png-use-zscale', action="store_true", help='Select ZScale normalisation (default is \'linear\').')
parser.add_argument('--png-zscale-contrast', type=float, default=0.1, help='Select contrast to use for zscale.')
parser.add_argument('--png-no-island-labels', action="store_true", help='Disable island lables on the png.')
parser.add_argument('--png-ellipse-pa-corr', type=float, help='Correction to apply to ellipse position angle if needed (in deg). Angle is from x-axis from left to right.', default=0.0)
parser.add_argument('--png-no-colorbar', action="store_true", help='Do not show the colorbar on the png.')
parser.add_argument('--ann', action="store_true", help='Create a kvis annotation file of the components.')
parser.add_argument('--reg', action="store_true", help='Create a DS9 region file of the components.')
parser.add_argument('--stokesv', action="store_true", help='Use Stokes V images and catalogues. Works with combined images only!')
parser.add_argument('--quiet', action="store_true", help='Turn off non-essential terminal output.')
parser.add_argument('--crossmatch-only', action="store_true", help='Only run crossmatch, do not generate any fits or png files.')
parser.add_argument('--selavy-simple', action="store_true", help='Only include flux density and uncertainty from selavy in returned table.')
parser.add_argument('--debug', action="store_true", help='Turn on debug output.')
parser.add_argument('--no-background-rms', action="store_true", help='Do not estimate the background RMS around each source.')


args=parser.parse_args()

logger = logging.getLogger()
s = logging.StreamHandler()
logformat='[%(asctime)s] - %(levelname)s - %(message)s'

if use_colorlog:
    formatter = colorlog.ColoredFormatter(
        # "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
        "%(log_color)s[%(asctime)s] - %(levelname)s - %(blue)s%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        reset=True,
        log_colors={
        'DEBUG':    'cyan',
        'INFO':     'green',
        'WARNING':  'yellow',
        'ERROR':    'red',
        'CRITICAL': 'red,bg_white',
        },
        secondary_log_colors={},
        style='%'
    )
else:
    formatter = logging.Formatter(logformat, datefmt="%Y-%m-%d %H:%M:%S")

s.setFormatter(formatter)
logger.addHandler(s)

if args.debug:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

# Sort out output directory
output_name = args.out_folder
if os.path.isdir(output_name):
    logger.critical("Requested output directory '{}' already exists! Will not overwrite.".format(output_name))
    logger.critical("Exiting.")
    sys.exit()
else:
    logger.info("Creating directory '{}'.".format(output_name))
    os.mkdir(output_name)

if " " not in args.coords:
    logger.info("Loading file {}".format(args.coords))
    #Give explicit check to file existence
    user_file = os.path.abspath(args.coords)
    if not os.path.isfile(user_file):
        logger.critical("{} not found!")
        sys.exit()
    try:
        catalog = pd.read_csv(user_file, comment="#")
        catalog.columns = map(str.lower, catalog.columns)
        if ("ra" not in catalog.columns) or ("dec" not in catalog.columns):
            logger.critical("Cannot find one of 'ra' or 'dec' in input file.")
            logger.critical("Please check column headers!")
            sys.exit()
        if "name" not in catalog.columns:
            catalog["name"] = ["{}_{}".format(i,j) for i,j in zip(catalog['ra'], catalog['dec'])]
    except:
        logger.critical("Pandas reading of {} failed!".format(args.coords))
        logger.critical("Check format!")
        sys.exit()
else:
    catalog_dict = {'ra':[], 'dec':[]}
    coords = args.coords.split(",")
    for i in coords:    
        ra_str, dec_str = i.split(" ")
        catalog_dict['ra'].append(ra_str)
        catalog_dict['dec'].append(dec_str)
    
    if args.source_names != "":
        source_names = args.source_names.split(",")
        if len(source_names) != len(catalog_dict['ra']):
            logger.critical("All sources must be named when using '--source-names'.")
            logger.critical("Please check inputs.")
            sys.exit()
    else:
        source_names = ["{}_{}".format(i,j) for i,j in zip(catalog_dict['ra'], catalog_dict['dec'])]
    
    catalog_dict['name'] = source_names
    
    catalog = pd.DataFrame.from_dict(catalog_dict)
        
imsize = Angle(args.imsize, unit=u.arcmin)
  
max_sep = args.maxsep

if args.use_tiles:
    outfile_prefix="tile"
else:
    outfile_prefix="combined"
    if args.stokesv:
        outfile_prefix+="_stokesv"

crossmatch_radius = Angle(args.crossmatch_radius,unit=u.arcsec)

if args.stokesv and args.use_tiles:
    logger.critical("Stokes V can only be used with combined mosaics at the moment.")
    logger.critical ("Run again but remove the option '--use-tiles'.")
    sys.exit()

QUIET = args.quiet

IMAGE_FOLDER = args.img_folder
if not IMAGE_FOLDER:
    if args.use_tiles:
        IMAGE_FOLDER = '/import/ada1/askap/RACS/aug2019_reprocessing/FLD_IMAGES/stokesI/'
    else:
        if args.stokesv:
            IMAGE_FOLDER = '/import/ada1/askap/RACS/aug2019_reprocessing/COMBINED_MOSAICS/V_mosaic_1.0/'
        else:
            IMAGE_FOLDER = '/import/ada1/askap/RACS/aug2019_reprocessing/COMBINED_MOSAICS/I_mosaic_1.0/'


SELAVY_FOLDER = args.cat_folder
if not SELAVY_FOLDER:
    if args.use_tiles:
        SELAVY_FOLDER = '/import/ada1/askap/RACS/aug2019_reprocessing/SELAVY_OUTPUT/stokesI_cat/'
    else:
        if args.stokesv:
            SELAVY_FOLDER = '/import/ada1/askap/RACS/aug2019_reprocessing/COMBINED_MOSAICS/racs_catv/'
        else:
            SELAVY_FOLDER = '/import/ada1/askap/RACS/aug2019_reprocessing/COMBINED_MOSAICS/racs_cat/'
            
BANE_FOLDER = args.rms_folder
if not BANE_FOLDER:
    if args.use_tiles:
        BANE_FOLDER = '/import/ada2/ddob1600/RACS_BANE/I_mosaic_1.0_BANE/' #Note: Should run BANE on tile images!!
    else:
        if args.stokesv:
            BANE_FOLDER = '/import/ada2/ddob1600/RACS_BANE/V_mosaic_1.0_BANE/'
        else:
            BANE_FOLDER = '/import/ada2/ddob1600/RACS_BANE/I_mosaic_1.0_BANE/'

if catalog['ra'].dtype == np.float64:
    hms = False
    deg = True
    
elif ":" in catalog['ra'].iloc[0]:
    hms = True
    deg = False
else:
    deg = True
    hms = False

if hms:
    src_coords = SkyCoord(catalog['ra'], catalog['dec'], unit=(u.hourangle, u.deg))
else:
    src_coords = SkyCoord(catalog['ra'], catalog['dec'], unit=(u.deg, u.deg))

logger.info("Finding RACS fields for sources...")
fields = Fields("racs_test4.csv")
src_fields, coords_mask = fields.find(src_coords, max_sep, catalog)

src_coords = src_coords[coords_mask]

uniq_fields = src_fields['field_name'].unique().tolist()

if len(uniq_fields) == 0:
    logger.error("Source(s) not in RACS!")
    sys.exit()

crossmatch_output_check = False

if QUIET:
    logger.info("Performing crossmatching for sources, please wait...")

for uf in uniq_fields:
    if not QUIET:
        logger.info("-------------------------------------------------------------")
        logger.info("Starting Field {}".format(uf))
        logger.info("-------------------------------------------------------------")
    mask = src_fields["field_name"]==uf
    srcs = src_fields[mask]
    indexes = srcs.index
    srcs = srcs.reset_index()
    field_src_coords = src_coords[mask]
    image = Image(srcs["sbid"].iloc[0], uf, tiles=args.use_tiles)
    
    if not args.no_background_rms:
      image.get_rms_img()
    
    for i,row in srcs.iterrows():
        field_name = uf
            
        SBID = row['sbid']
        
        number = row["original_index"]+1
        
        label = row["name"]

        if not QUIET:
            logger.info("Searching for crossmatch to source {}".format(label))

        outfile = "{}_{}_{}.fits".format(label.replace(" ", "_"), field_name, outfile_prefix)
        outfile = os.path.join(output_name, outfile)

        source = Source(field_name,SBID,tiles=args.use_tiles, stokesv=args.stokesv)
        src_coord = field_src_coords[i]
        if not args.crossmatch_only:
            source.make_postagestamp(image.data, image.hdu, image.wcs, src_coord, imsize, outfile)
              
        source.extract_source(src_coord, crossmatch_radius, args.stokesv)
        if not args.no_background_rms:
            source.get_background_rms(image.rms_data, image.rms_wcs, src_coord)
        
        #not ideal but line below has to be run after those above
        if source.selavy_fail == False:
            source.filter_selavy_components(src_coord, imsize)
            if args.ann:
                source.write_ann(outfile)
            if args.reg:
                source.write_reg(outfile)
        else:
            if not QUIET:
                logger.error("Selavy failed! No region or annotation files will be made if requested.")
        if args.create_png and not args.crossmatch_only:
            source.make_png(src_coord, imsize, args.png_selavy_overlay, args.png_linear_percentile, args.png_use_zscale, 
                args.png_zscale_contrast, outfile, args.png_ellipse_pa_corr, no_islands=args.png_no_island_labels, label=label, no_colorbar=args.png_no_colorbar)
        if not crossmatch_output_check:
            crossmatch_output = source.selavy_info
            crossmatch_output.index = [indexes[i]]
            crossmatch_output_check = True
        else:
            temp_crossmatch_output = source.selavy_info
            temp_crossmatch_output.index = [indexes[i]]
            crossmatch_output = crossmatch_output.append(source.selavy_info)
        if not QUIET:
            logger.info("-------------------------------------------------------------")

runend = datetime.datetime.now()
runtime = runend-runstart
logger.info("-------------------------------------------------------------")
logger.info("Summary")
logger.info("-------------------------------------------------------------")
logger.info("Number of sources searched for: {}".format(len(catalog.index)))
logger.info("Number of sources in RACS: {}".format(len(src_fields.index)))
logger.info("Number of sources with matches < {} arcsec: {}".format(crossmatch_radius.arcsec, len(crossmatch_output[~crossmatch_output["island_id"].isna()].index)))
logger.info("Processing took {:.1f} minutes.".format(runtime.seconds/60.))
#Create and write final crossmatch csv
if args.selavy_simple:
  crossmatch_output = crossmatch_output.filter(items=["flux_int","rms_image","BANE_rms"])
  crossmatch_output = crossmatch_output.rename(columns={"flux_int":"S_int", "rms_image":"S_err"})
final = src_fields.join(crossmatch_output)

output_crossmatch_name = "{}_racs_crossmatch.csv".format(output_name)
output_crossmatch_name = os.path.join(output_name, output_crossmatch_name)
final.to_csv(output_crossmatch_name, index=False)
logger.info("Written {}.".format(output_crossmatch_name))
logger.info("All results in {}.".format(output_name))

