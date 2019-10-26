#!/usr/bin/env python

#Example command:

# ./find_racs.py "16:16:00.22 +22:16:04.83" --create-png --imsize 5.0 --png-zscale-contrast 0.1 --png-selavy-overlay --use-combined

import argparse, sys
from astropy.coordinates import Angle
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
import os
import pandas as pd
from astropy.nddata.utils import Cutout2D
from astropy.coordinates import SkyCoord
from astropy.io import fits, ascii
from astropy.wcs import WCS
from astropy.utils.exceptions import AstropyWarning, AstropyDeprecationWarning
import warnings
warnings.filterwarnings('ignore', category=AstropyWarning, append=True)
warnings.filterwarnings('ignore', category=AstropyDeprecationWarning, append=True)

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection
from astropy.visualization import ZScaleInterval,ImageNormalize, simple_norm

class Fields:
    def __init__(self, fname):
        self.fields = ascii.read(fname,format='csv')
        self.direction = SkyCoord(Angle(self.fields["RA_HMS"], unit=u.hourangle), Angle(self.fields["DEC_DMS"], unit=u.deg))

    def find(self, src_dir, max_sep):
        seps = self.direction.separation(src_dir).deg
        within_beam = np.where(seps < max_sep)
        return self.fields[within_beam]

class Source:
    def __init__(self, field, sbid, combined=False):
        self.field = field
        self.sbid = sbid
        
        if combined:
            self.imgname = '%s.fits'%(self.field)
        else:
            self.imgname = 'image.i.SB%s.cont.%s.linmos.taylor.0.restored.fits'%(self.sbid, self.field)
        self.imgpath = os.path.join(IMAGE_FOLDER, self.imgname)
        
        if combined:
            self.selavyname = '%s-selavy.components.txt'%(self.field)
        else:
            self.selavyname = 'selavy-image.i.SB%s.cont.%s.linmos.taylor.0.restored.components.txt'%(self.sbid, self.field)
        self.selavypath = os.path.join(SELAVY_FOLDER, self.selavyname)
        
    
    def make_postagestamp(self, src_coord, size, outfile):
        self.hdu = fits.open(self.imgpath)[0]
        self.wcs = WCS(self.hdu.header, naxis=2)
        
        try:
            self.img_data = self.hdu.data[0,0,:,:]
        except:
            self.img_data = self.hdu.data
        
        self.cutout = Cutout2D(self.img_data, position=src_coord, size=size, wcs=self.wcs)
        
        # Put the cutout image in the FITS HDU
        self.hdu.data = self.cutout.data

        # Update the FITS header with the cutout WCS
        self.hdu.header.update(self.cutout.wcs.to_header())

        # Write the cutout to a new FITS file
        self.hdu.writeto(outfile, overwrite=True)
        
    def extract_source(self, src_coord, crossmatch_radius):
        try:
            with open(self.selavypath, "r") as f:
                lines=f.readlines()

            columns=lines[0].split()[1:-1]
            data=[i.split() for i in lines[2:]]

            self.selavy_cat=pd.DataFrame(data, columns=columns)
        except:
            print('Selavy image does not exist')
            return
        
        self.selavy_sc = SkyCoord(self.selavy_cat['ra_deg_cont'], self.selavy_cat['dec_deg_cont'], unit=(u.deg, u.deg))
        
        #src_coord = SkyCoord("12:00:00", "-30:00:00", unit=(u.hourangle, u.deg))
        
        match_id, match_sep, _dist = src_coord.match_to_catalog_sky(self.selavy_sc)
        
        if match_sep < crossmatch_radius:
            selavy_info = self.selavy_cat.iloc[match_id]
            
            selavy_ra = selavy_info['ra_hms_cont']
            selavy_dec = selavy_info['dec_dms_cont']
            
            selavy_iflux = selavy_info['flux_int']
            selavy_iflux_err = selavy_info['flux_int_err']
            
            print("Source in selavy catalogue %s %s, %s+/-%s mJy (%.0f arcsec offset) "%(selavy_ra, selavy_dec, selavy_iflux, selavy_iflux_err, match_sep.arcsec))
        else:
            print("No selavy catalogue match. Nearest source %.0f arcsec away."%(match_sep.arcsec))
            
        
    def write_ann(self, outfile):
        outfile=outfile.replace(".fits", ".ann")
        with open(outfile, 'w') as f:
            f.write("COORD W\n")
            f.write("PA SKY\n")
            f.write("COLOR GREEN\n")
            f.write("FONT hershey14\n")
            for i,row in self.selavy_cat_cut.iterrows():
                ra = row["ra_deg_cont"]
                dec = row["dec_deg_cont"]
                f.write("ELLIPSE {} {} {} {} {}\n".format(ra, dec,
                float(row["maj_axis"])/3600./2., float(row["min_axis"])/3600./2., float(row["pos_ang"])))
                f.write("TEXT {} {} {}\n".format(ra, dec, self._remove_sbid(row["island_id"])))
                
        print("Wrote annotation file {}.".format(outfile))
        
    def write_reg(self, outfile):
        outfile=outfile.replace(".fits", ".reg")
        with open(outfile, 'w') as f:
            f.write("# Region file format: DS9 version 4.0\n")
            f.write("global color=green font=\"helvetica 10 normal\" select=1 highlite=1 edit=1 move=1 delete=1 include=1 fixed=0 source=1\n")
            f.write("fk5\n")
            for i,row in self.selavy_cat_cut.iterrows():
                ra = row["ra_deg_cont"]
                dec = row["dec_deg_cont"]
                f.write("ellipse({} {} {} {} {})\n".format(ra, dec,
                float(row["maj_axis"])/3600./2., float(row["min_axis"])/3600./2., float(row["pos_ang"])+90.))
                f.write("text({} {} \"{}\")\n".format(ra, dec, self._remove_sbid(row["island_id"])))
                
        print("Wrote region file {}.".format(outfile))
    
    def _remove_sbid(self, island):
        temp = island.split("_")
        new_val = "_".join(temp[-2:])
        return new_val
    
    def filter_selavy_components(self, src_coord, imsize):
        #Filter out selavy components outside field of image
        seps = src_coord.separation(self.selavy_sc)
        mask = seps <= imsize/1.4 #I think cutout2d angle means the width of the image, not a radius hence /2
        #drop the ones we don't need
        self.selavy_cat_cut = self.selavy_cat[mask].reset_index(drop=True)
    
    def make_png(self, src_coord, imsize, selavy, zscale, contrast, outfile, colorbar, pa_corr):
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
            self.img_norms = simple_norm(cutout_data, 'linear')
        im = ax.imshow(cutout_data, norm=self.img_norms, cmap="gray_r")
        if selavy:
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
            patches = [Ellipse((x[i], y[i]), ww[i]*1.1, hh[i]*1.1, 90.+(180.-aa[i])+pa_corr) for i in range(len(x))]
            collection = PatchCollection(patches, fc="None", ec="C1", ls="--", lw=2, transform=ax.get_transform('world'))
            ax.add_collection(collection, autolim=False)
            #Add island labels, haven't found a better way other than looping at the moment.
            for i,val in enumerate(patches):
                ax.annotate(island_names[i], val.center, xycoords=ax.get_transform('world'), annotation_clip=True, color="C0", weight="bold")
        lon = ax.coords[0]
        lat = ax.coords[1]
        lon.set_axislabel("RA")
        lat.set_axislabel("Dec")
        if colorbar:
            cbar = fig.colorbar(im)
            cbar.set_label('mJy')
        plt.savefig(outfile, bbox_inches="tight")
        print("Saved {}".format(outfile))
        plt.clf()

parser=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('coord', metavar="\"HH:MM:SS [+/-]DD:MM:SS\"", type=str, help='Right Ascension and Declination in formnat "HH:MM:SS [+/-]DD:MM:SS", in quotes. E.g. "12:00:00 -20:00:00".')

parser.add_argument('--imsize', type=float, help='Edge size of the postagestamp in arcmin', default=30.)
parser.add_argument('--maxsep', type=float, help='Maximum separation of source from beam centre')
parser.add_argument('--outfile', type=str, help='Name of the output file (or prefix for multiple)')
parser.add_argument('--crossmatch_radius', type=float, help='Crossmatch radius in arcseconds')
parser.add_argument('--use-combined', action="store_true", help='Use the combined mosaics instead.')
parser.add_argument('--img_folder', type=float, help='Path to folder where images are stored')
parser.add_argument('--cat_folder', type=float, help='Path to folder where selavy catalogues are stored')
parser.add_argument('--create-png', action="store_true", help='Create a png of the fits cutout.')
parser.add_argument('--png-selavy-overlay', action="store_true", help='Overlay selavy components onto the png image.')
parser.add_argument('--png-use-zscale', action="store_true", help='Select ZScale normalisation (default is \'sqrt\').')
parser.add_argument('--png-zscale-contrast', type=float, default=0.1, help='Select contrast to use for zscale.')
parser.add_argument('--png-colorbar', action="store_true", help='Add a colorbar to the png plot.')
parser.add_argument('--png-ellipse-pa-corr', type=float, help='Correction to apply to ellipse position angle if needed (in deg). Angle is from x-axis from left to right.', default=0.0)
parser.add_argument('--ann', action="store_true", help='Create a kvis annotation file of the components.')
parser.add_argument('--reg', action="store_true", help='Create a DS9 region file of the components.')


args=parser.parse_args()
ra_str, dec_str = args.coord.split(" ")

imsize = Angle(args.imsize, unit=u.arcmin)
  
max_sep = args.maxsep
if not max_sep:
    max_sep = 1.0

if not args.outfile:
    outfile_prefix = '%s_%s'%(ra_str, dec_str)
    if args.use_combined:
        outfile_prefix+="_combined"
    else:
        outfile_prefix+="_tile"
else:
    outfile_prefix = args.outfile.replace('.fits','')
    
if not args.crossmatch_radius:
    crossmatch_radius = Angle(15,unit=u.arcsec)
else:
    crossmatch_radius = Angle(args.crossmatch_radius,unit=u.arcsec)

IMAGE_FOLDER = args.img_folder
if not IMAGE_FOLDER:
    if args.use_combined:
        IMAGE_FOLDER = '/import/ada1/askap/RACS/aug2019_reprocessing/COMBINED_MOSAICS/I_mosaic_1.0/'
    else:
        IMAGE_FOLDER = '/import/ada1/askap/RACS/aug2019_reprocessing/FLD_IMAGES/stokesI/'

SELAVY_FOLDER = args.cat_folder
if not SELAVY_FOLDER:
    if args.use_combined:
        SELAVY_FOLDER = '/import/ada1/askap/RACS/aug2019_reprocessing/COMBINED_MOSAICS/racs_cat/'
    else:
        SELAVY_FOLDER = '/import/ada1/askap/RACS/aug2019_reprocessing/SELAVY_OUTPUT/stokesI_cat/'
    
src_coord = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg))

fields = Fields("racs_test4.csv")
src_fields = fields.find(src_coord, max_sep)

uniq_fields = list(set(src_fields['FIELD_NAME']))

print(uniq_fields)

if len(uniq_fields) == 0:
    print("Source not in RACS")

elif len(uniq_fields) == 1:
    field_name = src_fields['FIELD_NAME'][0]
    SBID = src_fields['SBID'][0]
    outfile = "%s.fits"%(outfile_prefix)
    print("Source in %s %s"%(SBID, field_name))
    
    source = Source(field_name,SBID, combined=args.use_combined)
    source.make_postagestamp(src_coord, imsize, outfile)
    source.extract_source(src_coord, crossmatch_radius)
    source.filter_selavy_components(src_coord, imsize)
    if args.ann:
        source.write_ann(outfile)
    if args.reg:
        source.write_reg(outfile)
    if args.create_png:
        source.make_png(src_coord, imsize, args.png_selavy_overlay, args.png_use_zscale, args.png_zscale_contrast, outfile, args.png_colorbar, args.png_ellipse_pa_corr)

else:
    completed_fields = []

    for row in src_fields:
        field_name = row['FIELD_NAME']

        if field_name in completed_fields:
            continue
            
        SBID = row['SBID']

        print("Producing data for %s (SB%s)"%(field_name, SBID))
        outfile = "%s_%s.fits"%(outfile_prefix, field_name)

        source = Source(field_name,SBID,combined=args.use_combined)
        source.make_postagestamp(src_coord, imsize, outfile)
        source.extract_source(src_coord, crossmatch_radius)
        #not ideal but line below has to be run after those above
        source.filter_selavy_components(src_coord, imsize)
        if args.ann:
            source.write_ann(outfile)
        if args.reg:
            source.write_reg(outfile)
        if args.create_png:
            source.make_png(src_coord, imsize, args.png_selavy_overlay, args.png_use_zscale, args.png_zscale_contrast, outfile, args.png_colorbar, args.png_ellipse_pa_corr)

        completed_fields.append(field_name)




