#!/usr/bin/env python

import argparse, sys
from astropy.coordinates import Angle
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import ascii
import numpy as np

from astropy.nddata.utils import Cutout2D
from astropy.coordinates import SkyCoord
from astropy.io import fits, ascii
from astropy.wcs import WCS

class Fields:
    def __init__(self, fname):
        self.fields = ascii.read(fname,format='csv')
        self.direction = SkyCoord(Angle(self.fields["RA_HMS"], unit=u.hourangle), Angle(self.fields["DEC_DMS"], unit=u.deg))

    def find(self, src_dir, max_sep):
        seps = self.direction.separation(src_dir).deg
        within_beam = np.where(seps < max_sep)
        return self.fields[within_beam]

class Source:
    def __init__(self, field, sbid):
        self.field = field
        self.sbid = sbid
        
        self.imgname = 'image.i.SB%s.cont.%s.linmos.taylor.0.restored.fits'%(self.sbid, self.field)
        self.imgpath = '%s/%s'%(IMAGE_FOLDER, self.imgname)
        
        self.selavyname = 'selavy-image.i.SB%s.cont.%s.linmos.taylor.0.restored.components.txt'%(self.sbid, self.field)
        self.selavypath = '%s/%s'%(SELAVY_FOLDER, self.selavyname)
        
    
    def make_postagestamp(self, src_coord, size, outfile):
        self.hdu = fits.open(self.imgpath)[0]
        self.wcs = WCS(self.hdu.header, naxis=2)
        
        cutout = Cutout2D(self.hdu.data[0,0,:,:], position=src_coord, size=size, wcs=self.wcs)
        
        # Put the cutout image in the FITS HDU
        self.hdu.data = cutout.data

        # Update the FITS header with the cutout WCS
        self.hdu.header.update(cutout.wcs.to_header())

        # Write the cutout to a new FITS file
        self.hdu.writeto(outfile, overwrite=True)
        
    def extract_source(self, src_coord, crossmatch_radius):
        try:
            self.selavy_cat = ascii.read(self.selavypath)
        except:
            print('Selavy image does not exist')
            return
        
        self.selavy_sc = SkyCoord(self.selavy_cat['col4'], self.selavy_cat['col5'], unit=(u.hourangle, u.deg))
        
        #src_coord = SkyCoord("12:00:00", "-30:00:00", unit=(u.hourangle, u.deg))
        
        match_id, match_sep, _dist = src_coord.match_to_catalog_sky(self.selavy_sc)
        
        if match_sep < crossmatch_radius:
            selavy_info = self.selavy_cat[match_id]
            
            selavy_ra = selavy_info['col4']
            selavy_dec = selavy_info['col5']
            
            selavy_iflux = selavy_info['col13']
            selavy_iflux_err = selavy_info['col14']
            
            print("Source in selavy catalogue %s %s, %s+/-%s mJy (%.0f arcsec offset) "%(selavy_ra, selavy_dec, selavy_iflux, selavy_iflux_err, match_sep.arcsec))
        else:
            print("No selavy catalogue match. Nearest source %.0f arcsec away."%(match_sep.arcsec))
            
        
neg_dec = False

parser=argparse.ArgumentParser()

parser.add_argument('ra', metavar='HH:MM:SS', type=str, help='Right Ascension in HH:MM:SS')

#argparse doesn't play nice with '-' inside arguments
dec = sys.argv[2]
if dec.startswith('-'):
    neg_dec = True
    sys.argv[2] = dec[1:]
  
parser.add_argument('dec', metavar='DD:MM:SS', type=str, help='Declination in DD:MM:SS')

parser.add_argument('--imsize', type=float, help='Edge size of the postagestamp in arcmin')
parser.add_argument('--maxsep', type=float, help='Maximum separation of source from beam centre')
parser.add_argument('--outfile', type=float, help='Name of the output file (or prefix for multiple)')
parser.add_argument('--crossmatch_radius', type=float, help='Crossmatch radius in arcseconds')
parser.add_argument('--img_folder', type=float, help='Path to folder where images are stored')
parser.add_argument('--cat_folder', type=float, help='Path to folder where selavy catalogues are stored')


args=parser.parse_args()

ra_str = str(args.ra)
dec_str = str(args.dec)
if neg_dec:
    dec_str = '-'+dec_str


if not args.imsize:
    imsize = Angle(30, unit=u.arcmin)
else:
    imsize = Angle(args.imsize, unit=u.arcmin)
  
max_sep = args.maxsep
if not max_sep:
    max_sep = 1.0

if not args.outfile:
    outfile_prefix = '%s_%s'%(ra_str, dec_str)
else:
    outfile_prefix = args.outfile.replace('.fits','')
    
if not args.crossmatch_radius:
    crossmatch_radius = Angle(15,unit=u.arcsec)
else:
    crossmatch_radius = Angle(args.crossmatch_radius,unit=u.arcsec)

IMAGE_FOLDER = args.img_folder
if not IMAGE_FOLDER:
    IMAGE_FOLDER = '/import/ada1/askap/RACS/test4-jul19-images/stokesI'    

SELAVY_FOLDER = args.cat_folder
if not SELAVY_FOLDER:
    SELAVY_FOLDER = '/import/ada1/askap/RACS/test4-jul19-images/selavy-output'
    
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
    
    print("Source in %s %s"%(SBID, field_name))
    
    source = Source(field_name,SBID)
    source.make_postagestamp(src_coord, imsize, "%s.fits"%(outfile_prefix))
    source.extract_source(src_coord, crossmatch_radius)

else:
    completed_fields = []

    for row in src_fields:
        field_name = row['FIELD_NAME']

        if field_name in completed_fields:
            continue
            
        SBID = row['SBID']

        print("Producing data for %s (SB%s)"%(field_name, SBID))
        outfile = "%s_%s.fits"%(outfile_prefix, field_name)

        source = Source(field_name,SBID)
        source.make_postagestamp(src_coord, imsize, outfile)
        source.extract_source(src_coord, crossmatch_radius)

        completed_fields.append(field_name)




