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

IMAGE_FOLDER = '/import/ada1/askap/RACS/test4-jul19-images/stokesI'
SELAVY_FOLDER = '/import/ada1/askap/RACS/test4-jul19-images/selavy-output'

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
        
        print(self.field, self.sbid)
        
    
    def make_postagestamp(self, src_coord, size, outfile):
        

        self.filename = 'image.i.SB%s.cont.%s.linmos.taylor.0.restored.fits'%(self.sbid, self.field)
        self.filepath = '%s/%s'%(IMAGE_FOLDER, self.filename)
        
        self.hdu = fits.open(self.filepath)[0]
        self.wcs = WCS(self.hdu.header, naxis=2)
        
        cutout = Cutout2D(self.hdu.data[0,0,:,:], position=src_coord, size=size, wcs=self.wcs)
        
        # Put the cutout image in the FITS HDU
        self.hdu.data = cutout.data

        # Update the FITS header with the cutout WCS
        self.hdu.header.update(cutout.wcs.to_header())

        # Write the cutout to a new FITS file
        self.hdu.writeto(outfile, overwrite=True)
        
    def extract_source(self, crossmatch_radius):
        selavyfile = 'selavy-image.i.SB%s.cont.%s.linmos.taylor.0.restored.components.txt'%(self.sbid, self.field)
        selavypath = '%s/%s'%(SELAVY_FOLDER, selavyfile)
        
        selavy_cat = ascii.read(selavypath)
        
        print(selavy_cat)
        
        
        



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


args=parser.parse_args()

ra_str = str(args.ra)
dec_str = str(args.dec)


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

src_coord = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg))

fields = Fields("racs_test4.csv")
src_fields = fields.find(src_coord, max_sep)

sbid = src_fields['SBID'][0]
field_name = src_fields['FIELD_NAME'][0]

source = Source(field_name,sbid)

outfile = "%s_%s.fits"%(outfile_prefix, field_name)

source.make_postagestamp(src_coord, imsize, outfile)
source.extract_source(30)

exit()






uniq_fields = list(set(src_fields['FIELD_NAME']))

if len(uniq_fields) == 0:
    print("Source not in RACS")

elif len(uniq_fields) == 1:
    field_name = src_fields['FIELD_NAME'][0]
    SBID = src_fields['SBID'][0]
    
    source = Source(field_name,SBID)
    source.make_postagestamp(src_coord, imsize, "%s.fits"%(outfile_prefix))

else:
    completed_fields = []

    for row in src_fields:
        field_name = row['FIELD_NAME']

        if field_name in completed_fields:
            continue

        SBID = row['SBID']

        outfile = "%s_%s.fits"%(outfile_prefix, field_name)

        pstamp = Postagestamp(field_name,SBID)

        pstamp.make_postagestamp(src_coord, imsize, outfile)

        completed_fields.append(field_name)




