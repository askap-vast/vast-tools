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
from astropy.table import Table, Column, MaskedColumn, hstack, vstack
import astropy.table as table

from astropy.utils.exceptions import AstropyWarning, AstropyDeprecationWarning

import warnings
import logging

warnings.filterwarnings('ignore', category=AstropyWarning, append=True)
warnings.filterwarnings('ignore', category=AstropyDeprecationWarning, append=True)


def find_field(fields_csv, cat_coords, max_sep):
    beam_info = ascii.read(fields_csv, format='csv')
    
    beam_direction = SkyCoord(beam_info["RA_HMS"], beam_info["DEC_DMS"], unit=(u.hourangle, u.deg))
    
    nearest_beams, beam_sep, _d3d = cat_coords.match_to_catalog_sky(beam_direction)
    
    #within_beams = np.where(beam_sep < max_sep) #which sources are within a beam?
    
    mask = beam_sep > max_sep
    
    beam_ids = nearest_beams#[within_beams]
    
    SBIDs = MaskedColumn(beam_info["SBID"][beam_ids], mask=mask)
    field_names = MaskedColumn(beam_info["FIELD_NAME"][beam_ids], mask=mask)
    
    match_table = Table(masked=True)
    
    #match_table['SOURCE_ID'] = within_beams[0]
    match_table.add_column(SBIDs)
    match_table.add_column(field_names)
    
    return match_table
    
def add_RACS_info(cat_table, match_table):
    cat_table = hstack([cat_table, match_table])
    
    return cat_table
    
    
def load_cat(fname = 'test_cat.csv'):
    cat = ascii.read(fname, format='csv')
    
    return cat

def load_image(imgpath):
    hdu = fits.open(imgpath)[0]
    wcs = WCS(hdu.header, naxis=2)
    
    return hdu, wcs

def make_images(field, imgpath, size):
    full_hdu, full_wcs = load_image(imgpath)
    full_header = full_hdu.header
    img_data = full_hdu.data[0,0,:,:]
    
    for source in field:
        src_coord = SkyCoord(source['ra'], source['dec'], unit=(u.hourangle, u.deg))

        cutout = Cutout2D(img_data, position=src_coord, size=size, wcs=full_wcs)
        
        if 'name' in field.keys():
            outfile = '%s.fits'%source['name']
        else:
            outfile = '%s.fits'%(src_coord.to_string(style='hmsdms'))
        
        # Put the cutout image in the FITS HDU
        hdu = fits.PrimaryHDU(data=cutout.data)
        
        hdu.header = full_header
        # Update the FITS header with the cutout WCS
        hdu.header.update(cutout.wcs.to_header())

        # Write the cutout to a new FITS file
        hdu.writeto(outfile, overwrite=True)

def get_selavy_info(field, selavypath, crossmatch_radius):
    try:
        selavy_info = table.Table.read(selavypath,format='ascii')
    except:
        return
        
    header, units = selavy_info.meta['comments']
    selavy_info.rename_columns(selavy_info.colnames, header.split()[:-1])
    
    selavy_ra = selavy_info['ra_hms_cont']
    selavy_dec = selavy_info['dec_dms_cont']
    
    selavy_sc = SkyCoord(selavy_ra, selavy_dec, unit=(u.hourangle, u.deg))
    
    src_coords = SkyCoord(field['ra'], field['dec'], unit=(u.hourangle, u.deg))
    
    match_id, match_sep, _dist = src_coords.match_to_catalog_sky(selavy_sc)
    
    true_match = match_sep <= crossmatch_radius
    
    true_match_id = match_id[true_match]
       
    racs_info = Table(selavy_info[match_id], masked=True)
    
    for colname in racs_info.colnames:
        racs_info[colname].mask = ~true_match
    
    return racs_info
    

def search_catalogue(cat_table, imsize, crossmatch_radius, IMGFOLDER=None, IMGNAME=None, SELAVYFOLDER=None, SELAVYNAME=None, outputfile=None):
    if not IMGFOLDER:
        IMGFOLDER = '/import/ada1/askap/RACS/test4-jul19-images/stokesI'
        
    if not SELAVYFOLDER:
        SELAVYFOLDER = '/import/ada1/askap/RACS/test4-jul19-images/selavy-output'
    
    if not IMGNAME:
        IMGNAME = 'image.i.SB%s.cont.%s.linmos.taylor.0.restored.fits'
    
    if not SELAVYNAME:
        SELAVYNAME = 'selavy-image.i.SB%s.cont.%s.linmos.taylor.0.restored.components.txt'
    
    masked = cat_table.mask['FIELD_NAME']
    
    cat_by_field = cat_table[~masked].group_by('FIELD_NAME')
    
    field_grouping = cat_by_field.groups
    
    if 'name' in cat_table.keys():
        do_name_col = True
    else:
        do_name_col = False
    
    field_tables = []
    for field in field_grouping:
        sbid = field[0]['SBID']
        field_name = field[0]['FIELD_NAME']
    
        imgname = IMGNAME % (sbid, field_name)
        imgpath = '%s/%s'%(IMGFOLDER, imgname)
        
        selavyname = SELAVYNAME % (sbid, field_name)
        selavypath = '%s/%s'%(SELAVYFOLDER, selavyname)
        
        make_images(field, imgpath, imsize)
        
        selavy_info = get_selavy_info(field, selavypath, crossmatch_radius)
        
        if not selavy_info:
            continue
        
        field_table = hstack([field, selavy_info])
        field_tables.append(field_table)
    
    full_info = vstack(field_tables)
    
    print(full_info)
    
    if outputfile:
        ascii.write(full_info, outputfile)

if __name__ == '__main__':
    test_cat = load_cat()
    cat_coords = SkyCoord(test_cat['ra'], test_cat['dec'], unit=(u.hourangle, u.deg))
    match_table = find_field('racs_test4.csv', cat_coords, 1*u.deg)

    cat_table = add_RACS_info(test_cat, match_table)

    search_catalogue(cat_table, Angle(30, unit=u.arcmin), Angle(5, unit=u.arcsec), outputfile='source_table.dat')

