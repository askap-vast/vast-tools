import numpy as np
from astropy.nddata.utils import Cutout2D
from astropy.io import fits, ascii
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from astropy.coordinates import SkyCoord
from astropy import units as u

def get_clipped_rms(im_data, sigma_clip):
    std_dev = np.std(im_data)
    sigma_clipped = im_data[im_data < sigma_clip*std_dev]
    rms = np.std(sigma_clipped)
    
    return rms
    
def get_upper_lim(src_coord, data, wcs, size, sigma_clip=3):
    cutout = Cutout2D(data, position=src_coord, size=size, wcs=wcs)
    rms = get_clipped_rms(cutout.data, sigma_clip)
    
    pix_coord = np.rint(skycoord_to_pixel(src_coord, wcs)).astype(int)
    
    flux_val = data[pix_coord[0],pix_coord[1]]
    
    return flux_val, rms
    
def load_image(imgpath):
    hdu = fits.open(imgpath)[0]
    naxis = hdu.header['NAXIS']
    data = hdu.data
    
    if naxis == 4:
        data_shape = data.shape
        if data_shape[0] == 1:
            data = data[0,0,:,:]
        else:
            data = data[:,:,0,0]
    
    wcs = WCS(hdu.header, naxis=2)
    
    return data, wcs
    
imgpath = '/import/ada2/vlass/VLASS1.1/T01t01/VLASS1.1.ql.T01t01.J001221-363000.10.2048.v1/VLASS1.1.ql.T01t01.J001221-363000.10.2048.v1.I.iter1.image.pbcor.tt0.subim.fits'

data, wcs = load_image(imgpath)
sc = SkyCoord(3.5,36.9, unit=u.deg)

sc = SkyCoord('00:12:21','-36:30:10',unit=(u.hourangle, u.deg))

peak, rms = get_upper_lim(sc, data, wcs, 1*u.arcmin)
print(peak, rms)
