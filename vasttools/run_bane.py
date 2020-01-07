import os
from timeit import default_timer

os.nice(5)

#IMAGE_FOLDERS = ['/import/ada1/askap/RACS/aug2019_reprocessing/COMBINED_MOSAICS/V_mosaic_1.0/','/import/ada1/askap/RACS/aug2019_reprocessing/COMBINED_MOSAICS/I_mosaic_1.0/']
IMAGE_FOLDERS = ["/import/ada1/askap/PILOT/release/EPOCH02/COMBINED/STOKESI_IMAGES", "/import/ada1/askap/PILOT/release/EPOCH02/COMBINED/STOKESV_IMAGES"]

#BANE_BASE = '/import/ada2/ddob1600/RACS_BANE'
BANE_BASE = '/import/ada2/ddob1600/BANE_PILOT/EPOCH02'

for image_folder in IMAGE_FOLDERS:
  #bane_folder = image_folder.replace('/import/ada1/askap/RACS/aug2019_reprocessing/COMBINED_MOSAICS',BANE_BASE).replace('_mosaic_1.0','_mosaic_1.0_BANE')
  bane_folder = image_folder.replace('/import/ada1/askap/PILOT/release/EPOCH02/COMBINED',BANE_BASE).replace('_mosaic_1.0','_mosaic_1.0_BANE')
  
  if not os.path.isdir(bane_folder):
    os.mkdir(bane_folder)
  os.chdir(bane_folder)
  
  stokes_folder = os.path.split(os.path.split(image_folder)[0])[1]
  
  for fits_image in os.listdir(image_folder):
    start = default_timer()
    fits_path = os.path.join(image_folder,fits_image)
    bane_save = os.path.join(bane_folder,fits_image).replace('.fits','')
    
    if os.path.isfile(bane_save+'_bkg.fits') and os.path.isfile(bane_save+'_rms.fits'):
      continue
    
    print(bane_save)
    BANE_command = "BANE %s --out=%s"%(fits_path,bane_save)
    #os.system(BANE_command)
    end = default_timer()
    
    print(end-start)
