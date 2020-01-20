import os
from timeit import default_timer

os.nice(5)

IMAGE_FOLDERS = [
    ('/import/ada1/askap/RACS/aug2019_reprocessing/'
        'COMBINED_MOSAICS/V_mosaic_1.0/'),
    ('/import/ada1/askap/RACS/aug2019_reprocessing/'
        'COMBINED_MOSAICS/I_mosaic_1.0/')]

BANE_BASE = '/import/ada2/ddob1600/RACS_BANE'

for image_folder in IMAGE_FOLDERS:
    bane_folder = image_folder.replace(
        '/import/ada1/askap/RACS/aug2019_reprocessing/COMBINED_MOSAICS',
        BANE_BASE).replace(
        '_mosaic_1.0',
        '_mosaic_1.0_BANE')
    if not os.path.isdir(bane_folder):
        os.mkdir(bane_folder)
    os.chdir(bane_folder)

    stokes_folder = os.path.split(os.path.split(image_folder)[0])[1]

    for fits_image in os.listdir(image_folder):
        start = default_timer()
        fits_path = os.path.join(image_folder, fits_image)
        bane_save = os.path.join(bane_folder, fits_image).replace('.fits', '')

        print(fits_path)
        print(bane_save)

        BANE_command = "BANE %s --out=%s" % (fits_path, bane_save)
        os.system(BANE_command)
        end = default_timer()

        print(end - start)

    # print(os.listdir(image_folder))

#  os.system("BANE %s")
