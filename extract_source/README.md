This script will let you make a square postagestamp of a source of interest in RACS. This will only work on `ada`.

To use the script simply run `./find_racs.py HH:MM:SS DD:MM:SS [--imsize --maxsep --outfile]`

`--imsize` is the edge size of the postagestamp in arcmin.

`--maxsep` is the maximum separation of source from beam centre.

`--outfile` is the name of the output file (or prefix for sources that occur in multiple images).

`--crossmatch_radius` is the crossmatch radius in arcseconds for selavy source extraction.

`--img_folder` is the path to folder where images are stored on ada.

`--cat_folder` is the path to folder where selavy catalogues are stored.

Note: some sources appear in more than one RACS field. In this case the script will save multiple files in the format `outfile_FIELDNAME.fits`.

More improvements are coming, feel free to contribue.
