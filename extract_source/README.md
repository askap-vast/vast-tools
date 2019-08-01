This script will let you make a square postagestamp of a source of interest in RACS. This will only work on `ada`.

To use the script simply run `./find_racs.py HH:MM:SS DD:MM:SS [--imsize --maxsep --outfile]`

`--imsize` is the postagestamp side length in arcminutes (default 30 arcmin)
`--maxsep` is the maximum separation of the source from the beam centre in degrees (default 1 degree)
`--outfile` lets you set the name of the postagestamp file

Note: some sources appear in more than one RACS field. In this case the script will save multiple files in the format `outfile_FIELDNAME.fits`.
