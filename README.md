# AVISO Fetching and plotting routines

- `fetch.py` is a generic FTP fetching script to pull all the files in a particular directory

# Mesoscale Eddy Product

- `fetch.eddy.py` grabs Mesoscale Eddy Product NetCDF files from AVISO, if needed.
- `plot.eddy.py` Plot a particular day's eddies, both cyclonic and anticyclonic.
- `Makefile.eddy` generates a movie of the eddies. To invoke use `make -f Makefile.eddy -j8 SDATE=2019-03-01 EDATE=2019-05-01`

# The EEZ data is from [Mariner Regions](https://www.marineregions.org/downloads.php)
