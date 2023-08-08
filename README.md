# INSTRUCTIONS FOR USAGE AND TROUBLESHOOTING

This code is a niche, backend only script which requires lost of external packages. The code must be run in the following steps:

1. Generate_Graticules.py
2. Setup.py
3. Client.py

For a given set of inputs (a sample input is given in the dependencies folder) you may complete steps 1 and 2 once, and run the client as
many times as you want for the input. However, changing the inputs (adding more rasters, shapefiles for the client to choose from) requires
that you rerun generate_graticules and Setup.

This program relies on a specific file stack with meticulously named folders. I have attempted to configure the
github repository with this stack. However, in case this does not work, it is given bellow, assuming a base directory.

Base:
  Fake_Boulders_Script:
    git:
      fake-boulders:
        nb:
          Generate_Graticules.py
          Setup.py
          fbapi.py
          Client.py          
    fakeboulders:
      patches-with-boulders:
        labels:
        images:
      patches-no-boulders:
        labels:
        images:
      no_boulders:
        shp:
        raster:
      mapping:
        shp:
        raster:
        
      
  


