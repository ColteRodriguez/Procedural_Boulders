# INSTRUCTIONS FOR USAGE AND TROUBLESHOOTING

Note the required modules in MLtools and rastertools. To avoid racking up download space, I suggest using a conda environment and running the script in Jupyterlab. 

Required Modules (not including MLtools and rastertools)
- sys
- math
- random
- numpy
- matplotlib
- pandas
- geopandas
- shapely
- rasterio 
- albumentations
- cv2
- os
- pathlib
- colorama (optional)


This code is a backend script which requires lost of external packages. The code must be run in the following steps:

1. Generate_Graticules.py
2. Setup.py
3. Client.py

For a given set of inputs (a sample input is given in the dependencies folder) you may complete steps 1 and 2 once, and run the client as
many times as you want for the input. However, changing the inputs (adding more rasters, shapefiles for the client to choose from) requires
that you rerun generate_graticules and Setup.

This program relies on a specific file stack with meticulously named folders. Becasue you can not configure github repo with empty folders, I have given it bellow, assuming a base directory.

Base:
    Fake_Boulders_Script:
        git:
            fake-boulders:
                nb:
                    Generate_Graticules.py
                    Setup.py
                    fbapi.py
                    Client.py
                    MLtools: (directory from https://github.com/yellowchocobo/MLtools)
                    rastertools: (directory from https://github.com/yellowchocobo/rastertools)
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
        
      
  


