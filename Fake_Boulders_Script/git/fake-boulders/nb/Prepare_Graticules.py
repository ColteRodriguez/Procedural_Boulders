# Importing Required Modules
import sys
import os
from pathlib import Path
from MLtools import create_annotations_mask
from rastertools import raster

#Establishing the Home directory
parent_directory = Path(os.getcwd()).parent.parent.parent
sys.path.append(str(parent_directory) + "/GIT")

'''
1
Generate graticules (global tiles) on the source (mapped) and target (empty) rasters
These 500px graticules (patches) will be selected from the target raster to paste boulders boulders from the source raster
'''
# Generate Graticules for target (unmapped) raster
rasters_p = Path(str(parent_directory) + "/fakeboulders/no_boulders/raster")
rasters = list(rasters_p.glob("*.tif"))
for r in rasters:
    filename = r.parent.parent / "shp" / (r.stem + "-global-tiles.shp")
    create_annotations_mask.generate_graticule_from_raster(r, 500, 500, filename, stride=(0,0))
    
# Generate Graticules for source (mapped) raster
rasters_p = Path(str(parent_directory) + "/fakeboulders/mapping/raster/")
rasters = list(rasters_p.glob("*.tif"))
for r in rasters:
    filename = r.parent.parent / "shp" / "inputs" / (r.stem + "-global-tiles.shp")
    create_annotations_mask.generate_graticule_from_raster(r, 500, 500, filename, stride=(0,0))


print("***Raster graticules generated successfully. Procceed by opening each input raster and global tiles shp file in GIS, manually select tiles with no boulders, and save them as {path to Fake_Boudlers_parent/fake_boulders/no_boulders/shp/} + raster_id-empty-patches.shp. Also manually select tiles with boulders and save them as {path to Fake_Boudlers_parent/mapping/shp/inputs} + raster_id-ROM.shp***")
