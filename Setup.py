import sys
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import shapely
import rasterio as rio
import albumentations as A
import albumentations.augmentations.functional as F
import cv2
import os
from pathlib import Path
from colorama import Fore

#Establishing the Home directory
parent_directory = str(Path(os.getcwd()).parent.parent.parent)
sys.path.append(parent_directory + "/GIT")

from MLtools import create_annotations_mask
from rastertools import raster

'''
2
Generate image patches and clip boulders from good-image-patches
You must first execute Prepare_Graticules as that script requires manual action to setup this code
'''

# Directory of all base assets
completed_mapping = Path(parent_directory + "/fakeboulders/mapping")

# The manually selected, good image patches
ROMs = list(completed_mapping.rglob("*ROM.shp"))

# The pkl files for each good image patch
pkls = [list(ROM.parent.glob("*.pkl"))[0] for ROM in ROMs]

# The associated raster
rasters = [list((ROM.parent.parent.parent / "raster").glob("*.tif"))[0] for ROM in ROMs]

# The boulder shapes to be cut from the raster
boulder_outlines = [list(ROM.parent.glob("*boulder-mapping.shp"))[0] for ROM in ROMs]

if len(ROMs) == len(pkls) and len(pkls) == len(rasters) and len(rasters) == len(boulder_outlines):
    print(Fore.GREEN + "Paths retrieved")
else:
    raise Exception(Fore.RED + "Number of ROM, pkl, raster and boulder outine inputs is not equal.")

    
frames = []
block_width = 500
block_height = 500
dataset_directory = Path(parent_directory + "/fakeboulders")

# read the pickel and rom files, select a pickel file, combine (?) selected tiles
for i, ROM in enumerate(ROMs):
    gdf_rom = gpd.read_file(ROM)
    df_pkl = pd.read_pickle(pkls[i])
    df_pkl_selection = df_pkl[df_pkl.tile_id.isin(gdf_rom.tile_id)]
    frames.append(df_pkl_selection)
    
df_all_tiles = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True))

print(Fore.WHITE + f"Number of manually selected boulder tiles: {df_all_tiles.shape[0]}, with {df_all_tiles.shape[1]} attributes")

# Create a new column/attribute in the data frame
df_all_tiles["dataset"] = "patches-with-boulders"

# tiling of the raster (the new dataset directory)
create_annotations_mask.tiling_raster_from_dataframe(df_all_tiles, dataset_directory, block_width, block_height)

test = df_all_tiles.raster_ap.values
boulder_mapping_col = [list((Path(t).parent.parent / "shp" / "inputs").glob("*boulder-mapping.shp"))[0].as_posix() for t in test]
df_all_tiles["boulder_ap"] = boulder_mapping_col

create_annotations_mask.tiling_boulders_as_shp_from_df(df_all_tiles, dataset_directory, resolution_limit=2) 

'''
4.
Convert boulder outline shapefiles from world to image coordinates
'''

clipped_boulders_p = Path(parent_directory + "/fakeboulders/patches-with-boulders/labels")
clipped_boulders_shps = sorted(list(clipped_boulders_p.glob("*.shp")))

print(Fore.WHITE + "...Listing Clipped Boulder Images...")

'''
At this point we have a dataframe of source tiles each containing their own shape file and associated raster clipped from the source material. We also have a bunch of tiles in the target image onto which we now want to place the clipped features from the source tiles. OBS!

Rules
- I should not select boulders touching the edge of an image patch
- I should also probably use only moon and mars boulders (because of the large differences in resolution)
- Should I extract the average color of a boulder and only select boulders of about the same color as the background? (boulders on Earth are too bright?
'''

# This will throw an error which can be ignored
frames = []

# Iterate through each tile's shapefile, get its associated raster, shove each shape into a dataframe
pd.set_option('mode.chained_assignment', None)
print(Fore.RED + "WARNING: default behavior changed to ignore SettingWithCopyWarning")
for s in clipped_boulders_shps:
    raster_image = (s.parent.parent / "images" / s.stem.replace("_mask", "_image.tif")).as_posix()
    bbox_geom = shapely.geometry.box(*raster.get_raster_bbox(raster_image)).boundary
    gdf_s = gpd.read_file(s)
    idx_touches = gdf_s.geometry.touches(bbox_geom) # if touching edge, do not select
    # Bad practice to ignore warnings, but I've run this code thousands of times with no problems
    gdf_s.loc[:, "is_touching"] = idx_touches
    gdf_s_iw = gdf_s[gdf_s.is_touching == False]
    gdf_s_iw["mask"] = (clipped_boulders_p / s.name).as_posix()
    gdf_s_iw["image"] = raster_image
    frames.append(gdf_s_iw)

gdf_boulders = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True)) # how do you deal with the multiple?

path_to_save = parent_directory + "GeoDataFrame"  # Use an appropriate file extension like ".geojson"

# Save the GeoDataFrame to a file
gdf_boulders.to_file(path_to_save, driver='GeoJSON')

print(Fore.WHITE + f"Number of boulders extracted: {gdf_boulders.shape[0]} with {gdf_boulders.shape[1]} attributes.")

'''
5. 
Read image patches with no boulders
'''

no_boulders_p = Path(parent_directory + "/fakeboulders/no_boulders")

ROMs_nob = list(no_boulders_p.rglob("*empty-patches.shp"))
pkls_nob = [ROM.parent / ROM.name.replace("_empty-patches.shp", "_final-global-tiles.pkl") for ROM in ROMs_nob]
rasters_nob = [ROM.parent.parent / "raster" / ROM.name.replace("_empty-patches.shp", ".tif") for ROM in ROMs_nob]

frames = []
block_width = 500
block_height = 500
dataset_directory = Path(parent_directory + "/fakeboulders")

for i, ROM in enumerate(ROMs_nob):
    gdf_rom = gpd.read_file(ROM)
    df_pkl = pd.read_pickle(pkls_nob[i])
    df_pkl_selection = df_pkl[df_pkl.tile_id.isin(gdf_rom.tile_id)]
    frames.append(df_pkl_selection)
    
df_tiles_no_boulders = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True))

df_tiles_no_boulders["dataset"] = "patches-no-boulders"

# tiling of the raster
create_annotations_mask.tiling_raster_from_dataframe(df_tiles_no_boulders, dataset_directory, block_width, block_height)

row = df_tiles_no_boulders.iloc[0]


print("Setup complete. Client program can now be run to generate fake boulders.")
