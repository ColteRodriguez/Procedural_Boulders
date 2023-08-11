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


'''
6.
Defining the AUgmentation Pipeline
'''

# transform is "composed" of "one of" the 8 transformations, and affien 
# and a transpose. Finally an affine transform and a blur is applied
transform = A.Compose([
    
    A.OneOf([A.NoOp(p=1.0), 
             A.Affine(p=1.0, rotate=90.0),
             A.Affine(p=1.0, rotate=180.0), 
             A.Affine(p=1.0, rotate=270.0),
             A.HorizontalFlip(p=1.0), 
             A.VerticalFlip(p=1.0),
             A.Transpose(p=1.0), 
             A.Compose([A.Affine(p=1.0, rotate=180.0), A.Transpose(p=1.0)])]),
             A.Affine(scale=(1.0,1.0), translate_px=(-128,128), interpolation=2, mask_interpolation=2, 
             mode=0, cval=0, cval_mask=0, keep_ratio=True, p=1.0)])


# Modified from "Producing 2d Perlin Noise with numpy." Inquiry by user:4303737, answered by user:7207392. https://stackoverflow.com/questions/42147776/producing-2d-perlin-noise-with-numpy/42154921#42154921
def generate_target_noise(rim_distance, target, is_mare):
    def perlin(x, y, seed=0):
        # permutation table
        np.random.seed(seed)
        p = np.arange(256, dtype=int)
        np.random.shuffle(p)
        p = np.stack([p, p]).flatten()
        # coordinates of the top-left
        xi, yi = x.astype(int), y.astype(int)
        # internal coordinates
        xf, yf = x - xi, y - yi
        # fade factors
        u, v = fade(xf), fade(yf)
        # noise components
        n00 = gradient(p[p[xi] + yi], xf, yf)
        n01 = gradient(p[p[xi] + yi + 1], xf, yf - 1)
        n11 = gradient(p[p[xi + 1] + yi + 1], xf - 1, yf - 1)
        n10 = gradient(p[p[xi + 1] + yi], xf - 1, yf)
        # combine noises
        x1 = lerp(n00, n10, u)
        x2 = lerp(n01, n11, u)  # FIX1: I was using n10 instead of n01
        return lerp(x1, x2, v)  # FIX2: I also had to reverse x1 and x2 here

    def lerp(a, b, x):
        "linear interpolation"
        return a + x * (b - a)

    def fade(t):
        "6t^5 - 15t^4 + 10t^3"
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    def gradient(h, x, y):
        "grad converts h to the right gradient vector and return the dot product with (x,y)"
        vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
        g = vectors[h % 4]
        return g[:, :, 0] * x + g[:, :, 1] * y

    noise_scaling = (-440 * rim_distance) + 4000
    
    # EDIT : generating noise at multiple frequencies and adding them up
    p = np.zeros((500,500))
    for i in range(9):
        freq = 2**i
        lin = np.linspace(1, freq, 500, endpoint=False)
        x, y = np.meshgrid(lin, lin)  # FIX3: I thought I had to invert x and y here but it was a mistake
        p = perlin(x, y, seed=random.randint(1, 500)) / freq + (0.4 * p)
    
    if is_mare:
        target_raster_noisey = target + ((3 * 1500) * p) - (np.ones((500, 500)) * 45)
        for i in range(500):
            for j in range(500):
                if target_raster_noisey[j][i] < 0:
                    target_raster_noisey[j][i] = 0
    else:
        target_raster_noisey = target + ((noise_scaling) * p) + (np.ones((500, 500)) * 5)

    return target_raster_noisey

# Defining the perlin algorithm outside of generate_target_noise(). Redundant but I couln't get the code to run without doing this so
def perlin(x, y, seed=0):
        # permutation table
        np.random.seed(seed)
        p = np.arange(256, dtype=int)
        np.random.shuffle(p)
        p = np.stack([p, p]).flatten()
        # coordinates of the top-left
        xi, yi = x.astype(int), y.astype(int)
        # internal coordinates
        xf, yf = x - xi, y - yi
        # fade factors
        u, v = fade(xf), fade(yf)
        # noise components
        n00 = gradient(p[p[xi] + yi], xf, yf)
        n01 = gradient(p[p[xi] + yi + 1], xf, yf - 1)
        n11 = gradient(p[p[xi + 1] + yi + 1], xf - 1, yf - 1)
        n10 = gradient(p[p[xi + 1] + yi], xf - 1, yf)
        # combine noises
        x1 = lerp(n00, n10, u)
        x2 = lerp(n01, n11, u)  # FIX1: I was using n10 instead of n01
        return lerp(x1, x2, v)  # FIX2: I also had to reverse x1 and x2 here

def lerp(a, b, x):
    "linear interpolation"
    return a + x * (b - a)

def fade(t):
    "6t^5 - 15t^4 + 10t^3"
    return 6 * t**5 - 15 * t**4 + 10 * t**3

def gradient(h, x, y):
    "grad converts h to the right gradient vector and return the dot product with (x,y)"
    vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    g = vectors[h % 4]
    return g[:, :, 0] * x + g[:, :, 1] * y


# Extract shadow length approximation (in pixels). The shadow is randomized for 5-50% burrial. Reads from gdf_boulders
def shadow_len(area, AOI):
    AOI = 90 - AOI
    burried = random.uniform(0.5, 1.0)
    h = (1.0 * math.sqrt(area / math.pi)) * burried
    shadow_len = h / math.tan(math.radians(AOI))
    
    return shadow_len


# The ideal boulder density for the given distance
def compute_coverage(rim_dist): 
    return int(176.14 * ((rim_dist + 0.5) ** -0.746))

# These function compute the density of each size class as a percent of the total density. Cumulative density by size class is 
# typically within +- 8% of the ideal density
# These CSFD functions are poorly aproximated and will be updated in the near future
def small_density(rim_dist):
    return int((1.719 + (0.072 * np.log(rim_dist)) - 0.94) * compute_coverage(rim_dist))

def mid_density(rim_dist):
    return int((1.282 + (-0.032 * rim_dist) - 1) * compute_coverage(rim_dist))

def large_density(rim_dist):
    return int((1.086 * (rim_dist ** -0.018) - 1.06) * compute_coverage(rim_dist))

# Cumulative density by size class (for metadata)
def CDBSC(rim_dist):
    return small_density(rim_dist) + mid_density(rim_dist) + large_density(rim_dist)

# I make devs cower in fear with my garbage code
def generate_fake_boulders(rim_distance, AOI, fancy_shadows, gdf_boulders, target_raster_noisey):
    # Boudler generating specs
    i = 0.0                                                                                                               # Iterable
    fake_masks = []                                                                                                       # 3D arr of boulders
    fake_boulders_array = np.zeros((500, 500)).astype('uint8')                                           # cumul arr of fake boulders
    bins = [16, 26, 70, 500]                                                                                               # Size bins for CS(F/R)D
    density_threshold_by_bin = [small_density(rim_distance), mid_density(rim_distance), large_density(rim_distance)]      # num of boulders in each bin
    boulder_areas = []                                                                                                    # For degugging
    fancy_shadows = fancy_shadows                                                                                         # Redundant, meh
    
    shadowed_boulder_mask = np.zeros((500, 500))                                                                          # Prevents loss of detail in s-b overlap

    # Shadow generating specs    
    row_step, col_step = random.randint(-1, 1), random.randint(-1, 1)                                                     # AOI shadow dir vector
    light_direction = np.array([row_step, col_step])
    shadow_direction =-light_direction
    shadow_mask = np.zeros((500, 500))                                                                                    # Like fake_boulders_array, for shadows
    shadow_coords = []                                                                                                    # for s-b overlap
    
    # Generate randomized groupings using Perlin
    '''
    Groupings are currently generated as a low-threshold noise map. In future versions, a location argument 
    will be passed to generate groupings which simulate ejecta rays.
    '''
    bools = [True, True, True, True, True, True, False, False, False, False]
    groupings = np.zeros((500,500))
    for i in range(4):
        freq = 2**i
        lin = np.linspace(1, freq, 500, endpoint=False)
        x, y = np.meshgrid(lin, lin) 
        perpituity = perlin(x, y, seed=random.randint(1, 500))
        groupings = 255 * perpituity / freq + (1 * groupings)  
    for i in range(500):                                                                                                  # CPU and memory are crying over all these loops
        for j in range(500):
            if groupings[i][j] < 0:
                groupings[i][j] = 1
            else:
                groupings[i][j] = 0
    group_points = np.sum(groupings[groupings > 0])
    plt.imshow(groupings, cmap = "gray")
                
    # Complete a boulder iteration for each of the three size classes
    for t in range(3):
        area_covered = 0
        area_covered_thres = density_threshold_by_bin[t] # percent (select this value uniformely between 0 and 35 %)
        min_area, max_area = bins[t], bins[t + 1]
        print(f"Generating boulders for {t + 1}th size class. {density_threshold_by_bin[t]} boulders will be generated")
        
        while area_covered < area_covered_thres:
            
            # Should the boulder be in a group? Weighted towards yes
            in_groupings = bools[int(random.triangular(0, 9, 3))]
            
            # IS the boulder in a group? Defaul is yes -- we dont want to regenerate a boulder if its not supposed to be in a group
            in_group = False
            
            # select a random boulder from the source withing the SFD bin
            row_boulder = gdf_boulders[(min_area < gdf_boulders["area"]) & (gdf_boulders["area"] < max_area)].sample()
            boulder_areas.append(row_boulder["area"])
            # Set up the boulder for manipulation
            with rio.open(row_boulder.image.values[0]) as src:
                out, tt = rio.mask.mask(src, [row_boulder.geometry.values[0]], all_touched=False, invert=False, nodata=0, filled=True)
                # , crop=True, pad=(5, 0, 0, 0) ?
                out_image = out[0]
                out_mask = (out_image > 1).astype('uint8')

                # Run the boulder through the augmentation pipeline
                transformed = transform(image=out_image, mask=out_mask)
                transformed_image = transformed['image']
                transformed_mask = transformed['mask'] # need to remove empty mask

                # If the boulder is supposed to be in a group and it is not, regenerate
                if in_groupings:
                    temp = groupings + transformed_image
                    group_points_with_boulder= np.sum(temp[groupings > 0])
                    # If the boulder is not a valid group, regerate
                    if not (group_points < group_points_with_boulder):
                        in_group = True

            # if anything in this list is true, we regenerate the boulder
            # if mask is falling outside of the image, regenerate
            # if mask is smaller than 16 pixels, regenerate
            # if mask is overlapping with other boulder in fake_boulders_array, regenerate 
            # if mask is not located within groupings, regenerate
            # break if not working after 100 iterations (we don't want to be stuck in an infinite loop)
            n_iter = 0
            while (~np.any(transformed_image) or len(np.where(transformed_image > 0)[0]) <= 16 or 
                   np.nonzero((fake_boulders_array!=0) & (transformed_image!=0))[0].shape[0] > 0 or in_group): # don't want to have mini

                # if more than 100 iterations, let's give up
                if n_iter > 100:
                    break
                n_iter += 1

            if n_iter > 100:
                None
            else:
                # The final transformed boulder and its mask is added
                fake_masks.append(transformed_mask)
                fake_boulders_array = fake_boulders_array + transformed_image
                area_covered+=1

                # Generate the shadow and store it in a separate array
                boulder_indices = np.where(transformed_image > 0)
                boulder_points = len(boulder_indices[0]) # This is also the px area of the boulder

                rows = len(transformed_image)
                cols = len(transformed_image[1])
                shadow = int(shadow_len(boulder_points, AOI))
                if boulder_points > 120:
                    for i in range(boulder_points):
                        y, x = boulder_indices[0][i], boulder_indices[1][i]
                        # If the shadow generation will not lead out of bounds, continue
                        if x + shadow_direction[0] < 0 or x + shadow_direction[0] >= 500 or \
                           y + shadow_direction[1] < 0 or y + shadow_direction[1] >= 500:
                            continue
                        # Extend the shaded region by a given length (shadow)
                        if not shadow > 0:
                            break
                        for j in range(1, shadow):

                            px = x + int(shadow_direction[0] * j)
                            py = y + int(shadow_direction[1] * j)

                            if 0 <= py < len(transformed_image) and 0 <= px < len(transformed_image[1]):

                                # If the point is blank and within the array bounds
                                if transformed_image[py, px] != 0:
                                    break
                                else:
                                    # Pixel value should increase with distance
                                    shadow_mask[py, px] = ((target_raster_noisey[py, px] - (AOI / 1.2)) / ((shadow - j))) * random.uniform(0.9, 1.1)
                                    shadow_coords.append((py, px))
                                        
        # (Optional) recomended for higher AOI
        # DEPRICATED. apply_blur_transformation() now handles boulder/shadow/background raster shading.
        if fancy_shadows:
            for coord in shadow_coords: 
                y, x = coord[0], coord[1]
                # Check if the pixel has >= 1 empty neighbor
                if 1 <= y < len(transformed_image) - 1 and 1 <= x < len(transformed_image[1]) - 1:
                    if (shadow_mask[y + 1, x] == 0 or shadow_mask[y - 1, x] == 0 or shadow_mask[y, x + 1] == 0 or shadow_mask[y, x - 1] == 0):
                        shadow_mask[y, x] = int(target_raster_noisey[y, x] / random.uniform(2.3, 3.5)) 
    
    # handeling shadow-boudler overlap
    boulders = np.where(fake_boulders_array > 0)
    for i in range(len(boulders[0])):
        y, x = boulders[0][i], boulders[1][i]
        coord = (y, x)
        if coord in shadow_coords:
            new_val = fake_boulders_array[y][x] - 40
            if new_val <= 0:
                shadowed_boulder_mask[y][x] = 1
            else:
                shadowed_boulder_mask[y][x] = new_val
                
    return area_covered, fake_masks, fake_boulders_array, shadow_mask, shadowed_boulder_mask

def apply_blur_transformation(image_array):
    # Define the augmentation pipeline
    transform = A.Compose([
        A.Blur(p=1, blur_limit=(2, 5))  # Probability of applying the blur transformation is set to 1 (100%)
    ])

    # Since albumentations expects images with channel information, we will add a dummy channel (third dimension)
    # to the grayscale array to make it compatible.
    image_array_with_channel = np.expand_dims(image_array, axis=2)

    # Apply the transformation to the image array
    augmented = transform(image=image_array_with_channel)

    # Retrieve the augmented image
    augmented_image_array = augmented['image']

    # Remove the dummy channel to get back the grayscale image
    augmented_image_array = np.squeeze(augmented_image_array, axis=2)

    return augmented_image_array

