import geopandas as gpd
from shapely.geometry import LineString
import arcpy
import sys
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
import rasterio
from shapely.geometry import Point
from pathlib import Path


""" ------------------------- Script Summary -----------------------------------

- Segmentation: Divides geometries into segments of a specified length.
- Center Point Generation: Calculates the center point of each segment.
- DEM Sampling: Samples a DEM raster at each center point.
- LOESS Smoothing: Calculates the LOESS best fit line for each group of points.
- Difference Calculation: Computes the difference between the DEM value and the LOESS best fit.
- Natural Breaks Classification: Applies the Jenks method to classify differences, identifying potential obstructions.


Inputs:

- smooth_stream: Smoothed stream from ARC
- input_dem_raster: DEM raster dataset
- segment_length: Length of segments to create
- output_dir: Output directory for results



Outputs:

- Smooth stream 
- Segment lines
- Result (stream points with obstruction classification {stream_points})

"""



#--------------------------  Functions  ----------------------------------


def create_segments(df, segment_length):
    segments = []
    stream_id_counter = 1

    for _, row in df.iterrows():
        geom = row.geometry
        total_length = geom.length

        if total_length == 0:
            continue  # Skip zero-length geometries

        num_segments = max(int(total_length / segment_length), 1)  # Ensure at least one segment

        segment_step = total_length / num_segments
        segment_id_counter = 1

        for i in range(num_segments):
            start_distance = i * segment_step
            end_distance = (i + 1) * segment_step

            segment = LineString([geom.interpolate(start_distance), geom.interpolate(end_distance)])
            segment_with_id = {'stream_id': stream_id_counter, 'segment_id': segment_id_counter, 'geometry': segment}
            segments.append(segment_with_id)
            segment_id_counter += 1

        stream_id_counter += 1

    segments_gdf = gpd.GeoDataFrame(segments)

    # Assign the same CRS as the input DataFrame
    if not df.empty:
        segments_gdf.crs = df.crs

    return segments_gdf


# Function to calculate LOESS best fit line
def calculate_loess(group):
    x = group.index.values
    y = group['dem_value'].values
    smoothed = lowess(y, x, frac=0.2)  # Adjust frac as needed
    return smoothed[:, 1]


def create_center_points(segments_gdf):
    # Create a new GeoDataFrame for the center points
    center_points_gdf = segments_gdf.copy()
    
    # Calculate the center point of each segment
    center_points_gdf['geometry'] = segments_gdf['geometry'].centroid
    
    return center_points_gdf


def sample_dem_at_points(center_points_gdf, dem_raster_path):
    # Open the DEM raster
    with rasterio.open(dem_raster_path) as dem:
        # Sample the DEM at each center point
        center_points_gdf['dem_value'] = [
            next(dem.sample([(pt.x, pt.y)]))[0]
            for pt in center_points_gdf['geometry']
        ]
    
    return center_points_gdf


import jenkspy

def calculate_jenks_groups(df, column, nb_class):
    # Calculate Jenks breaks
    breaks = jenkspy.jenks_breaks(df[column], nb_class)

    # Create new column and assign groups based on Jenks breaks
    df['group'] = 0  # Initialize column with 0
    for i in range(1, nb_class):
        df.loc[df[column] > breaks[i], 'group'] = i  # Assign group number to values greater than the break

    return df

#------------------------  Arc parameters  ------------------------------------


#  ------------------------ Input Parameters  ------------------------
smooth_stream = arcpy.GetParameterAsText(0) # smooth stream (INPUT)
input_dem_raster = arcpy.GetParameterAsText(1) # dem raster dataset (INPUT)
segment_length = float(arcpy.GetParameterAsText(2)) # dem_resolution (INPUT)
output_dir = arcpy.GetParameterAsText(3) # output directory     (OUTPUT)



#  ------------------------ Main Function  ------------------------
smooth_stream = gpd.read_file(smooth_stream)
segment_lines = create_segments(smooth_stream, segment_length) # Create segments

# save
smooth_stream.to_file(output_dir /"S2_smooth_stream.shp")

# save
segment_lines.to_file(output_dir /"S2_segment.shp")


stream_points = create_center_points(segment_lines)  # Create center points
stream_points = sample_dem_at_points(stream_points, input_dem_raster)   # Sample DEM at center points

# Apply LOESS and calculate difference for each group
stream_points['best_fit'] = np.nan
stream_points['difference'] = np.nan
for _, group in stream_points.groupby('stream_id'):
    stream_points.loc[group.index, 'best_fit'] = calculate_loess(group) # Calculate LOESS
    stream_points.loc[group.index, 'difference'] = group['dem_value'] - stream_points.loc[group.index, 'best_fit']  # Calculate difference


stream_points = calculate_jenks_groups(stream_points, 'difference', 2)  # Apply Natural Breaks 
stream_points.rename(columns={'group': 'obstruction'}, inplace=True)    


# save
stream_points.to_file(output_dir /"S2_result.shp")


# Set the output parameters
arcpy.SetParameter(3, output_dir)


