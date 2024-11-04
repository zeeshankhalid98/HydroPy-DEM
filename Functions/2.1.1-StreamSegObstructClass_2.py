import geopandas as gpd
from shapely.geometry import LineString
# import arcpy
import sys
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
import rasterio
from shapely.geometry import Point
from pathlib import Path
import geopandas as gpd
from shapely.geometry import LineString
import numpy as np
import pandas as pd
from shapely import wkt
from pathlib import Path
from geopandas.tools import sjoin
import os

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


# # Function to calculate LOESS best fit line
# def calculate_loess(group):
#     x = group.index.values
#     y = group['dem_value'].values
#     print(f"here is x : {x}, y : {y}")
#     smoothed = lowess(y, x, frac=0.2)  # Adjust frac as needed
#     return smoothed[:, 1]

# Function to calculate LOESS best fit line
def calculate_loess(group):
    x = group.index.values
    y = group['dem_value'].values
    # print(f"here is x : {x}, y : {y}")
    smoothed = lowess(y, x, frac=0.2)  # Adjust frac as needed
    return smoothed[:, 1]



# # Function to calculate LOESS best fit line
# def calculate_loess(group, column='dem_value'):
#     x = group.index.values
#     y = group[column].values
#     smoothed = lowess(y, x, frac=0.2)  # Adjust frac as needed
#     return smoothed[:, 1]


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




def modify_obstruction(df, sens, column_name='obstruction'):
    """
        The function creates groups based on changes in the 'obstructio' column. For each group where the 'obstructio' value is 1,
    it sets the 'obstructio' value of 'sens' rows before the start of the group and 'sens' rows after the end of the group to 1.
    
    Input:
    - df: A DataFrame that contains a column named 'obstructio'.
    - sens: An integer that specifies the number of rows before and after each group to modify.
    
    Output:
    - A modified DataFrame where the 'obstructio' column has been adjusted according to the specified rules. 
    """
    modified_df = df.copy()
    # create group column
    modified_df['group'] = (modified_df[column_name].diff() != 0).astype(int).cumsum()
    # Get the indices of the first and last row of each group
    group_indices = modified_df.groupby('group')[column_name].apply(lambda x: (x.index[0], x.index[-1])).values
    # Iterate over groups
    for start, end in group_indices:
        # If the group's 'obstructio' value is 1
        if modified_df.loc[start, column_name] == 1:
            # Get the indices of the sens rows before and after the group
            indices = list(range(max(0, start - sens), start)) + list(range(end + 1, min(end + sens + 1, len(modified_df))))
            # Set the 'obstructio' value of these rows to 1
            modified_df.loc[indices, column_name] = 1
    # Drop the 'group' column
    # modified_df = modified_df.drop(columns='group')
    return modified_df



def create_transect(line, length=15):
    mid_point = line.interpolate(0.5, normalized=True)   
    x, y = line.xy
    dx = x[-1] - x[0]
    dy = y[-1] - y[0]
    angle = np.arctan2(dy, dx)  
    start_point = (mid_point.x - (length / 2) * np.cos(angle + np.pi/2), mid_point.y - (length / 2) * np.sin(angle + np.pi/2))
    end_point = (mid_point.x + (length / 2) * np.cos(angle + np.pi/2), mid_point.y + (length / 2) * np.sin(angle + np.pi/2))
    transect = LineString([start_point, end_point])
    return transect






#------------------------  Arc parameters  ------------------------------------


#  ------------------------ Input Parameters  ------------------------
# smooth_stream = arcpy.GetParameterAsText(0) # smooth stream (INPUT)
# input_dem_raster = arcpy.GetParameterAsText(1) # dem raster dataset (INPUT)
# segment_length = float(arcpy.GetParameterAsText(2)) # dem_resolution (INPUT)
# output_dir = arcpy.GetParameterAsText(3) # output directory     (OUTPUT)




# smooth_stream = r"C:\Users\zkhalid5\OneDrive - George Mason University - O365 Production\Documents\ArcGIS\Projects\HydroDEM - Tool Run\step3\RUN4\S2_smooth_stream.shp" # smooth stream (INPUT)
# input_dem_raster = r"C:\Users\zkhalid5\OneDrive - George Mason University - O365 Production\Documents\ArcGIS\Projects\HydroDEM - Tool Run\step3\RUN4\dem_clip.tif" # dem raster dataset (INPUT)
# segment_length = 1 # dem_resolution (INPUT)
# output_dir = Path(r"C:\Users\zkhalid5\OneDrive - George Mason University - O365 Production\Documents\ArcGIS\Projects\HydroDEM - Tool Run\step3\RUN4") # output directory     (OUTPUT)
# loess_iterations = 100  # Number of LOESS iterations

# smooth_stream = r"C:\Users\zkhalid5\OneDrive - George Mason University - O365 Production\Documents\ArcGIS\Projects\HydroDEM - Tool Run\GBDS\nova_output\smooth_stream.shp" # smooth stream (INPUT)

# smooth_stream = r"C:\Users\zkhalid5\OneDrive - George Mason University - O365 Production\Documents\ArcGIS\Projects\HydroDEM - Tool Run\Validation1\validation\smooth_stream.shp" # smooth stream (INPUT)

smooth_stream = r"C:\Users\zkhalid5\OneDrive - George Mason University - O365 Production\Documents\ArcGIS\Projects\HydroDEM - Tool Run\Validation1\validation\smooth_stream6.shp" # smooth stream (INPUT)

input_dem_raster = r"C:\Users\zkhalid5\OneDrive - George Mason University - O365 Production\Documents\ArcGIS\Projects\HydroDEM - Tool Run\Validation1\validation\dem.tif" # dem raster dataset (INPUT)
segment_length = 1 # dem_resolution (INPUT)
output_dir = Path(r"C:\Users\zkhalid5\OneDrive - George Mason University - O365 Production\Documents\ArcGIS\Projects\HydroDEM - Tool Run\Validation1\validation") # output directory     (OUTPUT)
loess_iterations = 10  # Number of LOESS iterations





#  ------------------------ Main Function  ------------------------
smooth_stream = gpd.read_file(smooth_stream)
smooth_stream = smooth_stream.drop(['from_node', 'to_node', 'InLine_FID', 'grid_code', 'Shape_Leng', 's_id', 'Shape_Le_1', 'arcid', 'ORIG_FID'], axis=1)
smooth_stream = smooth_stream.reset_index(drop=True)
smooth_stream.index.name = 'arcid'


segment_lines = create_segments(smooth_stream, segment_length) # Create segments

# # save
# smooth_stream.to_file(output_dir /"S2_smooth_stream.shp")

# # save
segment_lines.to_file(output_dir /"S2_segment.shp")




stream_points = create_center_points(segment_lines)  # Create center points

stream_points.to_file(output_dir /"S2_result_pts.shp")


# import again after delete points

stream_points = gpd.read_file(output_dir / "S2_result_pts.shp")


stream_points = sample_dem_at_points(stream_points, input_dem_raster)   # Sample DEM at center points


stream_points['dem_value'] = round(stream_points['dem_value'], 5)  # Round DEM values to 2 decimal places
stream_points['original_dem_value'] = stream_points['dem_value']  # Create a copy of the original DEM values


stream_points['best_fit'] = np.nan

for _ in range(loess_iterations):
    # ------------------------- loess --------------------------------
    for _, group in stream_points.groupby('stream_id'):
        stream_points.loc[group.index, 'best_fit'] = calculate_loess(group) # Calculate LOESS
    stream_points['dem_value'] = stream_points['best_fit']











# ran above it is working fine












# ------------------------- difference --------------------------------
stream_points['difference'] = stream_points['original_dem_value'] - stream_points['best_fit']  # Calculate difference

# ------------------------- obstruction --------------------------------

stream_points = calculate_jenks_groups(stream_points, 'difference', 2)  # Apply Natural Breaks 
stream_points.rename(columns={'group': 'obstruction'}, inplace=True)    


# save
stream_points.to_file(output_dir /"S2_result.shp")


# Set the output parameters
# arcpy.SetParameter(3, output_dir)




### 2.2 -  make transects 



# stream_points = gpd.read_file(output_dir / "S2_result73.shp")



#-------------------Inputs -----------------------------------------
output_dir 
segment_lines
result_points = stream_points
transect_length = 10 # default 15 ;for transect only


result_points['obstruction']

#--------------Expand obs points to cover entire region -----------------

mod_result_points = modify_obstruction(result_points, 3, column_name='obstruction')
obs_points = mod_result_points[mod_result_points['obstruction'] == 1]
obs_points.crs = segment_lines.crs
# save
obs_points.to_file(output_dir / "S2_obs_points.shp")


# obs_points = gpd.read_file(r"C:\Users\zkhalid5\OneDrive - George Mason University - O365 Production\Documents\ArcGIS\Projects\HydroDEM - Tool Run\step3\RUN2\S2_obs_points_mod_m.shp")





#--------------Road Check-----------------


# S2_roads_buffer = Path(output_dir / "buffered_roads_proj.shp")

S2_roads_buffer = Path(r"C:\Users\zkhalid5\OneDrive - George Mason University - O365 Production\Documents\ArcGIS\Projects\HydroDEM - Tool Run\Validation1\validation\road_buffer_project.shp")

S2_roads_buffer = gpd.read_file(S2_roads_buffer)




#-------------------  select by location / sjoin  ----------------------------

S2_mod_obs_points = sjoin(obs_points, S2_roads_buffer, how='left', op='intersects')

# Drop S3_roads_buffer cols
columns_to_drop = [col for col in S2_roads_buffer.columns if col != 'geometry']
S2_mod_obs_points = S2_mod_obs_points.drop(columns=columns_to_drop)
S2_mod_obs_points = S2_mod_obs_points.rename(columns={'index_right': 'distance_to_road'})

# near_roads col mark points near roads
S2_mod_obs_points['near_roads'] = S2_mod_obs_points['distance_to_road'].apply(lambda x: 0 if pd.isnull(x) else 1)

# save
S2_mod_obs_points.to_file(output_dir / "S2_mod_obs_points.shp")
# S2_mod_obs_points.to_file(os.path.join(output_dir, "S2_mod_obs_points.shp"))


# S2_mod_obs_points[S2_mod_obs_points['near_roads'] == 1]

#--------------obsstructions lines identifier -----------------

# join
merged_points = pd.merge(S2_mod_obs_points.drop(columns='geometry'), segment_lines, on=['stream_id', 'segment_id'], how='inner')

#--------------Check if gdf if not make it-----------------

# Check 'geometry' is strings
if isinstance(merged_points['geometry'].iloc[0], str):
    # Convert 'geometry' to GeoSeries
    merged_points['geometry'] = merged_points['geometry'].apply(wkt.loads)
# Convert the df to a gdf
obs_lines = gpd.GeoDataFrame(merged_points, geometry='geometry')
obs_lines.crs = segment_lines.crs

# save
obs_lines.to_file(output_dir / "S2_obs_lines.shp")


#--------------transect-----------------

transects = obs_lines[obs_lines['near_roads'] == 1]
transects['transect'] = transects['geometry'].apply(create_transect, args=(transect_length,))
transects = transects.drop(columns=['geometry'])
transects = transects.rename(columns={'transect': 'geometry'})

# save
transects.to_file(output_dir / "S2_transect.shp")




