import geopandas as gpd
from shapely.geometry import LineString
import numpy as np
import pandas as pd
from shapely import wkt
from pathlib import Path
from geopandas.tools import sjoin
import os

#------------------------------------------------------------
#                             Functions
#--------------obstruction points increase sens-----------------

def modify_obstruction(df, sens):
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
    modified_df['group'] = (modified_df['obstructio'].diff() != 0).astype(int).cumsum()
    # Get the indices of the first and last row of each group
    group_indices = modified_df.groupby('group')['obstructio'].apply(lambda x: (x.index[0], x.index[-1])).values
    # Iterate over groups
    for start, end in group_indices:
        # If the group's 'obstructio' value is 1
        if modified_df.loc[start, 'obstructio'] == 1:
            # Get the indices of the sens rows before and after the group
            indices = list(range(max(0, start - sens), start)) + list(range(end + 1, min(end + sens + 1, len(modified_df))))
            # Set the 'obstructio' value of these rows to 1
            modified_df.loc[indices, 'obstructio'] = 1
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



#-------------------Inputs -----------------------------------------
output_dir = Path("C:/Users/zkhalid5/OneDrive - George Mason University - O365 Production/Documents/ArcGIS/Projects/HydroDEM - Tool Run/step2")
segment_lines = gpd.read_file(output_dir/ "RUN2" /"S2_segment.shp")
result_points = gpd.read_file(output_dir / "RUN2" /"S2_result.shp")
transect_length = 10 # default 15 ;for transect only


#--------------Expand obs points to cover entire region -----------------

mod_result_points = modify_obstruction(result_points, 3)
obs_points = mod_result_points[mod_result_points['obstructio'] == 1]
obs_points.crs = segment_lines.crs
# save
obs_points.to_file(output_dir / "S2_obs_points.shp")








#--------------Road Check-----------------


S2_roads_buffer = Path(output_dir / "roads_buffer_project.shp")
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
# S2_mod_obs_points.to_file(output_dir / "S2_mod_obs_points.shp")
# S2_mod_obs_points.to_file(os.path.join(output_dir, "S2_mod_obs_points.shp"))




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