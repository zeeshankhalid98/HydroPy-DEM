import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from shapely.geometry import LineString
import sys
from statsmodels.nonparametric.smoothers_lowess import lowess
import rasterio
from shapely.geometry import Point
import jenkspy


#------------------------------------------------------------
# Functions
#------------------------------------------------------------

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

def calculate_jenks_groups(df, column, nb_class):
    # Calculate Jenks breaks
    breaks = jenkspy.jenks_breaks(df[column], nb_class)

    # Create new column and assign groups based on Jenks breaks
    df['group'] = 0  # Initialize column with 0
    for i in range(1, nb_class):
        df.loc[df[column] > breaks[i], 'group'] = i  # Assign group number to values greater than the break

    return df



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




# ------------------------------------------------------------
# ArcPy script tool parameters
# ------------------------------------------------------------

gdf = gpd.read_file(r"test shps\step2_smooth_stream.shp") ## good
output_shapefile = r"test shps\segment.shp"
input_dem_raster = r"test shps\dem_clip.tif"
bc_points = gpd.read_file(r"test shps\bc_points.shp")




input_shapefile = gdf
segment_length = 1
segments_gdf = create_segments(gdf, segment_length) # Create segments  
# segments_gdf.to_file(output_shapefile)





##################################################


segment_length = 10
output_center_points_shapefile = r"test shps\output\output2_center_result.shp"
input_dem_raster = r"test shps\dem_clip.tif"

df = gdf
segments_gdf = create_segments(df, segment_length) # Create segments    
center_points_gdf = create_center_points(segments_gdf)  # Create center points
center_points_gdf = sample_dem_at_points(center_points_gdf, input_dem_raster)   # Sample DEM at center points




# Apply LOESS and calculate difference for each group
center_points_gdf['best_fit'] = np.nan
center_points_gdf['difference'] = np.nan
for _, group in center_points_gdf.groupby('stream_id'):
    center_points_gdf.loc[group.index, 'best_fit'] = calculate_loess(group) # Calculate LOESS
    center_points_gdf.loc[group.index, 'difference'] = group['dem_value'] - center_points_gdf.loc[group.index, 'best_fit']  # Calculate difference


center_points_gdf = calculate_jenks_groups(center_points_gdf, 'difference', 2)  # Apply Natural Breaks 
center_points_gdf.rename(columns={'group': 'obstruction'}, inplace=True)    




# Gotta convert the center_points_gdf to obs_points_gdf because I only want points over the bridges


potential_obs = center_points_gdf[center_points_gdf['obstruction'] == 1]


# now this center_points_gdf has all the obs point but I gotta find a way to only get the points near roads


roads = gpd.read_file(r"test shps\roads.shp")
roads = roads.to_crs(center_points_gdf.crs)
buffered_roads = roads.buffer(50)
buffered_roads_union = buffered_roads.unary_union
# buffered_roads.plot()
# plt.show()
# buffered_roads.to_file(r"test shps\buffered_roads1.shp")


potential_obs.loc[:, 'confirmed'] = potential_obs['geometry'].apply(lambda x: x.intersects(buffered_roads_union)).astype(int)




confirmed_obs = potential_obs[potential_obs['confirmed'] == 1]







# now I have to work with 2 different files (segments_gdf and confirmed_obs) and I have to merge them together

confirmed_obs_tmp = confirmed_obs.drop(columns='geometry')
bridge_segments = pd.merge(segments_gdf, confirmed_obs_tmp, on=['stream_id', 'segment_id'])
# merged_df = merged_df.rename(columns={"geometry_x": "segment_geometry", "geometry_y": "observation_point"})



# bridge_segments[bridge_segments['obstruction'] != 1]


transects_gdf = bridge_segments.copy()

transects_gdf['transect'] = transects_gdf['geometry'].apply(create_transect, args=(10,))
transects_gdf = transects_gdf.drop(columns=['geometry'])
transects_gdf = transects_gdf.rename(columns={'transect': 'geometry'})


transects_gdf.to_file(r"test shps\output\transects123.shp")







# bridge_segments.plot()
# plt.show()









