import geopandas as gpd
from shapely.geometry import LineString
from shapely.ops import transform
from functools import partial
import numpy as np

#------------------------------------------------------------
# Functions
#------------------------------------------------------------

def create_transect(line, length=10):
    mid_point = line.interpolate(0.5, normalized=True)   
    # Calculate the angle of the line
    x, y = line.xy
    dx = x[-1] - x[0]
    dy = y[-1] - y[0]
    angle = np.arctan2(dy, dx)  
    # Create a line of specified length perpendicular to the original line at the midpoint
    start_point = (mid_point.x - (length / 2) * np.cos(angle + np.pi/2), mid_point.y - (length / 2) * np.sin(angle + np.pi/2))
    end_point = (mid_point.x + (length / 2) * np.cos(angle + np.pi/2), mid_point.y + (length / 2) * np.sin(angle + np.pi/2))
    transect = LineString([start_point, end_point])
    return transect


#------------------------------------------------------------
# Inputs
#------------------------------------------------------------
segments_select_file = r"test shps\segmentselection.shp"
transect_length = 10
output_location = r"test shps\transects.shp"


#------------------------------------------------------------
# Processing
#------------------------------------------------------------







segments_select = gpd.read_file(segments_select_file)
segments_select['transect'] = segments_select['geometry'].apply(create_transect, args=(transect_length,))
segments_select = segments_select.drop(columns=['geometry'])
# segments_select = segments_select.drop(columns=[ 'merge_1', 'stream_id_', 'segment__1', 'FID_1'])
segments_select = segments_select.rename(columns={'transect': 'geometry'})
segments_select.to_file(output_location)


#------------------------------------------------------------
# Output
#------------------------------------------------------------



