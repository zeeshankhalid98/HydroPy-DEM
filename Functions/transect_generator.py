import arcpy
import geopandas as gpd
from shapely.geometry import LineString
import numpy as np

#------------------------------------------------------------
# Functions
#------------------------------------------------------------

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

#------------------------------------------------------------
# Inputs
#------------------------------------------------------------

input_shapefile = arcpy.GetParameterAsText(0)
transect_length = float(arcpy.GetParameterAsText(1))
output_location = arcpy.GetParameterAsText(2)

#------------------------------------------------------------
# Processing
#------------------------------------------------------------

segments_select = gpd.read_file(input_shapefile)
segments_select['transect'] = segments_select['geometry'].apply(create_transect, args=(transect_length,))
segments_select = segments_select.drop(columns=['geometry'])
segments_select = segments_select.rename(columns={'transect': 'geometry'})
segments_select.to_file(output_location)

#------------------------------------------------------------
# Output
#------------------------------------------------------------

arcpy.SetParameter(2, output_location)