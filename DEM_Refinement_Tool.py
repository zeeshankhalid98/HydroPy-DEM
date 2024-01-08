# DEM Refinement Tool python library
# Author: Zeeshan Khalid


import os
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.enums import Resampling
from shapely.geometry import Point
from shapely.geometry import LineString
from shapely.geometry import MultiPoint

class DEMRefinementTool:
    def __init__(self):
        pass

    @staticmethod
    def create_segments(df, segment_length = 2.0):
        """
        Creates segments for the input GeoDataFrame based on the specified segment length.
        Parameters:
        - df (GeoDataFrame): The input GeoDataFrame containing a 'geometry' column with LineString objects.
        - segment_length (float): The desired length for each segment.
        Returns:
        - segments_gdf (GeoDataFrame): A GeoDataFrame containing segments of the input LineString with unique IDs.
        """
        original_line = df.loc[0, 'geometry']
        # segment_length = 2.0
        # Calculate the total number of segments needed based on length
        total_length = original_line.length
        num_segments = int(total_length / segment_length)
        segment_length_actual = total_length / num_segments
        segments = []
        # Iterate through each segment and create perpendiculars
        for i in range(num_segments):
            start_distance = i * segment_length_actual
            end_distance = (i + 1) * segment_length_actual
            segment = LineString(
                [original_line.interpolate(start_distance), original_line.interpolate(end_distance)]
            )
            # Assign a unique IDs to each segment
            segment_id = i + 1 
            segment_with_id = {'id': segment_id, 'geometry': segment}
            segments.append(segment_with_id)

        segments_gdf = gpd.GeoDataFrame(segments, geometry='geometry')
        return segments_gdf   
    
    @staticmethod
    def generate_perpendicular_lines(linestring):
        """
        Generates perpendicular lines/Transects for each line segment in the input LineString.
        Parameters:
        - linestring (LineString): The input LineString for which perpendicular lines will be generated.
        Returns:
        - perp_lines (list): A list containing LineString objects representing perpendicular lines
                            for each segment of the input LineString.
        """
        vertices = list(linestring.coords)
        perp_lines = []
        for i in range(len(vertices)-1):
            start_point = vertices[i]
            end_point = vertices[i+1]
            midpoint = MultiPoint([start_point, end_point]).centroid
            angle = np.arctan2(end_point[1] - start_point[1], end_point[0] - start_point[0])
            angle_perpendicular = angle + np.pi/2
            half_length = 20  # Adjust this value based on your requirements
            endpoint1 = (
                midpoint.x + np.cos(angle_perpendicular) * half_length,
                midpoint.y + np.sin(angle_perpendicular) * half_length
            )
            endpoint2 = (
                midpoint.x - np.cos(angle_perpendicular) * half_length,
                midpoint.y - np.sin(angle_perpendicular) * half_length
            )
            perp_lines.append(LineString([endpoint1, endpoint2]))

        return perp_lines   
        
    @staticmethod
    def extract_raster_values(point, src):
        """
        Extracts raster values at the given point coordinates using the provided raster source.
        Parameters:
        - point (Point): The Shapely Point object representing the coordinates.
        - src: The raster source from which values will be extracted.
        Returns:
        - float: The extracted raster value at the given point, or np.nan if no valid value is obtained.
        """
        values = list(src.sample([(point.x, point.y)]))
        return float(values[0][0]) if values and not np.isnan(values[0][0]) else np.nan

    @staticmethod
    def calculate_relative_slope(df_point):
        """
        Function calculates the slope by finding the minimum elevation point ('RASTERVALU') and 
        then calculates the slope on both sides of this minimum point. It creates two separate
        DataFrames (left_df and right_df) for points on the left and right sides of the minimum
        point. The slope is calculated by dividing the difference in elevation by the difference
        in index values. Finally, it concatenates these DataFrames to get the final result.
        """
        min_index = df_point['raster_values'].idxmin()
        base_point = df_point.loc[min_index]
        left_df = df_point.loc[:min_index].copy()
        left_df['slope'] = (base_point['raster_values'] - left_df['raster_values']) / left_df['point_id'].diff()
        right_df = df_point.loc[min_index:].copy()
        right_df['slope'] = (right_df['raster_values'].shift(-1) - base_point['raster_values']) / right_df['point_id'].diff()
        df_point['slope'] = pd.concat([left_df['slope'], right_df['slope'].iloc[1:]])
        return df_point

    @staticmethod
    def filter_slope_range(df_point):
        """
        Function calculates the slope by finding the minimum elevation point ('RASTERVALU') and 
        then calculates the slope on both sides of this minimum point. It creates two separate
        DataFrames (left_df and right_df) for points on the left and right sides of the minimum
        point. The slope is calculated by dividing the difference in elevation by the difference
        in index values. Finally, it concatenates these DataFrames, filters out rows with slopes
        not in the range (-1, 1), and returns the final result.
        """
        min_index = df_point['raster_values'].idxmin()
        base_point = df_point.loc[min_index]
        left_df = df_point.loc[:min_index].copy()
        left_df['slope'] = (base_point['raster_values'] - left_df['raster_values']) / left_df['point_id'].diff()
        right_df = df_point.loc[min_index:].copy()
        right_df['slope'] = (right_df['raster_values'].shift(-1) - base_point['raster_values']) / right_df['point_id'].diff()
        df_point['slope'] = pd.concat([left_df['slope'], right_df['slope'].iloc[1:]])
        df_point = df_point.dropna(subset=['slope'])
        filtered_df = df_point[
            (df_point['slope'] >= -0.481299) &
            (df_point['slope'] <= 0.870400) &
            (df_point['point_id'].diff().eq(df_point['point_id'].diff().iloc[1]))
        ]
        filtered_df = filtered_df.reset_index(drop=True)
        return filtered_df

    @staticmethod
    def prune_bottom_to_top(df):
        '''
        start from bottom and move upward checking transect_id difference of each row 
        if difference is greater than 2 then delete all the rows above it
        '''
        for i in range(len(df)-1, 0, -1):
            current_id = df.loc[i, 'point_id']
            prev_id = df.loc[i-1, 'point_id']
            if abs(current_id - prev_id) > 2:
                df = df[i:]
                break
        return df
    
    @staticmethod

    def prune_middle_to_bottom(df):
        '''
        start from middle and move downward checking transect_id difference of each row 
        if difference is greater than 2 then delete all the rows above it
        '''
        middle_index = len(df) // 2

        for i, row in df.iloc[middle_index:].iterrows():
            current_id = row['point_id']

            if i < len(df) - 1:
                next_id = df.loc[i + 1, 'point_id']

                if abs(current_id - next_id) > 2:
                    return df.iloc[:i + middle_index + 1]

        return df
