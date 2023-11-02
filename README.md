# DEM Waterway Refinement GIS Tool

## Introduction

Flooding poses significant threats to communities and ecosystems, making accurate flood modeling a crucial aspect of disaster preparedness and environmental management. High-resolution Digital Elevation Models (DEMs) serve as the foundation for these models, providing detailed topographic information essential for simulating flood scenarios. However, the challenge arises when raw Lidar data results in hydrologically disconnected DEMs, impacting the accuracy of flood model simulations.


The hydrological disconnection in Digital Elevation Models (DEMs) occurs as a consequence of Lidar scanning, which captures intricate details, including bridges and culverts, from an aerial perspective. When viewed from above, these structures might appear as disruptions in the stream network, leading Lidar to interpret them as barriers, akin to dams, and causing the DEM to represent disconnected waterways. In reality, water flows beneath bridges and within culverts, but Lidar's top-down interpretation introduces a spatial misrepresentation. Correcting this issue demands meticulous manual efforts by modelers to differentiate between genuine topographic features and artificial disruptions. The process involves recognizing and adjusting Lidar-induced misinterpretations, making the hydrological correction of DEMs a time-consuming and labor-intensive task.

Hydrologically disconnected DEMs can lead to inaccuracies in predicting flood patterns and waterway behaviors. When modelling structures like bridges, culverts, and man-made structures can disrupt the natural flow of water, creating artificial stream flows or, in severe cases, causing water pooling. Addressing this issue requires extensive time and effort from modelers, who must manually correct DEMs to ensure hydrological connectivity and remove obstructions.

In response to these challenges, the DEM Waterway Refinement GIS Tool emerges as a groundbreaking solution. This tool focuses on automating the preprocessing of DEMs, particularly in the identification and removal of artificial obstructions within waterways. By doing so, it aims to streamline the time-consuming and labor-intensive processes associated with flood modeling, ultimately improving the accuracy of simulations.

## Objectives

The DEM Waterway Refinement GIS Tool has the following key objectives:

1. **Automating DEM Preprocessing:**
   - Develop a robust GIS tool capable of automating the traditionally manual and time-consuming processes involved in preparing Digital Elevation Models for flood modeling.

2. **Enhancing Accuracy in Flood Modeling:**
   - Improve the accuracy of flood modeling by systematically detecting and removing bridges, culverts, and other unnatural obstructions within waterways, optimizing the natural flow of water.

3. **Open-Source Accessibility:**
   - Design the tool as an ArcGIS Pro Tool while also doubling as an open-source plugin for GIS platforms like QGIS. This ensures widespread accessibility and usability across different GIS environments.

4. **Revolutionizing Flood Modeling Processes:**
   - Leverage cutting-edge geospatial libraries and advanced techniques to revolutionize flood modeling processes, significantly reducing the time spent on geospatial data preprocessing.

These objectives collectively contribute to the overarching goal of providing a powerful and accessible tool that revolutionizes flood modeling and waterway analysis, making it more efficient, accurate, and user-friendly.


## Technical Details

- **Programming Language:** Python
- **GIS Platforms:** ArcGIS Pro, QGIS (as an open-source plugin)
- **Geospatial Libraries:** ArcGIS Pro Model Builder, Pandas, Numpy, Seaborn, Plotly, Scikit-learn
- **Dependencies:** Git (version control)
- **Skills:** Analytics, Data Science, Statistical Tools

## Workflow

### Description: 
The DEM Waterway Refinement GIS Tool follows a systematic workflow to automate the preprocessing of Digital Elevation Models (DEMs) and enhance accuracy in flood modeling. The workflow is organized into the following concise sections:

### Inputs:

- DEM
- Dissolved Smoothed Stream Network


### Methodology


#### Step 1: Watershed Delineation
- **Description:** In the first step we take the DEM and 

This initial step involves the extraction of watershed boundaries from the smoothed stream network, forming the foundation for subsequent processing.

#### Step 2: Preprocessing - ETL Operation
- **Description:** This section focuses on the Extraction, Transformation, and Loading (ETL) operations required for efficient DEM refinement.
  1. **Load Stream Network:** Import the smoothed stream network into the tool.
  2. **Split Lines:** Divide the stream lines into equal intervals.
  3. **Perpendicular Drawing:** Employ an advanced algorithm to draw perpendicular lines on each split, optimizing accuracy.
  4. **Point Generation:** Create points along each perpendicular line, contributing to a more precise mesh capturing waterway depths and widths.

#### Step 3: Transform
- **Operations:**
  - **Sample:** Sample elevation data from the DEM.
  - **Slope:** Calculate the slope of the terrain.
  - **Prune:** Refine the DEM by pruning unnecessary data, optimizing for hydrological accuracy.

#### Step 4: Advanced Operations
- **Operations:**
  - **Extract Waterway Banks:** Identify and extract waterway banks for detailed analysis.
  - **Concave Hull:** Apply a concave hull algorithm to outline the waterway's shape accurately.
  - ** 
#### **Step 5: More steps to update**

This organized workflow ensures a clear and logical progression through each stage of the DEM refinement process. Users can easily follow these steps, gaining insights into the underlying logic and implementation details at each major phase.

## Visual Aids:

Graphics to be attached.
