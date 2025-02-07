# XTENT: Influence Extent Calculation for QGIS

XTENT is a Python-based tool designed to compute spatial influence extents from point features using a customizable distance decay formula (defaulting to the Renfrew formulation). The script integrates with QGIS and GDAL to generate both raster and vector outputs representing the dominant influence areas of point features based on user-defined parameters.

## Overview

The script performs the following tasks:
- **Input Data Retrieval:** Loads a point layer (e.g., sites or "yacimientos") and a cumulative cost raster (e.g., generated from GRASS's `r.walk`) from the current QGIS project.
- **Raster Grid Generation:** Constructs a raster grid over a buffered extent of the input points.
- **Influence Calculation:** For each cell in the raster grid, it calculates the influence of nearby points using a distance decay formula. The default formula is:  
  `influence = size / (distance^beta)`  
  with a special case for zero distance.
- **Output Generation:** 
  - Writes two GeoTIFF raster files:
    - A **Dominant IDs** raster, where each cell holds the ID of the point exerting the maximum influence.
    - An **Influence Values** raster, which stores the calculated influence values.
  - Polygonizes the Dominant IDs raster to produce vector polygons and applies a smoothing algorithm to improve their visual appearance.
- **QGIS Integration:** Loads the generated layers directly into QGIS for immediate visualization and further analysis.

## Features

- **Customizable Parameters:** Easily adjust input layer names, output paths, decay parameters (`beta`, `max_distance`), and raster resolution.
- **Spatial Indexing:** Utilizes QGIS’s spatial index for efficient proximity searches.
- **GDAL Integration:** Outputs standardized GeoTIFF files and leverages GDAL’s polygonization.
- **Processing Tools:** Uses QGIS Processing algorithms to smooth the output polygons.

## Requirements

- **QGIS 3.x or later:** The script is designed to run within the QGIS Python environment.
- **Python Libraries:**
  - `math`
  - `numpy`
  - `osgeo` (GDAL and OSR)
  - QGIS Python modules (`qgis.core`, `qgis`, `processing`)

Make sure your QGIS installation includes these libraries (typically available by default in the QGIS Python environment).

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/chinijo21/xtent-qgis.git
    Open the project in QGIS:
        Load your point and cumulative cost raster layers into QGIS.
        Open the Python Console in QGIS and run the script, or add it as a custom processing script.

## Usage
  1. Prepare Your Data:
        -Point Layer: Ensure you have a point layer (default name: yacimientos) loaded in QGIS with a numeric field (default: z) representing the size or relevance of each point.
        -Cost Raster: Load your cumulative cost raster layer (default name: cumcost_lidarOk) into QGIS.

  2. Configure Parameters: Open the script and modify the following user inputs as needed:
        -input_layer_name: Name of your point layer.
        -size_field: Field containing the numeric size/relevance value.
        -cost_layer_name: Name of your cumulative cost raster.
        -max_cost: Maximum travel time threshold.
        -beta: Distance decay exponent.
        -max_distance: Maximum influence distance (in meters).
        -Output paths for the dominant IDs raster, influence raster, and polygon file.

  3. Run the Script: Execute the script from the QGIS Python Console or as a processing tool. The script will:
        -Build a spatial index for the point features.
        -Loop through each cell in a generated raster grid.
        -Calculate influence values based on the proximity and size of each point.
        -Write the output rasters and vector polygons to disk.
        -Automatically load the results into QGIS.

  4. Review the Results:
        -Dominant IDs Raster: Each cell indicates the ID of the point with the highest influence.
        -Influence Values Raster: Contains the computed influence for each cell.
        -Smoothed Polygons: Vector layer representing influence areas with smoothed boundaries.

## Code Structure

  - User Inputs:
    Set up the parameters (input layers, fields, decay parameters, output file paths) at the beginning of the script.

  - Layer & CRS Setup:
    Retrieves input layers from QGIS and ensures consistent coordinate reference systems (CRS).

  - Raster Processing Loop:
    Iterates over each raster cell, computes distances to nearby point features, and calculates influence using a customizable decay function.

  - Raster and Polygon Output:
        Writes the dominant ID and influence values as GeoTIFF files.
        Uses GDAL's polygonize tool to convert the dominant raster into vector polygons.
        Applies a smoothing algorithm to refine the polygon boundaries.

  - QGIS Integration:
        The final output layers are added to the current QGIS project for immediate visualization.

## Customization
  - Distance Decay Formula: modify the influence calculation section in the script to experiment with different decay functions (e.g., exponential, logarithmic).

  - Smoothing Parameters: adjust the number of iterations and offset in the smoothing algorithm (native:smoothgeometry) to control the smoothness of the polygons.

## WIP
  Using a cost surface instead of euclidean distance

Contributing

Contributions, suggestions, and improvements are welcome. If you encounter any issues or have ideas for new features, please open an issue or submit a pull request.
