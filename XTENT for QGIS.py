import math
import numpy as np
from qgis.core import (
    QgsProject, QgsPointXY, QgsSpatialIndex, QgsRectangle,
    QgsCoordinateReferenceSystem
)
from osgeo import gdal, osr
from qgis import processing

# Input parameters (adjust these according to your data)
input_layer_name = 'yacimientos'  # Name of the input point layer in QGIS
size_field = 'z'    # Field name representing the size (e.g., population)
beta = 2                     # Distance decay exponent
max_distance = 5000  # Maximum influence distance in meters
output_resolution = 1000     # Raster cell size in meters
output_dominant_path = 'C:/Users/elpor/Desktop/dominant.tif'  # Output path for dominant IDs raster
output_influence_path = 'C:/Users/elpor/Desktop/influence.tif'  # Output path for influence values raster
output_polygons_path = 'C:/Users/elpor/Desktop/influence_areas.gpkg'  # Output path for polygons

# Retrieve the input layer and its CRS
input_layer = QgsProject.instance().mapLayersByName(input_layer_name)[0]
output_crs = input_layer.crs()

# Calculate buffered extent to cover max_distance around points
extent = input_layer.extent()
buffered_extent = extent.buffered(max_distance)
x_min = buffered_extent.xMinimum()
y_max = buffered_extent.yMaximum()
res = output_resolution

# Determine raster dimensions
width = int((buffered_extent.xMaximum() - x_min) / res)
height = int((y_max - buffered_extent.yMinimum()) / res)

# Initialize numpy arrays for storing results (filled with NoData)
dominant_data = np.full((height, width), -9999, dtype=np.float32)
influence_data = np.full((height, width), -9999, dtype=np.float32)

# Build spatial index for efficient spatial queries
spatial_index = QgsSpatialIndex()
features = {feature.id(): feature for feature in input_layer.getFeatures()}
for fid, feature in features.items():
    spatial_index.insertFeature(feature)

# Iterate through each cell in the raster
for row in range(height):
    for col in range(width):
        # Calculate cell center coordinates
        x_center = x_min + (col + 0.5) * res
        y_center = y_max - (row + 0.5) * res
        current_point = QgsPointXY(x_center, y_center)
        
        # Define search area around the cell
        search_rect = QgsRectangle(
            x_center - max_distance,
            y_center - max_distance,
            x_center + max_distance,
            y_center + max_distance
        )
        # Query features within the search area
        candidate_fids = spatial_index.intersects(search_rect)
        if not candidate_fids:
            continue  # Skip if no candidates
        
        max_influence = 0.0
        dominant_fid = None
        
        for fid in candidate_fids:
            feature = features[fid]
            point = feature.geometry().asPoint()
            dx = point.x() - x_center
            dy = point.y() - y_center
            distance = math.hypot(dx, dy)
            
            if distance > max_distance:
                continue  # Skip points beyond max_distance
            
            size = feature[size_field]
            if not size or size <= 0:
                continue  # Skip points with invalid size
            
            # Calculate influence (handle zero distance)
            if distance == 0:
                influence = 1e9  # Use a large value instead of infinity
            else:
                influence = size / (distance ** beta)
            
            # Track maximum influence and dominant feature
            if influence > max_influence:
                max_influence = influence
                dominant_fid = fid
        
        # Update raster data if influence is valid
        if max_influence > 0:
            dominant_data[row, col] = dominant_fid
            influence_data[row, col] = max_influence

# Write dominant IDs raster to GeoTIFF
driver = gdal.GetDriverByName('GTiff')
ds_dominant = driver.Create(
    output_dominant_path, width, height, 1, gdal.GDT_Float32
)
ds_dominant.SetGeoTransform((x_min, res, 0, y_max, 0, -res))
ds_dominant.SetProjection(output_crs.toWkt())
ds_dominant.GetRasterBand(1).WriteArray(dominant_data)
ds_dominant.GetRasterBand(1).SetNoDataValue(-9999)
ds_dominant = None  # Close dataset

# Write influence values raster to GeoTIFF
ds_influence = driver.Create(
    output_influence_path, width, height, 1, gdal.GDT_Float32
)
ds_influence.SetGeoTransform((x_min, res, 0, y_max, 0, -res))
ds_influence.SetProjection(output_crs.toWkt())
ds_influence.GetRasterBand(1).WriteArray(influence_data)
ds_influence.GetRasterBand(1).SetNoDataValue(-9999)
ds_influence = None

# Polygonize the dominant IDs raster into vector polygons
processing.run("gdal:polygonize", {
    'INPUT': output_dominant_path,
    'BAND': 1,
    'FIELD': 'dominant_id',
    'OUTPUT': output_polygons_path
})

# Load the results into QGIS
iface.addRasterLayer(output_dominant_path, 'Dominant_IDs')
iface.addRasterLayer(output_influence_path, 'Influence_Values')
iface.addVectorLayer(output_polygons_path, 'Influence_Areas', 'ogr')

print("XTENT processing completed!")
