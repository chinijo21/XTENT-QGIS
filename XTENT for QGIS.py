import numpy as np
from qgis.core import (
    QgsProject, QgsPointXY, QgsSpatialIndex, QgsRectangle,
    QgsCoordinateReferenceSystem
)
from osgeo import gdal, osr
from qgis import processing

# -------------------USER INPUTS----------------------------------------
#At some point this will be done via QGIS at least inputing the path in the console
# Input parameters (adjust these according to your data)
use_costmodel = True #Set to false for an euclidean calc.

input_layer_name = 'yacimientos' # Name of the input point layer in QGIS
size_field = 'z' # Field name representing the size (population, relevance...) MUST BE A NUMBER

#Cost model params (only if cost = true)
cost_layer_path = r'C:\Users\elpor\...\cumcost_lidarOK.tif' if use_costmodel else None # Your cummulative cost, in this case using r.walk from GRASS. Forgot the extension - Use high definition data where possible
max_cost = 3600 if use_costmodel else None #Max travel time

#For both models
beta = 2    #Put your beta as you wish, might need recalibration
max_distance = 5000 #In meters
inf = -9999 #Just in case we go oob

# -------------------SETTING OUTPUTS-------------------------------------
output_res = 1000 # I meters for classic model
output_dominant_path = 'C:/.../dominant.tif' # Output path for dominant IDs raster
output_influence_path = 'C:/.../influence.tif' # Output path for influence values raster
output_polygons_path = 'C:/.../influence_areas.gpkg'  # Output path for polygons


# ------------------LOAD LAYERS + CRS-----------------------------------
#We need to get sites + cumcost
# Retrieve layers and its CRS
input_layer = QgsProject.instance().mapLayersByName(input_layer_name)[0]
#Cost model set
if use_costmodel:
    cost_raster = gdal.Open(cost_layer_path)
    cost_band = cost_raster.GetRasterBand(1)
    cost_transform = cost_raster.GetGeoTransform()
    output_crs = QgsCoordinateReferenceSystem(cost_raster.GetProjection())  # Use cost layer's CRS should be the same as the project

# -------------RASTER DIMENSIONS + ARRAYS-------------------------------
    #Previous version wasn't accountig for the cost layer's res (x_res - y_res instead)
    x_min = cost_transform[0]
    y_max = cost_transform[3]
    x_res = cost_transform[1]
    y_res = abs(cost_transform[5])  # :) I almost jumped of a bridge

    # Determine raster dimensions
    width = cost_raster.RasterXSize
    height = cost_raster.RasterYSize
else:
    #Classic model taken from v1
    output_crs = input_layer.crs()
    extent = input_layer.extent().buffered(max_distance)
    x_min = extent.xMinimum()
    y_max = extent.yMaximum()
    width = int((extent.xMaximum() - x_min) / output_res)
    height = int((y_max - extent.yMinimum()) / output_res)
   
# Initialize numpy arrays for storing results (filled with NoData)
# Might throw an error if you have null values WIP
dominant_data = np.full((height, width), inf, dtype=np.float32)
influence_data = np.full((height, width), inf, dtype=np.float32)

# -----------------SPATIAL INDEX----------------------------------------
# Build spatial index for efficient spatial queries
spatial_index = QgsSpatialIndex()
features = {f.id(): f for f in input_layer.getFeatures()}
for fid, feature in features.items():
    spatial_index.insertFeature(feature)

# -----------------ITERATE CELLS----------------------------------------
# Iterate through each cell in the raster
for row in range(height):
    for col in range(width):
        x_center = x_min + (col + 0.5) * x_res
        y_center = y_max - (row + 0.5) * y_res
        current_point = QgsPointXY(x_center, y_center)
        # Define search area around the cell (taken directly from QGIS manual)
        search_rect = QgsRectangle(
            x_center - max_distance * x_res,
            y_center - max_distance * y_res,
            x_center + max_distance * x_res,
            y_center + max_distance * y_res
        )
        # Query features within the search area
        candidate_fids = spatial_index.intersects(search_rect)
        if not candidate_fids:
            continue
        
        max_influence = 0.0
        dominant_fid = None
         #Iterate through fids. Keeps trakc of maximum influence value and the corresponding point (dominant_fid) for each
        for fid in candidate_fids:
            feature = features[fid]
            point = feature.geometry().asPoint()
            
            # Read cost from the precomputed layer
            cost_array = cost_band.ReadAsArray(col, row, 1, 1)
            cost = cost_array[0, 0] if cost_array is not None else inf
            
            if cost <= 0 or cost > max_cost:
                continue
            
            size = feature[size_field]
            if not size or size <= 0:
                continue
                #XTENT: modify as you wish
                #log
                #influence = size / (1 + math.log(distance))
                #expo
                #influence = size * math.exp(-beta * distance)
                #Classic
            influence = size / (cost ** beta)
            # Track maximum influence and dominant feature
            if influence > max_influence:
                max_influence = influence
                dominant_fid = fid
        # Update raster data if influence is valid
        if max_influence > 0:
            dominant_data[row, col] = dominant_fid
            influence_data[row, col] = max_influence

# ------------------SAVE RASTERS----------------------------------------
driver = gdal.GetDriverByName('GTiff')
ds_dominant = driver.Create(
    output_dominant_path, width, height, 1, gdal.GDT_Float32
)
ds_dominant.SetGeoTransform((x_min, x_res, 0, y_max, 0, -y_res))  # Use cost layer's resolution
ds_dominant.SetProjection(output_crs.toWkt())
ds_dominant.GetRasterBand(1).WriteArray(dominant_data)
ds_dominant.GetRasterBand(1).SetNoDataValue(inf)
ds_dominant = None

ds_influence = driver.Create(
    output_influence_path, width, height, 1, gdal.GDT_Float32
)
ds_influence.SetGeoTransform((x_min, x_res, 0, y_max, 0, -y_res))
ds_influence.SetProjection(output_crs.toWkt())
ds_influence.GetRasterBand(1).WriteArray(influence_data)
ds_influence.GetRasterBand(1).SetNoDataValue(inf)
ds_influence = None

# -----------------POLYGONIZE + SMOOTH----------------------------------
processing.run("gdal:polygonize", {
    'INPUT': output_dominant_path,
    'BAND': 1,
    'FIELD': 'dominant_id',
    'OUTPUT': output_polygons_path
})

smoothy_polygons_path = 'C:/.../smoothed.gpkg'
processing.run("native:smoothgeometry", {
    'INPUT': output_polygons_path,
    'ITERATIONS': 3,
    'OFFSET': 0.2,
    'OUTPUT': smoothy_polygons_path
})

# -----------------LOAD RESULTS-----------------------------------------
iface.addRasterLayer(output_dominant_path, 'Dominant_IDs')
iface.addRasterLayer(output_influence_path, 'Influence_Values')
iface.addVectorLayer(smoothy_polygons_path, 'Smoothed_Influence', 'ogr')

print("XTENT with cost distance completed!")
