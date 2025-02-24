import numpy as np
import math
from qgis.core import (
    QgsProject, QgsPointXY, QgsSpatialIndex, QgsRectangle,
    QgsCoordinateReferenceSystem, QgsProcessingException
)
from osgeo import gdal, osr
from qgis import processing

def get_input ():
    inputs = {} #Here to store all of our inputs

    #Ask for the type of analysis
    inputs['use_costmodel'] = input("\nChoose analysis mode: \n1. Cost-based\n2. Euclidean(classic)\nEnter 1 or 2: ") =='1'

    #Let's do some fun, let's scan user layers and make them select :)
    #1 a way to scan only multipoint layers ->function get_layers
    #2 enumerate said layers ->idem
    #3 selection of desire layer ->should be easy
    #4 Store the layer somewhere prop 1 point_layer = get_layers??
# -------------------USER INPUTS----------------------------------------
use_costmodel = True  # Set to False for Euclidean calculation
input_layer_name = 'yacimientos'
size_field = 'z'

# Cost model parameters
cost_layer_path = r'C:\Users\elpor\...\cumcost_lidarOK.tif' if use_costmodel else None
max_cost = 3600 if use_costmodel else None #Bc we dont really care in the other case

# Classic model parameters
max_distance = 5000  # Meters
beta = 2
inf = -9999
if beta <= 0: #Validate beta
    raise QgsProcessingException("Beta must be greater than zero!")

# -------------------SETTING OUTPUTS-------------------------------------
output_res = 1000
output_dominant_path = 'C:/.../dominant.tif'
output_influence_path = 'C:/.../influence.tif'
output_polygons_path = 'C:/.../influence_areas.gpkg'

# ------------------LOAD LAYERS + HANDLING ERRORS-----------------------------------------
try:
    input_layer = QgsProject.instance().mapLayersByName(input_layer_name)[0]
except IndexError:
    raise QgsProcessingException(f"Layer '{input_layer_name}' not found!")

cost_nodata = None
if use_costmodel:
    if not cost_layer_path:
        raise QgsProcessingException("Cost layer path required for cost model!")
    
    cost_raster = gdal.Open(cost_layer_path)
    if not cost_raster:
        raise QgsProcessingException("Failed to open cost raster!")
    
    cost_band = cost_raster.GetRasterBand(1)
    cost_nodata = cost_band.GetNoDataValue()
    cost_transform = cost_raster.GetGeoTransform()
    output_crs = QgsCoordinateReferenceSystem(cost_raster.GetProjection())
    x_min = cost_transform[0]
    y_max = cost_transform[3]

    width = cost_raster.RasterXSize
    height = cost_raster.RasterYSize
    
else:
    output_crs = input_layer.crs()
    if not output_crs.isValid():
        raise QgsProcessingException("Invalid layer CRS!")
    if output_crs.isGeographic():
        raise QgsProcessingException("Classic model requires projected CRS!")
    
    extent = input_layer.extent().buffered(max_distance)
    x_min = extent.xMinimum()
    y_max = extent.yMaximum()
    width = math.ceil((extent.width()) / output_res)
    height = math.ceil((extent.height()) / output_res)

# ------------------ARRAY INITIALIZATION--------------------------------
dominant_data = np.full((height, width), inf, dtype=np.float32)
influence_data = np.full((height, width), inf, dtype=np.float32)

# -----------------SPATIAL INDEX----------------------------------------
spatial_index = QgsSpatialIndex()
features = {}
for feature in input_layer.getFeatures():
    if feature.hasGeometry():
        features[feature.id()] = feature
        spatial_index.insertFeature(feature)

# -----------------ITERATE CELLS----------------------------------------
for row in range(height):
    for col in range(width):
        if use_costmodel:
            x_center = x_min + (col + 0.5) * cost_transform[1]
            y_center = y_max - (row + 0.5) * abs(cost_transform[5])
            cell_size_x = cost_transform[1]
            cell_size_y = abs(cost_transform[5])
        else:
            x_center = x_min + (col + 0.5) * output_res
            y_center = y_max - (row + 0.5) * output_res
            cell_size_x = cell_size_y = output_res

        current_point = QgsPointXY(x_center, y_center)
        search_rect = QgsRectangle(
            x_center - max_distance,
            y_center - max_distance,
            x_center + max_distance,
            y_center + max_distance
        )

        candidate_fids = spatial_index.intersects(search_rect)
        if not candidate_fids:
            continue

        max_influence = 0.0
        dominant_fid = None

        for fid in candidate_fids:
            feature = features[fid]
            point = feature.geometry().asPoint()
            
            if use_costmodel:
                try:
                    cost_array = cost_band.ReadAsArray(col, row, 1, 1)
                    cost = cost_array[0, 0] if cost_array else cost_nodata
                except:
                    cost = cost_nodata
                
                if cost == cost_nodata or cost <= 0 or cost > max_cost:
                    continue
                
                influence = feature[size_field] / (cost ** beta) #XTENT shout out to my boy renfrew
            else:
                dx = point.x() - x_center
                dy = point.y() - y_center
                distance = math.hypot(dx, dy)
                distance = max(distance, 1e-9)  # Prevent division by zero was giving me problems all the weekend while testing
                
                if distance > max_distance:
                    continue
                
                influence = feature[size_field] / (distance ** beta)

            if influence > max_influence:
                max_influence = influence
                dominant_fid = fid

        if max_influence > 0:
            dominant_data[row, col] = dominant_fid if dominant_fid is not None else inf
            influence_data[row, col] = max_influence

# ------------------SAVE RASTERS----------------------------------------
def save_raster(path, data, transform_params, use_cost): #Function bc at the end of the day they are saved the same way
    driver = gdal.GetDriverByName('GTiff')
    if use_cost:
        ds = driver.Create(path, width, height, 1, gdal.GDT_Float32)
        ds.SetGeoTransform(transform_params)
    else:
        ds = driver.Create(path, width, height, 1, gdal.GDT_Float32)
        ds.SetGeoTransform(transform_params)
    
    ds.SetProjection(output_crs.toWkt())
    ds.GetRasterBand(1).WriteArray(data)
    ds.GetRasterBand(1).SetNoDataValue(inf)
    ds = None

# Save both rasters
if use_costmodel:
    gt = (x_min, cost_transform[1], 0, y_max, 0, -abs(cost_transform[5]))
else:
    gt = (x_min, output_res, 0, y_max, 0, -output_res)

save_raster(output_dominant_path, dominant_data, gt, use_costmodel)
save_raster(output_influence_path, influence_data, gt, use_costmodel)

# -----------------POLYGONIZATION---------------------------------------
processing.run("gdal:polygonize", {
    'INPUT': output_dominant_path,
    'BAND': 1,
    'FIELD': 'dominant_id',
    'OUTPUT': output_polygons_path
})

# -----------------SMOOTHING--------------------------------------------
smoothy_polygons_path = 'C:/.../smoothed.gpkg'
processing.run("native:smoothgeometry", {
    'INPUT': output_polygons_path,
    'ITERATIONS': 3, #The greater the smmother
    'OFFSET': 0.2,
    'OUTPUT': smoothy_polygons_path
})

# -----------------LOAD RESULTS-----------------------------------------
iface.addRasterLayer(output_dominant_path, 'Dominant_IDs')
iface.addRasterLayer(output_influence_path, 'Influence_Values')
iface.addVectorLayer(smoothy_polygons_path, 'Smoothed_Influence', 'ogr')

print(f"XTENT processing completed using {'cost-based' if use_costmodel else 'classic'} model!")
