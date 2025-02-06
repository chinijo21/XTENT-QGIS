import math
import numpy as np
from qgis.core import (
    QgsProject, QgsPointXY, QgsSpatialIndex, QgsRectangle,
    QgsCoordinateReferenceSystem
)
from osgeo import gdal, osr
from qgis import processing

#-------------------USER INPUTS----------------------------------------
#At some point this will be done via QGIS at least inputing the path in the console
# Input parameters (adjust these according to your data)
input_layer_name = 'yacimientos'  # Name of the input point layer in QGIS
size_field = 'z'    # Field name representing the size (population, relevance...) MUST BE A NUMBER
cost_layer_name = 'cumcost_lidarOk' #Your cummulative cost, in this case using r.walk from GRASS, eventually I 
max_cost = 3600       #Maximum travel time
beta = 2                     # Distance decay exponent -> Renfrew formula
max_distance = 5000  # Maximum influence distance in meters

#-------------------SETTING OUTPUTS-------------------------------------
output_resolution = 1000     # Raster cell size in meters
output_dominant_path = 'C:/Users/elpor/Desktop/dominant.tif'  # Output path for dominant IDs raster
output_influence_path = 'C:/Users/elpor/Desktop/influence.tif'  # Output path for influence values raster
output_polygons_path = 'C:/Users/elpor/Desktop/influence_areas.gpkg'  # Output path for polygons

#-----------------GETTING LAYERS + CRS + TRANSFORMS --------------------
#We need to get sites + cumcost
# Retrieve layers and its CRS
input_layer = QgsProject.instance().mapLayersByName(input_layer_name)[0]
output_crs = input_layer.crs() #So we make sure we don't make a mess and export the final results in some random CRS

#Cost layer to raster GDAL !!WIP
cost_layer = QgsProject.instance().mapLayersByName(cost_layer_name)[0] #Let's import it directly bc im tired of writing paths
cost_raster = gdal.Open(cost_layer)
cost_band = cost_raster.GetRasterBand(1) #Gets the band with the info that we need
cost_transform = cost_raster.GetGeoTransform()
output_crs = QgsCoordinateReferenceSystem(cost_raster.GetProjection())

#-------------ACTUAL CALCULATION-------------------------------------------
# Calculate buffered extent to cover max_distance around points -> You can change this as you please (Agricol + pastoral = 5km radius)
extent = input_layer.extent()
buffered_extent = extent.buffered(max_distance)
x_min = buffered_extent.xMinimum()
y_max = buffered_extent.yMaximum()
res = output_resolution #Default to your project in QGIS

# Determine raster dimensions
width = int((buffered_extent.xMaximum() - x_min) / res)
height = int((y_max - buffered_extent.yMinimum()) / res)

# Initialize numpy arrays for storing results (filled with NoData)
# Might throw an error if you have null values WIP
dominant_data = np.full((height, width), -9999, dtype=np.float32)
influence_data = np.full((height, width), -9999, dtype=np.float32)

# Build spatial index for efficient spatial queries
spatial_index = QgsSpatialIndex()
features = {feature.id(): feature for feature in input_layer.getFeatures()}
for fid, feature in features.items():   #Go through your shp file and sites!
    spatial_index.insertFeature(feature)

# Iterate through each cell in the raster
for row in range(height):
    for col in range(width):
        # Calculate cell center coordinates
        x_center = x_min + (col + 0.5) * res
        y_center = y_max - (row + 0.5) * res
        current_point = QgsPointXY(x_center, y_center)
        
        # Define search area around the cell (taken directly from QGIS manual)
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
        #Iterate through fids. Keeps trakc of maximum influence value and the corresponding point (dominant_fid) for each
        for fid in candidate_fids:
            feature = features[fid] #Candidate points
            point = feature.geometry().asPoint()
            #Euclidian distance calculation
            #It finds the straight line distance from a point to x and y centers
            dx = point.x() - x_center #Horizontal dif
            dy = point.y() - y_center #Vertical dif
            distance = math.hypot(dx, dy) #math.sqrt(Dx2 + dy2) proves to be a little unstable
            #WIP retrieve costs from cumcost surface!¿!¿
#---------------------------ACTUAL XTENT CALCULATION-------------------------------------------            
            if distance > max_distance:
                continue  # Skip points beyond max_distance
            
            size = feature[size_field]
            if not size or size <= 0:
                continue  # Skip points with invalid size bc handling errors is not funny
            
            # Calculate influence (handle zero distance)
            if distance == 0:
                influence = 1e9  # Use a large value so it shows coherently. Two close sites might be related and share things (especially in tribal clanic)
            else:
                #XTENT: modify as you wish
                #log
                #influence = size / (1 + math.log(distance))
                #expo
                #influence = size * math.exp(-beta * distance)
                #Classic
                influence = size / (distance ** beta) # <----------------------------ACTUAL XTENT FORMULA CHECK RENFREW 
            
            # Track maximum influence and dominant feature
            if influence > max_influence:
                max_influence = influence
                dominant_fid = fid
        
        # Update raster data if influence is valid
        if max_influence > 0:
            dominant_data[row, col] = dominant_fid
            influence_data[row, col] = max_influence

#----------------RASTER + POLIGONS---------------------------------------------
# Write dominant IDs raster to GeoTIFF
driver = gdal.GetDriverByName('GTiff')
ds_dominant = driver.Create(
    output_dominant_path, width, height, 1, gdal.GDT_Float32
)
ds_dominant.SetGeoTransform((x_min, res, 0, y_max, 0, -res))
ds_dominant.SetProjection(output_crs.toWkt())
ds_dominant.GetRasterBand(1).WriteArray(dominant_data)
ds_dominant.GetRasterBand(1).SetNoDataValue(-9999) #Do not touch, only works like this (??)
ds_dominant = None  # Close dataset (me voy a suicidar)

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

# Instead of breaking my head, let's smoothify the polygons that we just made so they look circleish :)
smoothy_polygons_path = 'C:/Users/elpor/Desktop/smoooooth.gpkg'
processing.run("native:smoothgeometry", {
    'INPUT': output_polygons_path,
    'ITERATIONS': 5,          # Increase for smoother edges (computanial heavy as we increase the number but pretty)
    'OFFSET': 0.25,           # Controls rounding strength
    'OUTPUT': smoothy_polygons_path
})

#-------------ADDING THE DATA WE JUST GENERATED INTO QGIS---------------------------------
# Load the results into QGIS
iface.addRasterLayer(output_dominant_path, 'Dominant_IDs')
iface.addRasterLayer(output_influence_path, 'Influence_Values')
iface.addVectorLayer(smoothy_polygons_path, 'Smothy_influenced', 'ogr')

print("XTENT processing completed with smoothed polygons!")
