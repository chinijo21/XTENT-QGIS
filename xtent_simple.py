"""
XTENT Territorial Model - Simple Editor Version
Configure the parameters below and run from QGIS Python Editor
"""

from qgis.core import *
from osgeo import gdal
import math
import numpy as np
import os

# ============================================
# CONFIGURE THESE PARAMETERS
# ============================================

# Input shapefile path (CHANGE THIS to your file path)
INPUT_SHAPEFILE = "/path/to/your/multipoint.shp"

# XTENT Parameters
A_COEFFICIENT = 0.8  # Exponential coefficient for center weight
K_COEFFICIENT = 2.0  # Linear coefficient for distance
STRICT_MODE = False  # Set to True to enforce I >= 0 constraint

# Smoothing Parameters
SMOOTH_POLYGONS = True  # Set to True to create rounded polygon boundaries
SMOOTHING_ITERATIONS = 3  # Number of smoothing passes (1-5, higher = rounder)
SMOOTHING_OFFSET = 0.25  # Smoothing strength (0.0-0.5, higher = rounder)

# ============================================
# DO NOT EDIT BELOW THIS LINE
# ============================================

MARGIN = 0.01


def xtent_influence(x, y, x0, y0, weight, a, k):
    """Calculate XTENT influence: I = C^a - k*d"""
    distance = math.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    influence = (weight ** a) - (k * distance)
    return influence


def normalize_weights(points):
    """Normalize center weights to range [0..1]"""
    if not points:
        return points
    
    weights = [p[2] for p in points]
    min_weight = min(weights)
    max_weight = max(weights)
    
    if max_weight == min_weight:
        return [[p[0], p[1], 1.0] for p in points]
    
    normalized_points = []
    for p in points:
        norm_weight = (p[2] - min_weight) / (max_weight - min_weight)
        normalized_points.append([p[0], p[1], norm_weight])
    
    return normalized_points


def smooth_geometry(geom, iterations=3, offset=0.25):
    """
    Apply Chaikin's algorithm to smooth polygon geometries.
    Creates rounded, organic-looking boundaries.
    
    Parameters:
        geom: QgsGeometry to smooth
        iterations: Number of smoothing iterations (1-5)
        offset: Smoothing offset (0.0-0.5, controls roundness)
    
    Returns:
        Smoothed QgsGeometry
    """
    if geom.isNull() or geom.isEmpty():
        return geom
    
    for _ in range(iterations):
        if geom.isMultipart():
            # Handle multipart geometries
            smoothed_parts = []
            for part in geom.asGeometryCollection():
                smoothed_part = part.smooth(iterations=1, offset=offset)
                if smoothed_part and not smoothed_part.isNull():
                    smoothed_parts.append(smoothed_part)
            if smoothed_parts:
                geom = QgsGeometry.collectGeometry(smoothed_parts)
        else:
            # Single part geometry
            smoothed = geom.smooth(iterations=1, offset=offset)
            if smoothed and not smoothed.isNull():
                geom = smoothed
    
    return geom


# Main execution
print("=" * 60)
print("XTENT Territorial Model - Simple Editor Version")
print("=" * 60)
print(f"Input file: {INPUT_SHAPEFILE}")
print(f"Parameters: a={A_COEFFICIENT}, k={K_COEFFICIENT}")
print(f"Strict mode (I>=0): {STRICT_MODE}")
print("=" * 60)

# Load the points vector layer
pointsVector = QgsVectorLayer(INPUT_SHAPEFILE, 'centers', 'ogr')
if not pointsVector.isValid():
    print(f"\n❌ ERROR: Failed to load input file: {INPUT_SHAPEFILE}")
    print("\n👉 Please check:")
    print("   1. The file path is correct")
    print("   2. The file exists")
    print("   3. The file is a valid shapefile")
    raise Exception(f"Failed to load {INPUT_SHAPEFILE}")

# Check geometry type and handle MultiPoint
wkb_type = pointsVector.wkbType()
print(f"\nGeometry type detected: {QgsWkbTypes.displayString(wkb_type)}")

# If MultiPoint, convert to individual points
if wkb_type in [QgsWkbTypes.MultiPoint, QgsWkbTypes.MultiPoint25D, QgsWkbTypes.MultiPointZ, QgsWkbTypes.MultiPointM, QgsWkbTypes.MultiPointZM]:
    print("Converting MultiPoint to individual Points...")
    
    # Create a temporary Point layer
    crs = pointsVector.crs().authid()
    temp_layer = QgsVectorLayer(f"Point?crs={crs}", "temp_points", "memory")
    temp_provider = temp_layer.dataProvider()
    
    # Copy attributes from source layer
    temp_provider.addAttributes(pointsVector.fields())
    temp_layer.updateFields()
    
    # Extract individual points from MultiPoint features
    point_count = 0
    for feature in pointsVector.getFeatures():
        geom = feature.geometry()
        attrs = feature.attributes()
        
        if geom.isMultipart():
            # Extract each point from the multipoint
            points = geom.asMultiPoint()
            for point in points:
                new_feature = QgsFeature()
                new_feature.setGeometry(QgsGeometry.fromPointXY(point))
                new_feature.setAttributes(attrs)
                temp_provider.addFeature(new_feature)
                point_count += 1
        else:
            # Just copy single point
            new_feature = QgsFeature()
            new_feature.setGeometry(geom)
            new_feature.setAttributes(attrs)
            temp_provider.addFeature(new_feature)
            point_count += 1
    
    temp_layer.updateExtents()
    print(f"✓ Extracted {point_count} individual points from MultiPoint features")
    
    # Use the temporary layer instead
    pointsVector = temp_layer

# Add the vector layer to the map layer registry
QgsProject.instance().addMapLayer(pointsVector)

# Get layer extents with margin
bounding_box = pointsVector.extent()
extent_args = "-te " + str(bounding_box.xMinimum() - MARGIN) \
    + " " + str(bounding_box.yMinimum() - MARGIN) \
    + " " + str(bounding_box.xMaximum() + MARGIN) \
    + " " + str(bounding_box.yMaximum() + MARGIN)

# Rasterize the points with their weights
print("\n[1/5] Rasterizing input points...")
raster_temp = '/tmp/temp_rasterPoints.tif'

# Get the layer name for gdal_rasterize
layer_name = os.path.splitext(os.path.basename(INPUT_SHAPEFILE))[0]

os.system(f'gdal_rasterize -a z -ts 1000 1000 {extent_args} -l {layer_name} "{INPUT_SHAPEFILE}" "{raster_temp}"')

# Load the rasterized points
rasterPoints = QgsRasterLayer(raster_temp, 'rasterPoints')
QgsProject.instance().addMapLayer(rasterPoints)

# Read raster data
dataset = gdal.Open(raster_temp)
if dataset is None:
    print(f"\n❌ ERROR: Failed to create raster from input points")
    print("\n👉 Please check:")
    print("   1. Your shapefile has a 'z' attribute")
    print("   2. The 'z' attribute contains numeric values > 0")
    raise Exception("Failed to rasterize points")

numpy_array = dataset.ReadAsArray()

width, height = numpy_array.shape
points = []

# Extract all weighted points from the raster
print("\n[2/5] Extracting centers and weights from raster...")
for row in range(height):
    for col in range(width):
        if numpy_array[row, col] != 0:
            print(f"  ✓ Center found: weight={numpy_array[row, col]:.2f} at position ({row}, {col})")
            points.append([row, col, numpy_array[row, col]])

print(f"\nTotal centers found: {len(points)}")

if len(points) == 0:
    print("\n❌ ERROR: No centers found in input file")
    print("\n👉 Please check:")
    print("   1. Your shapefile has a 'z' attribute")
    print("   2. The 'z' attribute contains values > 0")
    print("   3. The attribute is named exactly 'z' (lowercase)")
    raise Exception("No centers found")

# Normalize weights to [0..1]
print("\n[3/5] Normalizing center weights to [0..1]...")
points = normalize_weights(points)
for i, p in enumerate(points):
    print(f"  Center {i}: normalized weight={p[2]:.4f}")

# Calculate XTENT influence for each cell
print(f"\n[4/5] Computing XTENT territorial allocation (I = C^{A_COEFFICIENT} - {K_COEFFICIENT}*d)...")
territoryGrid = np.zeros(shape=(height, width), dtype=np.int16)
influenceGrid = np.full(shape=(height, width), fill_value=-np.inf, dtype=np.float32)

total_cells = height * width
processed = 0

for row in range(height):
    for col in range(width):
        max_influence = -np.inf
        dominant_center = -1
        
        # Calculate influence from each center
        for i, (cx, cy, weight) in enumerate(points):
            influence = xtent_influence(row, col, cx, cy, weight, A_COEFFICIENT, K_COEFFICIENT)
            
            if influence > max_influence:
                max_influence = influence
                dominant_center = i
        
        # Apply strict mode constraint if enabled
        if STRICT_MODE and max_influence < 0:
            territoryGrid[row, col] = -1
            influenceGrid[row, col] = max_influence
        else:
            territoryGrid[row, col] = dominant_center
            influenceGrid[row, col] = max_influence
        
        # Progress indicator
        processed += 1
        if processed % (total_cells // 20) == 0:
            progress = (processed / total_cells) * 100
            print(f"  Progress: {progress:.0f}%")

print("  Progress: 100% - Complete!")

if STRICT_MODE:
    null_cells = np.sum(territoryGrid == -1)
    print(f"\nStrict mode: {null_cells} cells with I < 0 set to NULL")

# Save the territory grid as output raster
print("\n[5/5] Saving XTENT territories raster...")
outFileName = '/tmp/xtent_territories.tif'
driver = gdal.GetDriverByName('GTiff')

output = driver.Create(outFileName, height, width, 1, gdal.GDT_Int16)
output.SetGeoTransform(dataset.GetGeoTransform())
output.SetProjection(dataset.GetProjection())
output.GetRasterBand(1).WriteArray(territoryGrid)

if STRICT_MODE:
    output.GetRasterBand(1).SetNoDataValue(-1)

output.FlushCache()
output = None

# Load the result into QGIS
rasterXtent = QgsRasterLayer(outFileName, 'XTENT Territories')
QgsProject.instance().addMapLayer(rasterXtent)

# Save influence values
print("\nSaving influence strength raster...")
influenceFileName = '/tmp/xtent_influence.tif'
influence_output = driver.Create(influenceFileName, height, width, 1, gdal.GDT_Float32)
influence_output.SetGeoTransform(dataset.GetGeoTransform())
influence_output.SetProjection(dataset.GetProjection())
influence_output.GetRasterBand(1).WriteArray(influenceGrid)
influence_output.FlushCache()
influence_output = None

rasterInfluence = QgsRasterLayer(influenceFileName, 'XTENT Influence Strength')
QgsProject.instance().addMapLayer(rasterInfluence)

# Polygonize the result raster
print("\nConverting territories raster to shapefile...")
shp_output = '/tmp/XTENT_Territories.shp'
os.system(f'gdal_polygonize.py {outFileName} {shp_output} -b 1 -f "ESRI Shapefile" territories')

xtentVector = QgsVectorLayer(shp_output, 'XTENT Territories Vector', 'ogr')

# Apply smoothing if enabled
if SMOOTH_POLYGONS:
    print(f"\nApplying polygon smoothing (iterations={SMOOTHING_ITERATIONS}, offset={SMOOTHING_OFFSET})...")
    
    # Create a memory layer for smoothed polygons
    crs = xtentVector.crs().authid()
    smoothed_layer = QgsVectorLayer(f"Polygon?crs={crs}", "XTENT Territories (Smoothed)", "memory")
    smoothed_provider = smoothed_layer.dataProvider()
    
    # Copy attributes
    smoothed_provider.addAttributes(xtentVector.fields())
    smoothed_layer.updateFields()
    
    # Smooth each feature
    smoothed_count = 0
    for feature in xtentVector.getFeatures():
        geom = feature.geometry()
        smoothed_geom = smooth_geometry(geom, iterations=SMOOTHING_ITERATIONS, offset=SMOOTHING_OFFSET)
        
        new_feature = QgsFeature()
        new_feature.setGeometry(smoothed_geom)
        new_feature.setAttributes(feature.attributes())
        smoothed_provider.addFeature(new_feature)
        smoothed_count += 1
    
    smoothed_layer.updateExtents()
    print(f"✓ Smoothed {smoothed_count} polygons")
    
    # Add smoothed layer to QGIS
    QgsProject.instance().addMapLayer(smoothed_layer)
    
    # Optionally save smoothed layer
    smoothed_shp = '/tmp/XTENT_Territories_Smoothed.shp'
    QgsVectorFileWriter.writeAsVectorFormat(smoothed_layer, smoothed_shp, "UTF-8", smoothed_layer.crs(), "ESRI Shapefile")
    print(f"✓ Saved smoothed shapefile: {smoothed_shp}")
else:
    QgsProject.instance().addMapLayer(xtentVector)

print("\n" + "=" * 60)
print("✅ XTENT model computation complete!")
print("=" * 60)
print(f"Output files created:")
print(f"  - {outFileName} (territory assignments)")
print(f"  - {influenceFileName} (influence strength)")
print(f"  - {shp_output} (vector territories)")
if SMOOTH_POLYGONS:
    print(f"  - /tmp/XTENT_Territories_Smoothed.shp (smoothed polygons)")
print("=" * 60)
print("\n👉 Check your QGIS Layers panel for the new layers!")
