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

# Hierarchical Ruler System (Optional)
USE_HIERARCHY = False         # Enable political hierarchy system
RULER_FIELD = "ruler"         #Field name containing ruler ID/name (use feature ID, name, or FID)
HIERARCHY_MODE = "metadata"   # "metadata" = visualization only, "inherit" = vassals get ruler boost
VASSAL_BOOST_FACTOR = 0.3     # If inherit mode: vassals gain this % of their ruler's weight


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


def read_hierarchy_from_shapefile(vector_layer, ruler_field):
    """
    Read hierarchical ruler relationships from shapefile attributes.
    
    Returns:
        dict: {feature_id: ruler_id} mapping
        dict: {feature_id: feature_name} for lookups
    """
    ruler_dict = {}
    id_to_name = {}
    
    for feature in vector_layer.getFeatures():
        fid = feature.id()
        attrs = feature.attributes()
        fields = vector_layer.fields()
        
        # Get feature identifier (try name field first, then use FID)
        feature_name = None
        for field_idx, field in enumerate(fields):
            if field.name().lower() in ['name', 'id', 'fid']:
                feature_name = str(attrs[field_idx])
                break
        if not feature_name:
            feature_name = f"Center_{fid}"
        
        id_to_name[fid] = feature_name
        
        # Get ruler field if it exists
        ruler_field_idx = fields.indexOf(ruler_field)
        if ruler_field_idx >= 0:
            ruler_value = attrs[ruler_field_idx]
            if ruler_value and str(ruler_value).strip() and str(ruler_value).lower() not in ['null', 'none', '']:
                ruler_dict[fid] = str(ruler_value).strip()
    
    return ruler_dict, id_to_name


def build_hierarchy_tree(ruler_dict, id_to_name):
    """
    Resolve transitive ruler relationships.
    E.g., if E→F and F→A, then E's ultimate ruler is A.
    
    Returns:
        dict: {feature_id: top_ruler_name}
    """
    # Create name-to-id reverse lookup
    name_to_id = {name: fid for fid, name in id_to_name.items()}
    
    hierarchy = {}
    
    for fid, name in id_to_name.items():
        # Trace up the hierarchy chain
        current_id = fid
        visited = set()
        path = []
        
        while current_id in ruler_dict:
            if current_id in visited:
                # Circular dependency detected
                path_str = " → ".join([id_to_name.get(p, str(p)) for p in path])
                raise Exception(f"❌ Circular hierarchy detected: {path_str} → {id_to_name.get(current_id, str(current_id))}")
            
            visited.add(current_id)
            path.append(current_id)
            
            # Get ruler name and try to find its ID
            ruler_name = ruler_dict[current_id]
            
            # Try to find ruler by name
            if ruler_name in name_to_id:
                current_id = name_to_id[ruler_name]
            elif ruler_name in id_to_name.values():
                # Ruler name matches a feature name, find its ID
                current_id = [k for k, v in id_to_name.items() if v == ruler_name][0]
            else:
                # Ruler not found in dataset - treat as external/top ruler
                hierarchy[fid] = ruler_name
                current_id = None
                break
        
        # current_id is now the top ruler (or None if external)
        if current_id is not None:
            hierarchy[fid] = id_to_name.get(current_id, str(current_id))
        elif fid not in hierarchy:
            # No ruler found, this is an independent center
            hierarchy[fid] = id_to_name[fid]
    
    return hierarchy


def apply_hierarchy_influence(points, point_ids, hierarchy, id_to_name, mode, boost_factor):
    """
    Modify center weights based on hierarchy mode.
    
    points: list of [row, col, weight]
    point_ids: list of feature IDs corresponding to points
    hierarchy: dict of {fid: top_ruler_name}
    mode: "metadata" or "inherit"
    boost_factor: float 0.0-1.0
    
    Returns:
        modified points list
    """
    if mode == "metadata":
        # No modification, hierarchy is just for visualization
        return points
    
    elif mode == "inherit":
        # Vassals inherit partial weight from their rulers
        # First, calculate ruler weights
        ruler_weights = {}
        for i, fid in enumerate(point_ids):
            name = id_to_name[fid]
            weight = points[i][2]
            if name not in ruler_weights:
                ruler_weights[name] = 0
            ruler_weights[name] += weight
        
        # Then, apply boosts to vassals
        modified_points = []
        for i, fid in enumerate(point_ids):
            row, col, weight = points[i]
            name = id_to_name[fid]
            top_ruler = hierarchy.get(fid, name)
            
            if top_ruler != name and top_ruler in ruler_weights:
                # This is a vassal, apply boost
                ruler_weight = ruler_weights[top_ruler]
                boosted_weight = weight + (ruler_weight * boost_factor)
                modified_points.append([row, col, boosted_weight])
                print(f"  {name} (vassal of {top_ruler}): weight boosted {weight:.3f} → {boosted_weight:.3f}")
            else:
                modified_points.append([row, col, weight])
        
        return modified_points
    
    else:
        return points


def create_hierarchy_map(territory_grid, point_ids, hierarchy, id_to_name):
    """
    Create a raster where each pixel shows its ultimate top-level ruler.
    
    Returns:
        numpy array with ruler IDs
    """
    # Create mapping from territory ID (array index) to ruler ID
    territory_to_ruler = {}
    for i, fid in enumerate(point_ids):
        name = id_to_name[fid]
        top_ruler = hierarchy.get(fid, name)
        # Find the ID of the top ruler
        ruler_id = i  # default to self
        for j, other_fid in enumerate(point_ids):
            if id_to_name[other_fid] == top_ruler:
                ruler_id = j
                break
        territory_to_ruler[i] = ruler_id
    
    # Map territory grid to ruler grid
    hierarchy_grid = np.copy(territory_grid)
    for territory_id, ruler_id in territory_to_ruler.items():
        hierarchy_grid[territory_grid == territory_id] = ruler_id
    
    return hierarchy_grid



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
print("\n[3/6] Normalizing center weights to [0..1]...")
points = normalize_weights(points)
for i, p in enumerate(points):
    print(f"  Center {i}: normalized weight={p[2]:.4f}")

# Process hierarchical relationships if enabled
hierarchy_tree = None
id_to_name = None
point_ids = None

if USE_HIERARCHY:
    print(f"\n[4/6] Processing hierarchical relationships (mode: {HIERARCHY_MODE})...")
    
    try:
        # Build list of point IDs corresponding to rasterized points
        point_ids = []
        for feature in pointsVector.getFeatures():
            point_ids.append(feature.id())
        
        # Read hierarchy from shapefile
        ruler_dict, id_to_name = read_hierarchy_from_shapefile(pointsVector, RULER_FIELD)
        
        if ruler_dict:
            print(f"  ✓ Found {len(ruler_dict)} ruler relationships")
            
            # Build complete hierarchy tree
            hierarchy_tree = build_hierarchy_tree(ruler_dict, id_to_name)
            
            # Report hierarchy
            for fid in point_ids[:len(points)]:  # Only for points we found
                name = id_to_name.get(fid, f"Center_{fid}")
                top_ruler = hierarchy_tree.get(fid, name)
                if top_ruler != name:
                    # Find the chain
                    chain = [name]
                    current_fid = fid
                    while current_fid in ruler_dict:
                        ruler_name = ruler_dict[current_fid]
                        chain.append(ruler_name)
                        # Find next in chain
                        found = False
                        for check_fid, check_name in id_to_name.items():
                            if check_name == ruler_name:
                                current_fid = check_fid
                                found = True
                                break
                        if not found:
                            break
                    print(f"  {' → '.join(chain)}")
            
            # Apply hierarchy influence modification
            points = apply_hierarchy_influence(
                points, 
                point_ids[:len(points)], 
                hierarchy_tree, 
                id_to_name, 
                HIERARCHY_MODE, 
                VASSAL_BOOST_FACTOR
            )
        else:
            print(f"  ⚠ Ruler field '{RULER_FIELD}' not found or empty - skipping hierarchy")
            USE_HIERARCHY = False
            
    except Exception as e:
        print(f"  ❌ Error processing hierarchy: {str(e)}")
        print(f"  ⚠ Continuing without hierarchy...")
        USE_HIERARCHY = False


# Calculate XTENT influence for each cell
print(f"\n[5/6] Computing XTENT territorial allocation (I = C^{A_COEFFICIENT} - {K_COEFFICIENT}*d)...")
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
print("\n[6/6] Saving XTENT territories raster...")
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

# Generate hierarchy map if hierarchies were processed
if USE_HIERARCHY and hierarchy_tree and point_ids and id_to_name:
    print("\nCreating political hierarchy map...")
    try:
        hierarchy_grid = create_hierarchy_map(territoryGrid, point_ids[:len(points)], hierarchy_tree, id_to_name)
        
        # Save hierarchy raster
        hierarchy_filename = '/tmp/xtent_hierarchy.tif'
        hierarchy_output = driver.Create(hierarchy_filename, height, width, 1, gdal.GDT_Int16)
        hierarchy_output.SetGeoTransform(dataset.GetGeoTransform())
        hierarchy_output.SetProjection(dataset.GetProjection())
        hierarchy_output.GetRasterBand(1).WriteArray(hierarchy_grid)
        
        if STRICT_MODE:
            hierarchy_output.GetRasterBand(1).SetNoDataValue(-1)
        
        hierarchy_output.FlushCache()
        hierarchy_output = None
        
        rasterHierarchy = QgsRasterLayer(hierarchy_filename, 'XTENT Political Hierarchy')
        QgsProject.instance().addMapLayer(rasterHierarchy)
        print(f"✓ Hierarchy map saved: {hierarchy_filename}")
        
    except Exception as e:
        print(f"⚠ Warning: Could not create hierarchy map: {str(e)}")


# Polygonize the result raster
print("\nConverting territories raster to shapefile...")
shp_output = '/tmp/XTENT_Territories.shp'
os.system(f'gdal_polygonize.py {outFileName} {shp_output} -b 1 -f "ESRI Shapefile" territories')

xtentVector = QgsVectorLayer(shp_output, 'XTENT Territories Vector', 'ogr')

# Add hierarchy attributes if enabled
if USE_HIERARCHY and hierarchy_tree and point_ids and id_to_name:
    print("\nAdding political hierarchy attributes to polygons...")
    try:
        # Add new field for top ruler
        xtentVector.startEditing()
        xtentVector.dataProvider().addAttributes([QgsField("top_ruler", QVariant.String)])
        xtentVector.updateFields()
        
        # Update each feature with its top ruler
        for feature in xtentVector.getFeatures():
            territory_id = feature.attribute('DN')  # DN is the territory ID from gdal_polygonize
            if territory_id is not None and territory_id >= 0 and territory_id < len(point_ids):
                fid = point_ids[territory_id]
                top_ruler = hierarchy_tree.get(fid, id_to_name.get(fid, "Unknown"))
                feature.setAttribute('top_ruler', top_ruler)
                xtentVector.updateFeature(feature)
        
        xtentVector.commitChanges()
        print("✓ Added 'top_ruler' attribute to polygons")
        
    except Exception as e:
        print(f"⚠ Warning: Could not add hierarchy attributes: {str(e)}")
        xtentVector.rollBack()


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
if USE_HIERARCHY and hierarchy_tree:
    print(f"  - /tmp/xtent_hierarchy.tif (political hierarchy map)")
print(f"  - {shp_output} (vector territories)")
if SMOOTH_POLYGONS:
    print(f"  - /tmp/XTENT_Territories_Smoothed.shp (smoothed polygons)")
if USE_HIERARCHY and hierarchy_tree:
    print(f"\n📊 Hierarchy features:")
    print(f"  - 'top_ruler' attribute added to polygon layers")
    print(f"  - Hierarchy map shows ultimate political control")
print("=" * 60)
print("\n👉 Check your QGIS Layers panel for the new layers!")
