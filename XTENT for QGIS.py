import numpy as np
import math
from qgis.core import (
    QgsProject, QgsPointXY, QgsSpatialIndex, QgsRectangle,
    QgsCoordinateReferenceSystem, QgsProcessingException, QgsVectorLayer
)
from osgeo import gdal, osr
from qgis import processing
import qgis.utils
from PyQt5.QtWidgets import QInputDialog, QMessageBox, QFileDialog
from pathlib import Path

#-------------------------------------FUNCTIONS----------------------------------------------------------------------------

def save_raster(path, data, transform_params, width, height, output_crs, nodata_value):
    # Validate parameters
    if not path.strip() or width <= 0 or height <= 0:
        raise QgsProcessingException("Invalid output path or raster dimensions!")
    
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(path, width, height, 1, gdal.GDT_Float32)
    if ds is None:
        raise QgsProcessingException(f"Failed to create raster at {path}!")
    
    ds.SetGeoTransform(transform_params)
    ds.SetProjection(output_crs.toWkt())
    ds.GetRasterBand(1).WriteArray(data)
    ds.GetRasterBand(1).SetNoDataValue(nodata_value)
    ds = None

def get_layers():
    layers = QgsProject.instance().mapLayers().values()
    vector_layers = [layer.name() for layer in layers if isinstance(layer, QgsVectorLayer)]
    msg = "Available layers:\n" + "\n".join(f"{i+1}. {name}" for i, name in enumerate(vector_layers))
    QMessageBox.information(None, "Layers", msg)
    return vector_layers

def get_numlayer(name_layer):
    layer_list = QgsProject.instance().mapLayersByName(name_layer)
    if not layer_list:
        raise QgsProcessingException(f"Layer '{name_layer}' not found!")
    layer = layer_list[0]
    fields = [field.name() for field in layer.fields() if field.isNumeric()]
    msg = "Available numeric fields:\n" + "\n".join(f"{i+1}. {f}" for i, f in enumerate(fields))
    QMessageBox.information(None, "Numeric Fields", msg)
    return fields

def get_input():
    inputs = {}
    
    # Choose analysis mode
    mode, ok = QInputDialog.getText(None, "Analysis Mode", 
        "Choose analysis mode:\n1. Cost-based\n2. Euclidean (classic)\nEnter 1 or 2:")
    if not ok:
        raise QgsProcessingException("Operation canceled.")
    inputs['use_costmodel'] = (mode.strip() == '1')

    # Layer selection
    vector_layers = get_layers()
    layer_choice, ok = QInputDialog.getInt(None, "Input Layer", 
        "Enter the number for the input point layer:", 1, 1, len(vector_layers))
    if not ok:
        raise QgsProcessingException("Operation canceled.")
    inputs['input_layer'] = vector_layers[layer_choice - 1]

    # Size field selection
    num_fields = get_numlayer(inputs['input_layer'])
    field_choice, ok = QInputDialog.getInt(None, "Size Field", 
        "Enter the number for the size field:", 1, 1, len(num_fields))
    if not ok:
        raise QgsProcessingException("Operation canceled.")
    inputs['size_field'] = num_fields[field_choice - 1]

    # Beta value
    beta, ok = QInputDialog.getDouble(None, "Beta Value", 
        "Enter beta value (>0):", decimals=2, min=0.01)
    if not ok:
        raise QgsProcessingException("Operation canceled.")
    inputs['beta'] = beta

    # Mode-specific inputs
    if inputs['use_costmodel']:
        cost_path, ok = QInputDialog.getText(None, "Cost Raster Path", "Enter full path to cost raster:")
        if not ok:
            raise QgsProcessingException("Operation canceled.")
        inputs['cost_path'] = cost_path.strip()

        max_cost, ok = QInputDialog.getDouble(None, "Max Cost", 
            "Enter maximum cost value:", decimals=2)
        if not ok:
            raise QgsProcessingException("Operation canceled.")
        inputs['max_cost'] = max_cost

        max_distance, ok = QInputDialog.getDouble(None, "Max Search Distance", 
            "Enter maximum search distance in meters:", decimals=2)
        if not ok:
            raise QgsProcessingException("Operation canceled.")
        inputs['max_distance'] = max_distance
    else:
        max_distance, ok = QInputDialog.getDouble(None, "Max Distance", 
            "Enter maximum distance in meters:", decimals=2)
        if not ok:
            raise QgsProcessingException("Operation canceled.")
        inputs['max_distance'] = max_distance

        output_res, ok = QInputDialog.getDouble(None, "Output Resolution", 
            "Enter output resolution in meters:", decimals=2)
        if not ok:
            raise QgsProcessingException("Operation canceled.")
        inputs['output_res'] = output_res

        iterations, ok = QInputDialog.getInt(None, "Smoothing Iterations", 
            "Enter smoothing iterations:", 3, 1, 100)
        if not ok:
            raise QgsProcessingException("Operation canceled.")
        inputs['smoothing_iterations'] = iterations

        offset, ok = QInputDialog.getDouble(None, "Smoothing Offset", 
            "Enter smoothing offset:", decimals=2, value=0.2)
        if not ok:
            raise QgsProcessingException("Operation canceled.")
        inputs['smoothing_offset'] = offset

    return inputs

def main():
    try:
        QMessageBox.information(None, "XTENT Analysis", "Starting XTENT Analysis.")
        params = get_input()
        
        # Get output directory
        output_dir = QFileDialog.getExistingDirectory(None, "Select Output Directory")
        if not output_dir:
            raise QgsProcessingException("No output directory selected!")
        
        use_costmodel = params['use_costmodel']
        input_layer_name = params['input_layer']
        size_field = params['size_field']
        beta = params['beta']
        nodata_value = -9999

        # Load input layer
        input_layer = QgsProject.instance().mapLayersByName(input_layer_name)
        if not input_layer:
            raise QgsProcessingException(f"Layer '{input_layer_name}' not found!")
        input_layer = input_layer[0]

        # Cost model setup
        if use_costmodel:
            cost_raster = gdal.Open(params['cost_path'])
            if not cost_raster:
                raise QgsProcessingException(f"Cost raster not found at {params['cost_path']}!")
            cost_transform = cost_raster.GetGeoTransform()
            if not cost_transform:
                raise QgsProcessingException("Invalid cost raster geotransform!")
            output_crs = QgsCoordinateReferenceSystem(cost_raster.GetProjection())
            x_min = cost_transform[0]
            y_max = cost_transform[3]
            width = cost_raster.RasterXSize
            height = cost_raster.RasterYSize
        else:
            output_crs = input_layer.crs()
            if not output_crs.isValid() or output_crs.isGeographic():
                raise QgsProcessingException("Classic model requires a valid projected CRS!")
            max_distance = params['max_distance']
            output_res = params['output_res']
            extent = input_layer.extent().buffered(max_distance)
            x_min = extent.xMinimum()
            y_max = extent.yMaximum()
            width = max(math.ceil(extent.width() / output_res), 1)
            height = max(math.ceil(extent.height() / output_res), 1)

        # Initialize arrays
        dominant_data = np.full((height, width), nodata_value, dtype=np.float32)
        influence_data = np.full((height, width), nodata_value, dtype=np.float32)

        # Spatial index
        spatial_index = QgsSpatialIndex()
        features = {}
        for feature in input_layer.getFeatures():
            if feature.hasGeometry():
                features[feature.id()] = feature
                spatial_index.insertFeature(feature)

        # Raster processing
        for row in range(height):
            for col in range(width):
                if use_costmodel:
                    x_center = x_min + (col + 0.5) * cost_transform[1]
                    y_center = y_max - (row + 0.5) * abs(cost_transform[5])
                else:
                    x_center = x_min + (col + 0.5) * output_res
                    y_center = y_max - (row + 0.5) * output_res

                search_rect = QgsRectangle(
                    x_center - params['max_distance'],
                    y_center - params['max_distance'],
                    x_center + params['max_distance'],
                    y_center + params['max_distance']
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
                            cost_array = cost_raster.GetRasterBand(1).ReadAsArray(col, row, 1, 1)
                            cost = cost_array[0, 0] if cost_array is not None else nodata_value
                        except Exception:
                            cost = nodata_value
                        
                        if cost == nodata_value or cost <= 0 or cost > params['max_cost']:
                            continue
                        
                        influence = feature[size_field] / (cost ** beta)
                    else:
                        dx = point.x() - x_center
                        dy = point.y() - y_center
                        distance = math.hypot(dx, dy)
                        if distance > params['max_distance'] or distance < 1e-9:
                            continue
                        influence = feature[size_field] / (distance ** beta)

                    if influence > max_influence:
                        max_influence = influence
                        dominant_fid = fid

                if max_influence > 0:
                    dominant_data[row, col] = dominant_fid if dominant_fid is not None else nodata_value
                    influence_data[row, col] = max_influence

        # Define output paths
        output_dominant_path = f"{output_dir}/dominant.tif"
        output_influence_path = f"{output_dir}/influence.tif"
        output_polygons_path = f"{output_dir}/influence_areas.gpkg"
        smoothy_polygons_path = f"{output_dir}/smoothed.gpkg"

        # Set geotransform
        if use_costmodel:
            gt = (x_min, cost_transform[1], 0, y_max, 0, -abs(cost_transform[5]))
        else:
            gt = (x_min, output_res, 0, y_max, 0, -output_res)

        # Save rasters
        save_raster(output_dominant_path, dominant_data, gt, width, height, output_crs, nodata_value)
        save_raster(output_influence_path, influence_data, gt, width, height, output_crs, nodata_value)

        # Polygonize
        processing.run("gdal:polygonize", {
            'INPUT': output_dominant_path,
            'BAND': 1,
            'FIELD': 'dominant_id',
            'OUTPUT': output_polygons_path
        })

        # Smooth
        processing.run("native:smoothgeometry", {
            'INPUT': output_polygons_path,
            'ITERATIONS': params.get('smoothing_iterations', 3),
            'OFFSET': params.get('smoothing_offset', 0.2),
            'OUTPUT': smoothy_polygons_path
        })

        # Load results
        try:
            iface = qgis.utils.iface
            iface.addRasterLayer(output_dominant_path, 'Dominant_IDs')
            iface.addRasterLayer(output_influence_path, 'Influence_Values')
            iface.addVectorLayer(smoothy_polygons_path, 'Smoothed_Influence', 'ogr')
        except Exception as e:
            QMessageBox.warning(None, "Layer Loading", f"Could not load layers: {str(e)}")

        QMessageBox.information(None, "Success", 
            f"XTENT completed using {'cost-based' if use_costmodel else 'classic'} model!")

    except Exception as e:
        QMessageBox.critical(None, "Error", f"XTENT failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()