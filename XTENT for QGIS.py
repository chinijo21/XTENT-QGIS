import numpy as np
import math
from qgis.core import (
    QgsProject, QgsPointXY, QgsSpatialIndex, QgsRectangle,
    QgsCoordinateReferenceSystem, QgsProcessingException, QgsVectorLayer
)
from osgeo import gdal, osr
from qgis import processing
import qgis.utils
from PyQt5.QtWidgets import (
    QInputDialog, QMessageBox, QFileDialog, QDialog, QVBoxLayout,
    QLabel, QListWidget, QSpinBox, QDialogButtonBox
)
from pathlib import Path

#-------------------------------------CUSTOM DIALOGS-----------------------------------------------------------------------
class LayerSelectionDialog(QDialog):
    def __init__(self, title, items, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        
        layout = QVBoxLayout()
        self.list_widget = QListWidget()
        self.list_widget.addItems([f"{i+1}. {item}" for i, item in enumerate(items)])
        self.list_widget.setCurrentRow(0)
        
        self.spin_box = QSpinBox()
        self.spin_box.setMinimum(1)
        self.spin_box.setMaximum(len(items))
        self.spin_box.setValue(1)
        
        layout.addWidget(QLabel("Available layers:"))
        layout.addWidget(self.list_widget)
        layout.addWidget(QLabel("Enter selection number:"))
        layout.addWidget(self.spin_box)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.setLayout(layout)
        self.list_widget.itemDoubleClicked.connect(self.on_double_click)

    def get_selection(self):
        return self.spin_box.value() - 1

    def on_double_click(self, item):
        selected_row = self.list_widget.row(item)
        self.spin_box.setValue(selected_row + 1)
        self.accept()

class NumericFieldDialog(QDialog):
    def __init__(self, title, fields, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        
        layout = QVBoxLayout()
        self.list_widget = QListWidget()
        self.list_widget.addItems([f"{i+1}. {field}" for i, field in enumerate(fields)])
        self.list_widget.setCurrentRow(0)
        
        self.spin_box = QSpinBox()
        self.spin_box.setMinimum(1)
        self.spin_box.setMaximum(len(fields))
        self.spin_box.setValue(1)
        
        layout.addWidget(QLabel("Available numeric fields:"))
        layout.addWidget(self.list_widget)
        layout.addWidget(QLabel("Enter selection number:"))
        layout.addWidget(self.spin_box)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.setLayout(layout)
        self.list_widget.itemDoubleClicked.connect(self.on_double_click)  # Critical connection

    def get_selection(self):
        return self.spin_box.value() - 1

    def on_double_click(self, item):  # Verify exact spelling
        selected_row = self.list_widget.row(item)
        self.spin_box.setValue(selected_row + 1)
        self.accept()

class RulerFieldDialog(QDialog):
    def __init__(self, title, fields, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        
        layout = QVBoxLayout()
        self.list_widget = QListWidget()
        self.list_widget.addItems([f"{i+1}. {field}" for i, field in enumerate(fields)])
        self.list_widget.setCurrentRow(0)
        
        self.spin_box = QSpinBox()
        self.spin_box.setMinimum(1)
        self.spin_box.setMaximum(len(fields))
        self.spin_box.setValue(1)
        
        layout.addWidget(QLabel("Available relationship fields:"))
        layout.addWidget(self.list_widget)
        layout.addWidget(QLabel("Enter selection number:"))
        layout.addWidget(self.spin_box)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.setLayout(layout)
        self.list_widget.itemDoubleClicked.connect(self.on_double_click)

    def get_selection(self):
        return self.spin_box.value() - 1

    def on_double_click(self, item):
        selected_row = self.list_widget.row(item)
        self.spin_box.setValue(selected_row + 1)
        self.accept()

#-------------------------------------CORE FUNCTIONS------------------------------------------------------------------------
def save_raster(path, data, transform_params, width, height, output_crs, nodata_value, data_type=gdal.GDT_Float32):
    if not path.strip() or width <= 0 or height <= 0:
        raise QgsProcessingException("Invalid output path or raster dimensions!")
    
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(path, width, height, 1, data_type)
    if ds is None:
        raise QgsProcessingException(f"Failed to create raster at {path}!")
    
    ds.SetGeoTransform(transform_params)
    ds.SetProjection(output_crs.toWkt())
    band = ds.GetRasterBand(1)
    band.WriteArray(data)
    band.SetNoDataValue(nodata_value)
    ds = None

def get_layers():
    layers = QgsProject.instance().mapLayers().values()
    vector_layers = [layer.name() for layer in layers if isinstance(layer, QgsVectorLayer)]
    if not vector_layers:
        raise QgsProcessingException("No vector layers found!")
    
    dlg = LayerSelectionDialog("Select Input Layer", vector_layers)
    if not dlg.exec_():
        raise QgsProcessingException("Operation canceled.")
    return vector_layers[dlg.get_selection()]

def get_numlayer(layer_name):
    layer = QgsProject.instance().mapLayersByName(layer_name)[0]
    fields = [field.name() for field in layer.fields() if field.isNumeric()]
    if not fields:
        raise QgsProcessingException("No numeric fields found!")
    
    dlg = NumericFieldDialog("Select Size Field", fields)
    if not dlg.exec_():
        raise QgsProcessingException("Operation canceled.")
    return fields[dlg.get_selection()]

def get_ruler_field(layer_name):
    layer = QgsProject.instance().mapLayersByName(layer_name)[0]
    fields = [field.name() for field in layer.fields()]
    if not fields:
        raise QgsProcessingException("No fields found!")
    
    dlg = RulerFieldDialog("Select Ruler Relationship Field", fields)
    if not dlg.exec_():
        raise QgsProcessingException("Operation canceled.")
    return fields[dlg.get_selection()]

def get_input():
    inputs = {}
    
    # Analysis mode
    mode, ok = QInputDialog.getText(None, "Analysis Mode", 
        "Choose analysis mode:\n1. Cost-based\n2. Euclidean (classic)\nEnter 1 or 2:")
    if not ok:
        raise QgsProcessingException("Operation canceled.")
    inputs['use_costmodel'] = (mode.strip() == '1')

    # Layer and field selection
    inputs['input_layer'] = get_layers()
    inputs['size_field'] = get_numlayer(inputs['input_layer'])
    inputs['ruler_field'] = get_ruler_field(inputs['input_layer'])

    # Beta value
    beta, ok = QInputDialog.getDouble(None, "Beta Value", 
        "Enter beta value (>0):", decimals=2, min=0.01)
    if not ok:
        raise QgsProcessingException("Operation canceled.")
    inputs['beta'] = beta

    # Mode-specific parameters
    if inputs['use_costmodel']:
        cost_path, ok = QInputDialog.getText(None, "Cost Raster Path", "Enter full path to cost raster:")
        if not ok:
            raise QgsProcessingException("Operation canceled.")
        inputs['cost_path'] = cost_path.strip()

        max_cost, ok = QInputDialog.getDouble(None, "Max Cost", 
            "Enter maximum cost value:", decimals=2)
        inputs['max_cost'] = max_cost if ok else 0

        max_distance, ok = QInputDialog.getDouble(None, "Max Search Distance", 
            "Enter maximum search distance (meters):", decimals=2)
        inputs['max_distance'] = max_distance if ok else 0
    else:
        max_distance, ok = QInputDialog.getDouble(None, "Max Distance", 
            "Enter maximum distance (meters):", decimals=2)
        inputs['max_distance'] = max_distance if ok else 0

        output_res, ok = QInputDialog.getDouble(None, "Output Resolution", 
            "Enter resolution (meters):", decimals=2)
        inputs['output_res'] = output_res if ok else 0

        iterations, ok = QInputDialog.getInt(None, "Smoothing Iterations", 
            "Enter iterations (5-10 recommended):", 5, 1, 20)
        inputs['smoothing_iterations'] = iterations if ok else 5

        offset, ok = QInputDialog.getDouble(None, "Smoothing Offset", 
            "Enter offset (0.3-0.5 recommended):", decimals=2, value=0.4)
        inputs['smoothing_offset'] = offset if ok else 0.4

    return inputs

def main():
    try:
        QMessageBox.information(None, "XTENT Analysis", "Starting XTENT Analysis with Ruler Relationships")
        params = get_input()
        
        # Output directory
        output_dir = QFileDialog.getExistingDirectory(None, "Select Output Directory")
        if not output_dir:
            raise QgsProcessingException("No output directory selected!")
        
        use_costmodel = params['use_costmodel']
        input_layer_name = params['input_layer']
        size_field = params['size_field']
        ruler_field = params['ruler_field']
        beta = params['beta']
        nodata_value = -9999

        # Load input layer
        input_layer = QgsProject.instance().mapLayersByName(input_layer_name)[0]
        features = {f.id(): f for f in input_layer.getFeatures()}

        # Build ruler hierarchy map
        ruler_map = {}
        name_to_fid = {f[size_field]: f.id() for f in input_layer.getFeatures()}
        for fid, feature in features.items():
            ruler_name = feature[ruler_field]
            if ruler_name and ruler_name in name_to_fid:
                ruler_map[fid] = name_to_fid[ruler_name]

        # Cost/Euclidean setup
        if use_costmodel:
            cost_raster = gdal.Open(params['cost_path'])
            cost_transform = cost_raster.GetGeoTransform()
            output_crs = QgsCoordinateReferenceSystem(cost_raster.GetProjection())
            x_min, y_max = cost_transform[0], cost_transform[3]
            width, height = cost_raster.RasterXSize, cost_raster.RasterYSize
        else:
            output_crs = input_layer.crs()
            extent = input_layer.extent().buffered(params['max_distance'])
            x_min, y_max = extent.xMinimum(), extent.yMaximum()
            output_res = params['output_res']
            width = max(math.ceil(extent.width() / output_res), 1)
            height = max(math.ceil(extent.height() / output_res), 1)

        # Initialize arrays
        dominant_data = np.full((height, width), nodata_value, dtype=np.float32)
        influence_data = np.full((height, width), nodata_value, dtype=np.float32)
        comp_id_data = np.full((height, width), nodata_value, dtype=np.float32)
        comp_strength_data = np.full((height, width), 0.0, dtype=np.float32)

        # Spatial index
        spatial_index = QgsSpatialIndex()
        for fid in features:
            spatial_index.insertFeature(features[fid])

        # Raster processing
        for row in range(height):
            for col in range(width):
                # Cell center calculation
                if use_costmodel:
                    x_center = x_min + (col + 0.5) * cost_transform[1]
                    y_center = y_max - (row + 0.5) * abs(cost_transform[5])
                else:
                    x_center = x_min + (col + 0.5) * params['output_res']
                    y_center = y_max - (row + 0.5) * params['output_res']

                # Search candidates
                search_rect = QgsRectangle(
                    x_center - params['max_distance'],
                    y_center - params['max_distance'],
                    x_center + params['max_distance'],
                    y_center + params['max_distance']
                )
                candidate_fids = spatial_index.intersects(search_rect)
                if not candidate_fids:
                    continue

                # Calculate influences with ruler hierarchy
                influence_dict = {}
                for fid in candidate_fids:
                    feature = features[fid]
                    point = feature.geometry().asPoint()
                    
                    # Resolve ruler hierarchy
                    current_fid = fid
                    while current_fid in ruler_map:
                        current_fid = ruler_map[current_fid]
                    
                    # Calculate influence
                    if use_costmodel:
                        try:
                            cost_array = cost_raster.GetRasterBand(1).ReadAsArray(col, row, 1, 1)
                            cost = cost_array[0, 0] if cost_array is not None else nodata_value
                        except:
                            cost = nodata_value
                        if cost <= 0 or cost > params['max_cost']:
                            continue
                        influence = feature[size_field] / (cost ** beta)
                    else:
                        dx = point.x() - x_center
                        dy = point.y() - y_center
                        distance = math.hypot(dx, dy)
                        if distance > params['max_distance'] or distance < 1e-9:
                            continue
                        influence = feature[size_field] / (distance ** beta)
                    
                    # Aggregate influences by final ruler
                    if current_fid in influence_dict:
                        influence_dict[current_fid] += influence
                    else:
                        influence_dict[current_fid] = influence

                # Sort aggregated influences
                sorted_influences = sorted(influence_dict.items(), key=lambda x: x[1], reverse=True)
                if len(sorted_influences) == 0:
                    continue

                # Set dominant and competitor
                dominant_fid, top_influence = sorted_influences[0]
                dominant_data[row, col] = dominant_fid
                influence_data[row, col] = top_influence

                if len(sorted_influences) > 1:
                    comp_fid, comp_influence = sorted_influences[1]
                    comp_strength = comp_influence / top_influence
                    comp_id_data[row, col] = comp_fid
                    comp_strength_data[row, col] = comp_strength

        # Output paths
        output_dominant_path = f"{output_dir}/dominant.tif"
        output_influence_path = f"{output_dir}/influence.tif"
        output_comp_id_path = f"{output_dir}/competitor_ids.tif"
        output_comp_strength_path = f"{output_dir}/competition_strength.tif"
        output_polygons_path = f"{output_dir}/influence_areas.gpkg"
        smoothy_polygons_path = f"{output_dir}/smoothed.gpkg"

        # Geotransform
        gt = (x_min, 
              cost_transform[1] if use_costmodel else params['output_res'], 
              0, 
              y_max, 
              0, 
              -abs(cost_transform[5]) if use_costmodel else -params['output_res'])

        # Save rasters
        save_raster(output_dominant_path, dominant_data, gt, width, height, output_crs, nodata_value, gdal.GDT_Float32)
        save_raster(output_influence_path, influence_data, gt, width, height, output_crs, nodata_value, gdal.GDT_Float32)
        save_raster(output_comp_id_path, comp_id_data, gt, width, height, output_crs, nodata_value, gdal.GDT_Float32)
        save_raster(output_comp_strength_path, comp_strength_data, gt, width, height, output_crs, nodata_value, gdal.GDT_Float32)

        # Post-processing
        processing.run("gdal:sieve", {
            'INPUT': output_dominant_path,
            'THRESHOLD': 10,
            'OUTPUT': f"{output_dir}/dominant_sieved.tif"
        })
        
        processing.run("gdal:polygonize", {
            'INPUT': f"{output_dir}/dominant_sieved.tif",
            'BAND': 1,
            'FIELD': 'dominant_id',
            'OUTPUT': output_polygons_path
        })

        processing.run("native:smoothgeometry", {
            'INPUT': output_polygons_path,
            'ITERATIONS': params.get('smoothing_iterations', 5),
            'OFFSET': params.get('smoothing_offset', 0.4),
            'OUTPUT': smoothy_polygons_path
        })

        # Load results
        try:
            iface = qgis.utils.iface
            iface.addRasterLayer(output_dominant_path, 'Dominant_IDs')
            iface.addRasterLayer(output_influence_path, 'Influence_Values')
            iface.addRasterLayer(output_comp_id_path, 'Competitor_IDs')
            iface.addRasterLayer(output_comp_strength_path, 'Competition_Strength')
            iface.addVectorLayer(smoothy_polygons_path, 'Smoothed_Influence', 'ogr')
        except Exception as e:
            QMessageBox.warning(None, "Layer Loading", f"Could not load layers: {str(e)}")

        QMessageBox.information(None, "Success", 
            "Analysis complete!\nDominant, Competitor, and Competition Strength rasters generated.")

    except Exception as e:
        QMessageBox.critical(None, "Error", f"Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()