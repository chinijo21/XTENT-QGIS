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
    QDialog, QVBoxLayout, QLabel, QListWidget, QSpinBox, QDialogButtonBox,
    QProgressDialog, QApplication, QWidget, QFormLayout, QLineEdit, QComboBox,
    QPushButton, QGroupBox, QHBoxLayout, QFileDialog, QMessageBox, QProgressBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from pathlib import Path

#-------------------------------------CORE FUNCTIONS------------------------------------------------------------------------
def save_raster(path, data, transform_params, width, height, output_crs, nodata_value, data_type=gdal.GDT_Float32):
    """Missing function added back with validation"""
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

#-------------------------------------CUSTOM DIALOGS-----------------------------------------------------------------------
class InputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("XTENT Analysis - Input Parameters")
        self.setMinimumWidth(500)
        self.create_widgets()
        self.setup_layout()
        self.connect_events()

    def create_widgets(self):
        # Analysis Mode
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Cost-based", "Euclidean (classic)"])
        
        # Layer Selection
        self.layer_combo = QComboBox()
        self.populate_layers()
        
        # Numeric Field
        self.numeric_combo = QComboBox()
        
        # Ruler Field
        self.ruler_combo = QComboBox()
        
        # Beta Value
        self.beta_edit = QLineEdit("1.0")
        
        # Cost-based parameters
        self.cost_path_edit = QLineEdit()
        self.cost_browse_btn = QPushButton("Browse...")
        self.max_cost_edit = QLineEdit("1000")
        self.max_dist_cost_edit = QLineEdit("5000")
        
        # Euclidean parameters
        self.max_dist_euclid_edit = QLineEdit("5000")
        self.output_res_edit = QLineEdit("100")
        self.iterations_edit = QSpinBox()
        self.iterations_edit.setRange(1, 20)
        self.iterations_edit.setValue(5)
        self.offset_edit = QLineEdit("0.4")

    def setup_layout(self):
        layout = QVBoxLayout()
        
        # Main form
        form_layout = QFormLayout()
        form_layout.addRow("Analysis Mode:", self.mode_combo)
        form_layout.addRow("Input Layer:", self.layer_combo)
        form_layout.addRow("Size Field:", self.numeric_combo)
        form_layout.addRow("Ruler Field:", self.ruler_combo)
        form_layout.addRow("Beta Value:", self.beta_edit)
        
        # Cost parameters group
        cost_group = QGroupBox("Cost-based Parameters")
        cost_layout = QFormLayout()
        hbox = QHBoxLayout()
        hbox.addWidget(self.cost_path_edit)
        hbox.addWidget(self.cost_browse_btn)
        cost_layout.addRow("Cost Raster:", hbox)
        cost_layout.addRow("Max Cost Value:", self.max_cost_edit)
        cost_layout.addRow("Max Search Distance (m):", self.max_dist_cost_edit)
        cost_group.setLayout(cost_layout)
        
        # Euclidean parameters group
        euclid_group = QGroupBox("Euclidean Parameters")
        euclid_layout = QFormLayout()
        euclid_layout.addRow("Max Distance (m):", self.max_dist_euclid_edit)
        euclid_layout.addRow("Output Resolution (m):", self.output_res_edit)
        euclid_layout.addRow("Smoothing Iterations:", self.iterations_edit)
        euclid_layout.addRow("Smoothing Offset:", self.offset_edit)
        euclid_group.setLayout(euclid_layout)
        
        layout.addLayout(form_layout)
        layout.addWidget(cost_group)
        layout.addWidget(euclid_group)
        
        # Dialog buttons
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(btn_box)
        
        self.setLayout(layout)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)

    def connect_events(self):
        self.layer_combo.currentIndexChanged.connect(self.populate_fields)
        self.cost_browse_btn.clicked.connect(self.browse_cost_raster)
        self.mode_combo.currentIndexChanged.connect(self.toggle_mode)
        self.toggle_mode(0)

    def toggle_mode(self, index):
        for widget in [self.max_dist_cost_edit, self.cost_path_edit, self.cost_browse_btn, self.max_cost_edit]:
            widget.setVisible(index == 0)
        for widget in [self.max_dist_euclid_edit, self.output_res_edit, self.iterations_edit, self.offset_edit]:
            widget.setVisible(index == 1)

    def populate_layers(self):
        self.layer_combo.clear()
        layers = QgsProject.instance().mapLayers().values()
        vector_layers = [layer.name() for layer in layers if isinstance(layer, QgsVectorLayer)]
        self.layer_combo.addItems(vector_layers)
        if vector_layers:
            self.populate_fields()

    def populate_fields(self):
        layer_name = self.layer_combo.currentText()
        layer = QgsProject.instance().mapLayersByName(layer_name)[0]
        numeric_fields = [f.name() for f in layer.fields() if f.isNumeric()]
        if not numeric_fields:
            QMessageBox.critical(self, "Error", "Selected layer has no numeric fields!")
            self.reject()
            return
        
        self.numeric_combo.clear()
        self.numeric_combo.addItems(numeric_fields)
        
        self.ruler_combo.clear()
        self.ruler_combo.addItems([f.name() for f in layer.fields()])

    def browse_cost_raster(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Cost Raster", "", "GeoTIFF (*.tif *.tiff)")
        if path:
            self.cost_path_edit.setText(path)

    def get_parameters(self):
        #Added input validation
        try:
            return {
                'use_costmodel': self.mode_combo.currentIndex() == 0,
                'input_layer': self.layer_combo.currentText(),
                'size_field': self.numeric_combo.currentText(),
                'ruler_field': self.ruler_combo.currentText(),
                'beta': float(self.beta_edit.text()),
                'cost_path': self.cost_path_edit.text(),
                'max_cost': float(self.max_cost_edit.text()),
                'max_distance': float(self.max_dist_cost_edit.text() if self.mode_combo.currentIndex() == 0 else self.max_dist_euclid_edit.text()),
                'output_res': float(self.output_res_edit.text()),
                'smoothing_iterations': self.iterations_edit.value(),
                'smoothing_offset': float(self.offset_edit.text())
            }
        except ValueError as e:
            QMessageBox.critical(self, "Input Error", f"Invalid numeric value: {str(e)}")
            raise

#-------------------------------------PROGRESS DIALOG--------------------------------------------------------------------
class ProgressDialog(QDialog):
    progress_updated = pyqtSignal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Processing Progress")
        self.setMinimumWidth(400)
        layout = QVBoxLayout()
        self.progress_bar = QProgressBar()
        self.status_label = QLabel("Initializing...")
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)
        self.setLayout(layout)

    def update_progress(self, value):
        self.progress_bar.setValue(value)
        QApplication.processEvents()

    def update_status(self, message):
        self.status_label.setText(message)
        QApplication.processEvents()

#-------------------------------------PROCESSING THREAD--------------------------------------------------------------------
class XtentProcessor(QThread):
    progress_updated = pyqtSignal(int)
    finished = pyqtSignal(bool, str)

    def __init__(self, params, output_dir):
        super().__init__()
        self.params = params
        self.output_dir = output_dir
        self.nodata_value = -9999
        self.cancel_requested = False

    def run(self):
        try:
            # Validate QGIS input layer and its CRS
            input_layer = QgsProject.instance().mapLayersByName(self.params['input_layer'])[0]
            if not input_layer.crs().isValid():
                raise QgsProcessingException("Input layer has invalid CRS!")
            
            # Unique ID check for size field
            size_values = [f[self.params['size_field']] for f in input_layer.getFeatures()]
            if len(size_values) != len(set(size_values)):
                QMessageBox.warning(None, "Data Warning", "Size field contains duplicate values - ruler hierarchy may be affected")
            
            features = {f.id(): f for f in input_layer.getFeatures()}

            # Build ruler hierarchy map
            ruler_map = {}
            name_to_fid = {f[self.params['size_field']]: f.id() for f in input_layer.getFeatures()}
            for fid, feature in features.items():
                ruler_name = feature[self.params['ruler_field']]
                if ruler_name and ruler_name in name_to_fid:
                    ruler_map[fid] = name_to_fid[ruler_name]

            # Cost/Euclidean setup
            if self.params['use_costmodel']:
                cost_raster = gdal.Open(self.params['cost_path'])
                cost_transform = cost_raster.GetGeoTransform()
                output_crs = QgsCoordinateReferenceSystem(cost_raster.GetProjection())
                x_min, y_max = cost_transform[0], cost_transform[3]
                width, height = cost_raster.RasterXSize, cost_raster.RasterYSize
            else:
                output_crs = input_layer.crs()
                extent = input_layer.extent().buffered(self.params['max_distance'])
                x_min, y_max = extent.xMinimum(), extent.yMaximum()
                output_res = self.params['output_res']
                width = max(math.ceil(extent.width() / output_res), 1)
                height = max(math.ceil(extent.height() / output_res), 1)

            # Initialize arrays
            dominant_data = np.full((height, width), self.nodata_value, dtype=np.float32)
            influence_data = np.full((height, width), self.nodata_value, dtype=np.float32)
            comp_id_data = np.full((height, width), self.nodata_value, dtype=np.float32)
            comp_strength_data = np.full((height, width), 0.0, dtype=np.float32)

            # Spatial index
            spatial_index = QgsSpatialIndex()
            for fid in features:
                spatial_index.insertFeature(features[fid])

            # ---------- MAIN PROCESSING LOOP ----------
            total_rows = max(height, 1)  # Prevent division by zero
            for row in range(height):
                if self.cancel_requested:
                    break
                
                # Update progress (0-100%)
                self.progress_updated.emit(int((row / total_rows) * 100))
                
                for col in range(width):
                    # ---------- CELL PROCESSING ----------
                    # Cell center calculation
                    if self.params['use_costmodel']:
                        x_center = x_min + (col + 0.5) * cost_transform[1]
                        y_center = y_max - (row + 0.5) * abs(cost_transform[5])
                    else:
                        x_center = x_min + (col + 0.5) * self.params['output_res']
                        y_center = y_max - (row + 0.5) * self.params['output_res']

                    # Search candidates
                    search_rect = QgsRectangle(
                        x_center - self.params['max_distance'],
                        y_center - self.params['max_distance'],
                        x_center + self.params['max_distance'],
                        y_center + self.params['max_distance']
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
                        if self.params['use_costmodel']:
                            try:
                                cost_array = cost_raster.GetRasterBand(1).ReadAsArray(col, row, 1, 1)
                                cost = cost_array[0, 0] if cost_array is not None else self.nodata_value
                            except:
                                cost = self.nodata_value
                            if cost <= 0 or cost > self.params['max_cost']:
                                continue
                            influence = feature[self.params['size_field']] / (cost ** self.params['beta'])
                        else:
                            dx = point.x() - x_center
                            dy = point.y() - y_center
                            distance = math.hypot(dx, dy)
                            if distance > self.params['max_distance'] or distance < 1e-9:
                                continue
                            influence = feature[self.params['size_field']] / (distance ** self.params['beta'])
                        
                        # Aggregate influences
                        if current_fid in influence_dict:
                            influence_dict[current_fid] += influence
                        else:
                            influence_dict[current_fid] = influence

                    # Sort and store results
                    sorted_influences = sorted(influence_dict.items(), key=lambda x: x[1], reverse=True)
                    if not sorted_influences:
                        continue

                    dominant_fid, top_influence = sorted_influences[0]
                    dominant_data[row, col] = dominant_fid
                    influence_data[row, col] = top_influence

                    if len(sorted_influences) > 1:
                        comp_fid, comp_influence = sorted_influences[1]
                        comp_strength = comp_influence / top_influence
                        comp_id_data[row, col] = comp_fid
                        comp_strength_data[row, col] = comp_strength

            # ---------- POST-PROCESSING ----------
            if not self.cancel_requested:
                # Save rasters
                gt = (x_min, 
                      cost_transform[1] if self.params['use_costmodel'] else self.params['output_res'], 
                      0, 
                      y_max, 
                      0, 
                      -abs(cost_transform[5]) if self.params['use_costmodel'] else -self.params['output_res'])

                output_paths = {
                    'dominant': f"{self.output_dir}/dominant.tif",
                    'influence': f"{self.output_dir}/influence.tif",
                    'comp_id': f"{self.output_dir}/competitor_ids.tif",
                    'comp_strength': f"{self.output_dir}/competition_strength.tif"
                }

                save_raster(output_paths['dominant'], dominant_data, gt, width, height, output_crs, self.nodata_value)
                save_raster(output_paths['influence'], influence_data, gt, width, height, output_crs, self.nodata_value)
                save_raster(output_paths['comp_id'], comp_id_data, gt, width, height, output_crs, self.nodata_value)
                save_raster(output_paths['comp_strength'], comp_strength_data, gt, width, height, output_crs, self.nodata_value)

                # Additional processing with validation
                self.post_process(output_paths['dominant'])
                self.load_results()

            self.finished.emit(not self.cancel_requested, self.output_dir)

        except Exception as e:
            self.finished.emit(False, str(e))

    def load_results(self):
        """Added result loading functionality"""
        try:
            iface = qgis.utils.iface
            outputs = [
                ('dominant.tif', 'Dominant_IDs'),
                ('influence.tif', 'Influence_Values'),
                ('competitor_ids.tif', 'Competitor_IDs'),
                ('competition_strength.tif', 'Competition_Strength'),
                ('smoothed.gpkg', 'Smoothed_Influence')
            ]
            
            for fname, layer_name in outputs:
                path = f"{self.output_dir}/{fname}"
                if Path(path).exists():
                    if fname.endswith('.tif'):
                        iface.addRasterLayer(path, layer_name)
                    else:
                        iface.addVectorLayer(path, layer_name, 'ogr')
                        
        except Exception as e:
            QMessageBox.warning(None, "Layer Loading", f"Could not load layers: {str(e)}")

    def post_process(self, dominant_path):
        """Added processing validation"""
        if not Path(dominant_path).exists():
            return

        sieved_path = f"{self.output_dir}/dominant_sieved.tif"
        processing.run("gdal:sieve", {
            'INPUT': dominant_path,
            'THRESHOLD': 10,
            'OUTPUT': sieved_path
        })
        
        if not Path(sieved_path).exists():
            raise QgsProcessingException("Sieving failed!")
        
        # Continue with polygonization and smoothing
        output_polygons = f"{self.output_dir}/influence_areas.gpkg"
        processing.run("gdal:polygonize", {
            'INPUT': sieved_path,
            'BAND': 1,
            'FIELD': 'dominant_id',
            'OUTPUT': output_polygons
        })

        processing.run("native:smoothgeometry", {
            'INPUT': output_polygons,
            'ITERATIONS': self.params.get('smoothing_iterations', 5),
            'OFFSET': self.params.get('smoothing_offset', 0.4),
            'OUTPUT': f"{self.output_dir}/smoothed.gpkg"
        })

    def cancel(self):
        self.cancel_requested = True

#-------------------------------------MAIN EXECUTION-----------------------------------------------------------------------
def main():
    try:
        # Added QGIS initialization check
        if not QgsProject.instance():
            raise QgsProcessingException("QGIS project not initialized!")

        # Input collection
        input_dialog = InputDialog()
        if input_dialog.exec_() != QDialog.Accepted:
            return
        
        params = input_dialog.get_parameters()
        # Validate cost raster if needed
        if params['use_costmodel'] and not Path(params['cost_path']).exists():
            raise QgsProcessingException("Cost raster path is invalid!")
        
        output_dir = QFileDialog.getExistingDirectory(None, "Select Output Directory")
        if not output_dir:
            return

        # Progress monitoring
        progress_dialog = ProgressDialog()
        processor = XtentProcessor(params, output_dir)
        
        # Connect signals
        processor.progress_updated.connect(progress_dialog.update_progress)
        processor.finished.connect(lambda success, msg: (
            progress_dialog.close(),
            QMessageBox.information(None, "Success", "Analysis completed!") if success else
            QMessageBox.critical(None, "Error", f"Failed: {msg}")
        ))
        
        # Start processing
        progress_dialog.show()
        processor.start()
        
    except Exception as e:
        QMessageBox.critical(None, "Error", f"Operation failed: {str(e)}")

if __name__ == "__main__":
    main()
