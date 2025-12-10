# XTENT and Weighted Voronoi Territorial Analysis for QGIS

[![QGIS](https://img.shields.io/badge/QGIS-3.x-green.svg)](https://qgis.org)
[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Python scripts for archaeological and geographic territorial analysis in QGIS, implementing the **XTENT territorial model** (Renfrew & Level, 1979) and weighted Voronoi diagrams.

## 🚀 Quick Start

**New users**: Start with [`xtent_simple.py`](#xtent-simple-editor-xtent_simplepy---recommended-for-beginners) - just edit the parameters at the top and run in QGIS Python Editor!



## Scripts Overview


### 1. **XTENT Model - Simple Editor** (`xtent_simple.py`) ⭐
User-friendly XTENT implementation designed for QGIS Python Editor with easy-to-configure parameters at the top of the script. Includes advanced features like polygon smoothing and MultiPoint support.

**Key Features:**
- 🎯 **Easy Configuration**: Edit parameters directly at the top of the script - no command-line arguments needed
- 📐 **Polygon Smoothing**: Optional Chaikin smoothing for rounded, organic-looking territory boundaries
- 🔄 **MultiPoint Support**: Automatically handles MultiPoint geometries and extracts individual centers
- 📊 **Dual Output**: Generates both territory assignments and influence strength rasters
- ✅ **Robust Error Handling**: Clear error messages and validation checks
- 🎨 **QGIS Integration**: Automatically loads all outputs into QGIS layers panel

---

## XTENT Model

### What is XTENT?

XTENT is a territorial modeling algorithm based on the formula published by Renfrew and Level (1979):

```
I = C^a - k*d
```

Where:
- **I** = Influence strength at any location
- **C** = Center weight (size/importance), normalized to [0..1]
- **a** = Exponential coefficient for center weight
- **k** = Linear coefficient for distance
- **d** = Euclidean distance from center

Each cell is assigned to the center with the **highest influence** I at that location.

### How XTENT Works

The XTENT formula balances two competing factors:

1. **Center Size (C^a)**: Larger centers have exponentially greater influence
2. **Distance (k*d)**: Distance reduces influence linearly

**Key Characteristics:**
- Large centers can dominate even at greater distances
- The balance between size and distance is controlled by `a` and `k`
- Increasing `a` amplifies the advantage of larger centers
- Increasing `k` makes distance more important, reducing territory sizes

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--a` | float | 0.5 | Exponential coefficient for center weight |
| `--k` | float | 1.0 | Linear coefficient for distance |
| `--strict` | flag | False | Enforce I >= 0 (original XTENT behavior) |

### Input Requirements

- **Format**: ESRI Shapefile with point geometry
- **Required Attribute**: `z` (numeric) containing center weights
- **CRS**: Any projected coordinate system

### Outputs

The script generates three outputs:

1. **`xtent_territories.tif`**: Raster with territory assignments (center IDs)
2. **`xtent_influence.tif`**: Raster showing influence strength values
3. **`XTENT_Territories.shp`**: Vector polygons of territories

---

## XTENT Simple Editor (`xtent_simple.py`) 

### Quick Start Guide

The easiest way to run XTENT analysis in QGIS:

1. Open QGIS and your project
2. Open Python Editor: `Plugins` → `Python Console` → Click the editor icon 📝
3. Open `xtent_simple.py` in the editor
4. **Edit lines 17-27** with your parameters
5. Click Run (▶️) or press `Ctrl+Enter`

### Configuration Parameters

Simply edit these values at the top of the script:

```python
# Input Configuration  
INPUT_SHAPEFILE = "/path/to/your/centers.shp"

# XTENT Parameters
A_COEFFICIENT = 0.8   # Exponential coefficient (default: 0.5-1.0)
K_COEFFICIENT = 2.0   # Distance coefficient (default: 1.0-3.0)
STRICT_MODE = False   # True = enforce I >= 0 constraint

# Smoothing Parameters
SMOOTH_POLYGONS = True         # Enable polygon smoothing
SMOOTHING_ITERATIONS = 3       # 1-5 (higher = smoother)
SMOOTHING_OFFSET = 0.25        # 0.0-0.5 (higher = more rounded)
```

**Parameter Effects:**
- **A_COEFFICIENT** ↑ = Larger centers dominate more
- **K_COEFFICIENT** ↑ = Distance matters more (smaller territories)
- **SMOOTH_POLYGONS** = Creates organic, rounded boundaries instead of pixelated edges
- **STRICT_MODE** = Allows "no man's land" zones where no center has positive influence

### Input Data Format

Your shapefile must have:
- **Geometry**: Point or MultiPoint
- **Required field**: `z` (numeric) with center weights
- **CRS**: Projected coordinate system (meters recommended)

**Example:**
```
FID | Name       | z
----|------------|------
1   | Capital    | 100
2   | Town       | 50
3   | Village    | 20
```

### Output Files

Four layers automatically loaded into QGIS:

1. `xtent_territories.tif` - Territory IDs (raster)
2. `xtent_influence.tif` - Influence values (raster)
3. `XTENT_Territories.shp` - Territory polygons (vector)
4. `XTENT_Territories_Smoothed.shp` - Smoothed polygons (if enabled)

### Example Use Cases

**Archaeological territories:**
```python
A_COEFFICIENT = 0.7
K_COEFFICIENT = 1.5
SMOOTH_POLYGONS = True
```

**Urban influence zones:**
```python
A_COEFFICIENT = 1.2  # Size matters a lot  
K_COEFFICIENT = 0.8  # Wide reach
```

**Local market areas:**
```python
A_COEFFICIENT = 0.5
K_COEFFICIENT = 3.0  # Very local
STRICT_MODE = True   # Gaps between markets
```

### Special Features

✅ **MultiPoint Auto-Conversion** - Automatically extracts points from MultiPoint features  
✅ **Progress Display** - Shows detailed status updates during computation  
✅ **Error Guidance** - Clear, helpful error messages with solutions  
✅ **Polygon Smoothing** - Chaikin algorithm for publication-quality maps  
✅ **Weight Normalization** - Automatically scales weights to [0,1]


## Examples

### Example 1: Voronoi Equivalence
With equal weights (C=1.0 for all), `a=0.5`, and `k=1.0`, XTENT produces a standard Voronoi diagram:

```python
sys.argv = ["equal_weights.shp", "--a", "0.5", "--k", "1.0"]
```

### Example 2: Weighted Territories
With varying weights, larger centers dominate proportionally more area:

```python
sys.argv = ["weighted_centers.shp", "--a", "0.5", "--k", "1.0"]
```

### Example 3: Size-Dominant Model
Increase `a` to give larger centers a stronger advantage:

```python
sys.argv = ["centers.shp", "--a", "1.0", "--k", "1.0"]
```

### Example 4: Distance-Dominant Model
Increase `k` to make distance more important (smaller territories):

```python
sys.argv = ["centers.shp", "--a", "0.5", "--k", "2.0"]
```

### Example 5: Incomplete Partitioning
Use strict mode to create NULL cells where no center has positive influence:

```python
sys.argv = ["centers.shp", "--a", "0.5", "--k", "3.0", "--strict"]
```

---

## Requirements

- QGIS 3.x with Python 3
- GDAL/OGR
- NumPy

---

## References

Renfrew, C., & Level, E. V. (1979). Exploring dominance: Predicting polities from centers. In *Transformations: Mathematical Approaches to Culture Change* (pp. 145-167).

---

## Notes

- Input shapefile must have a `z` attribute for center weights
- Weights are automatically normalized to [0..1]
- Output rasters have 1000x1000 cell resolution by default
- Strict mode sets cells to NULL (NoData) where all centers have I < 0
