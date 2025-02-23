# 📖 XTENT for QGIS
Calculate Influence Areas with Cost or Euclidean Distance.
Model territorial influence for the giggles.

## ✨ Introduction
--------------

This Python script brings Colin Renfrew’s XTENT method into QGIS, letting you map influence areas for settlements using cost-based travel (terrain difficulty, time) or straight-line distance. Perfect for archaeologists, historians, or anyone who’s ever wondered, “How far could Bronze Age Bob flex his grain silo dominance?”

## 🚀 Features
-----------

### Two Modes, One Script 🧮

*   **Cost-Based:** Because walking uphill should matter. Uses raster travel costs (e.g., slope).
    
*   **Euclidean:** Classic “as-the-crow-flies” mode. Great for when you’re feeling nostalgic for 1970s archaeology papers.
    

### Beta Control 🔧

*   Tweak beta to make influence decay faster (**beta > 1**) or slower (**beta < 1**).
    
*   Warning: **Beta = 0** will break physics and Renfrew’s heart, and the script probably will shout at you.
    

### Smooth Operator 🌀

*   Turns jagged raster borders into polished polygons. No more Tetris looking territories.
    

## ⚙️ Requirements
---------------

*   **QGIS 3.x** (with Python console access), wich kinda was assumed but yk
    
*   Imports and libraries needed:
    *   numpy as np. For array magic 🪄
    *   math.Because someone needs to do the math 💡
    *   from qgis.core import (QgsProject, QgsPointXY, QgsSpatialIndex, QgsRectangle, QgsCoordinateReferenceSystem), **QGIS core tools** 🛠️
    *   from osgeo import gdal, osr. For raster wrangling 🤠
    *   from qgis import processing # For polygon smoothness 📐
    
*   **Input Data:**
    
    *   A point layer with a numeric “size” field (population, wealth, or how many goats a settlement has). A number **:)**
        
    *   A cost raster (if using cost mode). Pro tip: Elevation rasters work, but “how many wolves are here” is not a valid cost metric.
        

## 🔧 Installation
---------------

1.  pip install numpy, gdal # The usual suspects 🕵️

2.  **QGIS** imports should be already installed if you have QGIS aswell **Python**     

3.  **Download the Script:** Save XTENT\_for\_QGIS.py somewhere you won’t forget (like next to your “important” vacation photos).
    

## 🎯 Usage
--------

### Configure Settings (Edit the script):

`use_costmodel = True  # Cost-based 🏔️ or Euclidean 📏? Choose wisely.`

`input_layer_name = "your_layer"  # Case-sensitive! QGIS won’t guess.`

`size_field = "wealth"  # Or "goat_count". We don’t judge. 🐐`  

`beta = 2  # Bigger beta = faster influence drop-off. Science!` 

`max_distance = 5000  # Meters (Euclidean) or cost units. Don’t overreach.`

### Run It:

1.  Open **QGIS** → **Python Console** → **Show Editor**.
    
2.  Load the script and click **Run**.
    
3.  Wait. If it crashes, blame GDAL (it is probably your fault).
    

## 📂 Outputs
----------

*   **Dominant\_IDs.tif** 🏆: Shows which settlement dominates each pixel.
    
*   **Influence\_Values.tif** 📈: How strongly they dominate. Higher = more “my goats here!” energy.
    
*   **Smoothed\_Influence.gpkg** 🎨: Polygons so smooth, they’d make a Roman road jealous.
    

## 📜 About XTENT & Colin Renfrew
------------------------------

Colin Renfrew’s 1970s XTENT model answered: _“Who’s the boss here?”_ for ancient settlements. The formula:

$Influence=SizeDistanceβ\\text{Influence} = \\frac{\\text{Size}}{\\text{Distance}^\\beta}$

*   **Size**: Settlement clout (population, resources, or goats).
    
*   **Distance**: Travel cost or straight-line distance.
    
*   **Beta**: Decay rate. Renfrew used **1.0**, but you’re allowed to rebel. 
    

This script modernizes his work, because even theoretical archaeologists deserve GIS tools and what he knew about rasters and all of that.
There are/were another solutions but I found them quite tedious (a lot of dependencies) and outdated

## 🚨 Troubleshooting
------------------

*   **“Layer not found!”** → Check spelling. QGIS is very literal.
    
*   **Invalid CRS** → Use projected coordinates (meters). Latitude won’t cut it. 🌍
    
*   **Cost raster glitches** → Ensure it’s not just a photo of your cat. 🐱 I personally tested it with cost layers created via GRASS.

## 🔧 WIP
----------
*   Would love to make you type in the console your paths, options..., instead of modificating the script.

*   Temporal layers instead of making hardcoded paths
    
## 📜 License
----------

**MIT License** — Use freely, but I’m not responsible if your polygons spark a debate about Neolithic geopolitics.

Coded with ☕ by **Juan**, because someone had to bridge Renfrew and QGIS, or at least give you an updated version
