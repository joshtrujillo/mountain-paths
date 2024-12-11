# Mountain Paths
This python program reads the elevation data from a GeoTIFF file and finds the path of least elevation change between two points. The program uses three different algorithms: greedy, a*, and dijkstra. The paths are visualized with matplotlib, folium, and gpxpy. The grayscale elevation data grid and path are displayed, an .html file is created to visualize the path over a topographic map, and a .gpx file is created for use in gps software like CalTopo, Gaia, etc.
## Usage
### Run the program
```
python MountainPaths.py --file [.tif filepath] --algorithm [algorithm index] --animate [optional flag, enable to see algorithm animation]
```
For the `--algorithm` flag, choose `0` for Greedy, `1` for A*, and `2` for Dijkstra.
