"""
MountainPaths.py
CMPT 306: Algorithms
Final Project
@author Josh Trujillo

This program finds the path with the least ammount of elevation change
accross a geological region. This program uses the bridges library to get
elevation data from a bounding box defined by lat and long. A ColorGrid
representative of the elevation data is created to visualize the path
found by the algorithm.
"""

import os
import argparse
import rasterio
import folium
import gpxpy
import gpxpy.gpx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from heapq import heappush, heappop

from typing import Tuple

from rasterio.windows import from_bounds
from rasterio.transform import rowcol, xy


class Node:
    def __init__(self, state: Tuple[int, int], cost_from_start: int, parent=None):
        """
        Args:
            state: Pixel coordinates of node.
            cost_from_start: Cost from start to this node.
            parent: The previous node if applicable.
        """
        self.state = state
        self.cost_from_start = cost_from_start
        self.parent = parent


class MountainPath:
    def __init__(
        self,
        algorithm: str,
        file_path: str,
        animate: bool,
        elevation_data,
        bounding_box: Tuple[float, float, float, float],
        transform,
        start: Tuple[float, float],
        end: Tuple[float, float],
    ):
        """
        Args:
            algorithm: Which algorithm is to be used to find the path.
            animate: Animate if true.
            elevation_data: 2d array of elevation data.
            bounding_box: Area to visualize.
            transform: rasterio affine transform to get lat, lon from pixels and vice versa.
            start: Starting lat, lon.
            end: Ending lat, lon.

        Fields:
            algorithm: Greedy or A*.
            bounding_box: Area to visualize.
            transform: Affine transform.
            elevation_data: 2d array of elevation data.
            rows, cols: Shape of elevation data.
            visited: Set of visited states for optimization.
        """
        self.algorithm = algorithm
        self.file_path = file_path
        self.animate = animate
        self.bounding_box = bounding_box
        self.transform = transform
        self.elevation_data = elevation_data
        self.rows, self.cols = elevation_data.shape
        self.visited = set()
        self.start = self.lat_long_to_pixel(start[0], start[1])
        self.end = self.lat_long_to_pixel(end[0], end[1])

    def goal_test(self, current_state: Tuple[int, int]) -> bool:
        """
        Tests if the passed state is the goal.

        Args:
            current_state: State to test.

        Returns:
            bool
        """
        return True if current_state == self.end else False

    def priority(self, node: Node):
        """
        Calculates the priority of a node depending on the algorithm used.
        The heuristic used for A* calculates the absolute difference between
        the current elevation and the goal elevation.

        Args:
            node: Node object to calculate priority.

        Returns:
            int: The calculated priority.
        """
        match self.algorithm:
            case "Greedy":
                if node.parent:
                    return self.get_cost(node.parent.state, node.state)
                else:
                    return 0
            case "AStar":
                # Estimate elevation difference to goal
                current_elevation = self.elevation_data[node.state[0], node.state[1]]
                goal_elevation = self.elevation_data[self.end[0], self.end[1]]
                elevation_diff = abs(current_elevation - goal_elevation)
                return node.cost_from_start + self.heuristic(node.state)
            case "Dijkstra":
                if node.parent:
                    return node.cost_from_start
                else:
                    return 0

    def heuristic(self, state: Tuple[int, int]):
        current_row, current_col = state
        end_row, end_col = self.end
        current_elevation = self.elevation_data[current_row, current_col]
        distance = ((current_row - end_row) ** 2 + (current_col - end_col) ** 2) ** 0.5
        return abs(current_elevation - self.elevation_data[self.end]) / max(1, distance)

    def get_cost(
        self, current_state: Tuple[int, int], next_state: Tuple[int, int]
    ) -> int:
        """
        Calculates the elevation change from current position to next position.

        Args:
            current_state: Tuple for the row, col coordinate
                of the current location.
            next_state: Tuple for the row, col coordinate
                of the next location.
        Returns:
            int: the elevation change.
        """
        current_elevation = self.elevation_data[current_state[0], current_state[1]]
        next_elevation = self.elevation_data[next_state[0], next_state[1]]
        return abs(current_elevation - next_elevation)

    def get_successors(self, state: Tuple[int, int]) -> list[Tuple]:
        """
        Finds all possible next states given the current state.

        Args:
            state: a tuple for the row, col coordinate
                of the current location.

        Returns:
            list[Tuple[int, int]] of successors
        """
        successors = []
        directions = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
        for dr, dc in directions:
            new_row, new_col = state[0] + dr, state[1] + dc
            if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
                successors.append((new_row, new_col))
        return successors

    def solve(self, update_interval=10):
        """
        Finds the path with the least elevation change. Uses either greedy or
        A* algorithm. After a path is found, solve() calls self.draw() to
        draw the path on the bridges ColorGrid.
        """
        fringe = []
        node = Node(self.start, 0, None)
        heappush(fringe, (0, 0, node.state, node))
        steps = []
        count = 0
        final_path = None
        # loop while there are unexplored nodes in the fringe.
        while fringe:
            _, _, _, node = heappop(fringe)
            if len(steps) == 0 or count % update_interval == 0:
                steps.append(node)
            successsors = self.get_successors(node.state)
            for next_state in successsors:
                if next_state in self.visited:
                    continue
                self.visited.add(next_state)
                next_cost = node.cost_from_start + self.get_cost(node.state, next_state)
                next_node = Node(next_state, next_cost, node)
                if self.goal_test(next_state):
                    final_path = next_node
                    break
                heappush(
                    fringe,
                    (
                        self.priority(next_node),
                        abs(node.state[0] - next_state[0]),
                        next_state,
                        next_node,
                    ),
                )
            count += 1
            if final_path:
                print(f"Total Elevation Change (meters): {final_path.cost_from_start}")
                break

        if self.animate:
            # Animation
            fig, ax = plt.subplots(figsize=(10, 8))
            image = ax.imshow(self.elevation_data, cmap="gray", aspect="equal")
            scatter = ax.scatter([], [], c="red", s=1)  # Scatter for the path
            # Set plot labels and colorbar
            scatter.set_label("Path of least elevation change")
            cost_label = ax.text(
                5, 15, "", fontsize=14, color="orange"
            )  # Text for displaying current cost
            ax.set_title(f"{self.algorithm} Pathfinding in Progress")
            ax.set_xlabel("Column Index (Longitude)")
            ax.set_ylabel("Row Index (Latitude)")
            cbar = plt.colorbar(image, ax=ax)
            cbar.set_label("Elevation (meters)")
            plt.text(
                self.start[1],
                self.start[0],
                "Start",
                color="blue",
                fontsize=12,
                ha="right",
                va="bottom",
            )
            plt.text(
                self.end[1],
                self.end[0],
                "End",
                color="green",
                fontsize=12,
                ha="left",
                va="top",
            )
            plt.legend()

            # Animation update function
            def update(frame):
                node = steps[frame]
                path = self.extract_path(node)
                scatter.set_offsets(
                    [(p[1], p[0]) for p in path]
                )  # Update scatter points
                cost_label.set_text(f"Current Lowest Cost: {node.cost_from_start} m")
                return (scatter, cost_label)

            # Total number of frames is the number of steps
            ani = FuncAnimation(fig, update, frames=len(steps), interval=0, blit=True)
            # Save the animation as .mp4
            ani.save(
                f"{self.algorithm}_{self.file_path.rstrip(".tif")}.mp4",
                writer="ffmpeg",
                fps=30,
            )
            # Show the animation
            plt.show()
        # Draw the final path
        if final_path:
            self.draw_grayscale(final_path)
            self.draw_topo_map(final_path)

    def extract_path(self, node):
        """
        Extracts the path from the given node back to the start.

        Args:
            node (Node): The node to trace back from.

        Returns:
            list[Tuple[int, int]]: List of (row, col) coordinates in the path.
        """
        path = []
        while node:
            path.append(node.state)
            node = node.parent
        return path[::-1]  # Reverse the path to start -> end

    def draw_grayscale(self, node: Node):
        """
        Draws the final path on a grayscale elevation map.

        Args:
            node (Node): The last node in the path found by the algorithm.
        """
        # Extract the full path from the final node
        path = self.extract_path(node)

        # Create a figure and plot
        plt.figure(figsize=(10, 8))
        image = plt.imshow(self.elevation_data, cmap="gray", aspect="equal")

        # Overlay the full path in red
        line = plt.scatter(
            [p[1] for p in path], [p[0] for p in path], c="red", s=1, label="Path"
        )
        line.set_label("Path of least elevation change")

        # Annotate start and end points
        start, end = path[0], path[-1]
        plt.text(
            start[1],
            start[0],
            "Start",
            color="blue",
            fontsize=12,
            ha="right",
            va="bottom",
        )
        plt.text(end[1], end[0], "End", color="green", fontsize=12, ha="left", va="top")
        plt.text(
            5,
            15,
            f"Final cost: {node.cost_from_start} m",
            fontsize=14,
            color="orange",
        )  # Text for displaying current cost

        # Add labels, title, and legend
        plt.title(f"Final Path on Elevation Map found with {self.algorithm}")
        plt.xlabel("Column Index (Longitude)")
        plt.ylabel("Row Index (Latitude)")
        cbar = plt.colorbar(image)
        cbar.set_label("Elevation (meters)")
        plt.legend()

        # Display the final path
        plt.tight_layout()
        plt.show()

    def draw_topo_map(self, node: Node):
        """
        Accesses all parent nodes of a given node to draw a red path based on lat, lon
        over a topo map, which is saved as a .html file.

        Args:
            node: The last node in the path found by the algorithm.
        """
        # find lat, lon path
        path = []
        while node.parent:
            row, col = node.state
            lat, lon = self.pixel_to_lat_long(row, col)
            path.append((lat, lon))
            node = node.parent
        row, col = node.state
        lat, lon = self.pixel_to_lat_long(row, col)
        path.append((lat, lon))
        # get start and end points
        start, end = path[-1], path[0]
        # set the map location
        min_lon, max_lon = self.bounding_box[0], self.bounding_box[2]
        min_lat, max_lat = self.bounding_box[1], self.bounding_box[3]
        map_center = [
            (self.bounding_box[1] + self.bounding_box[3]) / 2,  # Average latitude
            (self.bounding_box[0] + self.bounding_box[2]) / 2,  # Average longitude
        ]
        # create map with USGS topo
        map = folium.Map(
            tiles="https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/tile/{z}/{y}/{x}",
            attr='Tiles courtesy of the <a href="https://usgs.gov/">U.S. Geological Survey</a>',
            max_bounds=True,
            location=map_center,
            control_scale=True,
            zoom_control=False,
            zoom_start=15,
            min_lat=min_lat,
            max_lat=max_lat,
            min_lon=min_lon,
            max_lon=max_lon,
        )
        folium.TileLayer(
            tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
            attr='Map data: &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, <a href="http://viewfinderpanoramas.org">SRTM</a> | Map style: &copy; <a href="https://opentopomap.org">OpenTopoMap</a> (<a href="https://creativecommons.org/licenses/by-sa/3.0/">CC-BY-SA</a>)',
        ).add_to(map)
        folium.LayerControl().add_to(map)
        # add start and end markers
        folium.Marker(
            location=start,
            tooltip="Start",
            popup="Start of least elevation path.",
            icon=folium.Icon(icon="map_pin", color="green"),
        ).add_to(map)
        folium.Marker(
            location=end,
            tooltip="End",
            popup="End of least elevation path.",
            icon=folium.Icon(icon="map_pin", color="red"),
        ).add_to(map)
        # draw path
        folium.PolyLine(
            path, color="red", weight=2.5, tooltip=f"{self.algorithm}"
        ).add_to(map)
        # save .gpx of path
        self.export_to_gpx(path)
        # save map as html
        file_name = f"{self.algorithm.lower()}_{self.file_path.rstrip(".tif")}.html"
        map.save(file_name)
        full_path = os.path.abspath(file_name)
        print(f"HTML file saved at file://{full_path}")

    def pixel_to_lat_long(self, row: int, col: int) -> Tuple[float, float]:
        """
        Converts pixel coordinates to latitude and logitude using the affine transform.
        Args:
            row (int): Row of the pixel.
            col (int): Column of the pixel.
        Returns:
            Tuple[float, float]: latitude and longitude.
        """
        lon, lat = xy(self.transform, row, col)
        return lat, lon

    def lat_long_to_pixel(self, lat: float, lon: float) -> Tuple[int, int]:
        """
        Converts latitude and longitude to pixel coordinates using the affine transform.
        Args:
            lat (float): Latitude of the point.
            lon (float): Longitude of the point.
        Returns:
            Tuple[int, int]: pixel coordinates.
        """
        row, col = rowcol(self.transform, lon, lat)
        return int(row), int(col)

    def export_to_gpx(self, lat_lon_path: list[Tuple[float, float]]):
        """
        Exports the given latitude/longitude path to a GPX file.

        Args:
            lat_lon_path (list of tuple): List of (latitude, longitude) coordinates.
        """
        # Create a GPX object
        gpx = gpxpy.gpx.GPX()
        # Create a GPX track
        gpx_track = gpxpy.gpx.GPXTrack()
        gpx.tracks.append(gpx_track)
        # Create a segment in the track
        gpx_segment = gpxpy.gpx.GPXTrackSegment()
        gpx_track.segments.append(gpx_segment)
        # Add points to the segment
        for lat, lon in lat_lon_path:
            gpx_segment.points.append(
                gpxpy.gpx.GPXTrackPoint(latitude=lat, longitude=lon)
            )
        # Write to a file
        output_file = (
            f"{self.algorithm.lower()}_{self.file_path.rstrip(".tif")}_path.gpx"
        )
        with open(output_file, "w") as f:
            f.write(gpx.to_xml())
        print(f"GPX file saved as {output_file}")


def get_bounding_box(
    point1: Tuple[float, float], point2: Tuple[float, float]
) -> Tuple[float, float, float, float]:
    """
    Calculates the bounding box from two points.
    Args:
        point1 (tuple): (latitude, longitude) of the first point.
        point2 (tuple): (latitude, longitude) of the second point.
    Returns:
        tuple: (min_lon, min_lat, max_lon, max_lat)
    """
    lat_min = min(point1[0], point2[0])
    lat_max = max(point1[0], point2[0])
    lon_min = min(point1[1], point2[1])
    lon_max = max(point1[1], point2[1])
    # buffer so start and end points aren't on the edge of the map area.
    buffer_ratio = 0.1
    lat_buffer = (lat_max - lat_min) * buffer_ratio
    lon_buffer = (lon_max - lon_min) * buffer_ratio
    return (
        lon_min - lon_buffer,
        lat_min - lat_buffer,
        lon_max + lon_buffer,
        lat_max + lat_buffer,
    )


def extract_region(file_path: str, bounding_box: Tuple[float, float, float, float]):
    """
    Extracts the region of interest from a GeoTIFF file based on the bounding box.
    Args:
        file_path (str): Path to the GeoTIFF file.
        bounding_box (tuple): (min_lon, min_lat, max_lon, max_lat).
    Returns:
        np.array: Elevation data for the region.
        Affine: Transform for the extracted region.
    """
    with rasterio.open(file_path) as src:
        print(f"Width: {src.width}, Height: {src.height}")
        # Convert bounding box to pixel window
        window = from_bounds(*bounding_box, transform=src.transform)
        # Read the elevation data within the window
        elevation_data = src.read(1, window=window)
        # Get the transform for the new cropped region
        transform = src.window_transform(window)
    return elevation_data, transform


def main():
    # Parsing
    # Use python MountainPaths.py -algorithm <0 for Greedy, 1 for A*, 2 for Dijkstra> -file <path/to/GeoTIF/file.tif>
    algorithms = ["Greedy", "AStar", "Dijkstra"]
    parser = argparse.ArgumentParser(description="mountain path")
    parser.add_argument(
        "--algorithm",
        dest="algorithm_index",
        required=True,
        type=int,
        help="index of algorithm",
    )
    parser.add_argument(
        "--file",
        dest="file_path",
        required=True,
        type=str,
        help="Path of GeoTIF file",
    )
    parser.add_argument(
        "--animate",
        dest="animate",
        action="store_true",
        help="Include this flag to enable animation",
    )
    args = parser.parse_args()

    # death valley to mt whitney.
    # start = (36.20043, -116.85046)
    # end = (36.57849, -118.29238)

    # lone peak to gobblers knob.
    start = (40.52633, -111.75532)
    end = (40.67089, -111.68265)

    # home to hidden peak.
    # start = (40.61806, -111.84917)
    # end = (40.56093, -111.64511)

    # Get elevation data from GeoTIFF
    bounding_box = get_bounding_box(start, end)
    elevation_data, transform = extract_region(args.file_path, bounding_box)

    # Create MountainPath object and solve.
    mountain_path = MountainPath(
        algorithms[args.algorithm_index],
        args.file_path,
        args.animate,
        elevation_data,
        bounding_box,
        transform,
        start,
        end,
    )
    mountain_path.solve(70)


if __name__ == "__main__":
    main()
