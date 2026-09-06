import os
from collections import OrderedDict

import metpy.calc as mpcalc
import numpy as np
import pandas as pd
from bresenham import bresenham
from geopy.distance import geodesic
from metpy.units import units
from scipy.spatial import distance

from .altitude_extractor import AltitudeExtractor


class Graph:
    def __init__(self, station_fp, altitude_fp, use_altitude=True):
        """
        Initialize Graph class using direct file paths.

        Args:
            station_fp: Full path to the station CSV file
            altitude_fp: Full path to the altitude grid file
            use_altitude: Whether to use altitude information for edge filtering
        """
        self.dist_thres = 3
        self.alti_thres = 1200
        self.use_altitude = use_altitude
        self.station_fp = station_fp
        self.altitude_fp = altitude_fp

        # Always use AltitudeExtractor for consistent altitude handling
        if os.path.exists(self.altitude_fp):
            self.altitude_extractor = AltitudeExtractor(self.altitude_fp)
            self.altitude = self.altitude_extractor.altitude_grid
        else:
            raise FileNotFoundError(f"Missing altitude file: {self.altitude_fp}")
        self.nodes = self.generate_nodes()
        self.node_attr = self.add_node_attr()
        self.node_num = len(self.nodes)
        self.edge_index, self.edge_attr = self.generate_edges()
        if self.use_altitude:
            self.update_edges()
        self.edge_num = self.edge_index.shape[1]
        self.adj = np.zeros((self.node_num, self.node_num), dtype=np.uint8)
        self.adj[self.edge_index[0], self.edge_index[1]] = 1

    def lonlat2xy(self, lon, lat, is_aliti=True):
        """
        Convert lon/lat coordinates to grid indices.

        If AltitudeExtractor is available, use its coordinate system.
        Otherwise, fall back to hardcoded parameters for backwards compatibility.
        """
        if self.altitude_extractor is not None:
            # Use AltitudeExtractor's coordinate system
            lat_idx, lon_idx = self.altitude_extractor.coord_to_grid_index(lat, lon)
            return lon_idx, lat_idx  # Return as (x, y) to match original function
        else:
            # Legacy hardcoded parameters for backwards compatibility
            if is_aliti:
                lon_l = 100.0
                lat_u = 48.0
                res = 0.05
            else:
                lon_l = 103.0
                lat_u = 42.0
                res = 0.125
            x = np.int64(np.round((lon - lon_l - res / 2) / res))
            y = np.int64(np.round((lat_u + res / 2 - lat) / res))
            return x, y

    def generate_nodes(self):
        """Generate nodes from station CSV file."""
        nodes = OrderedDict()

        # Load station CSV file
        if not os.path.isfile(self.station_fp):
            raise FileNotFoundError(f"Station CSV file not found: {self.station_fp}")

        print(f"Loading stations from: {self.station_fp}")
        stations_df = pd.read_csv(self.station_fp)

        for idx, row in stations_df.iterrows():
            station_id = row["station_id"]
            city = row.get("city", row.get("station_name", "unknown"))
            lon, lat = float(row["lon"]), float(row["lat"])

            # Get altitude using extractor or grid lookup
            if self.altitude_extractor is not None:
                altitude = self.altitude_extractor.extract_altitude(lat, lon)
            else:
                x, y = self.lonlat2xy(lon, lat, True)
                altitude = self.altitude[y, x]

            nodes.update({idx: {"city": city, "altitude": altitude, "lon": lon, "lat": lat, "station_id": station_id}})

        return nodes

    def add_node_attr(self):
        node_attr = []
        altitude_arr = []
        for i in self.nodes:
            altitude = self.nodes[i]["altitude"]
            altitude_arr.append(altitude)
        altitude_arr = np.stack(altitude_arr)
        node_attr = np.stack([altitude_arr], axis=-1)
        return node_attr

    def traverse_graph(self):
        lons = []
        lats = []
        citys = []
        idx = []
        for i in self.nodes:
            idx.append(i)
            city = self.nodes[i]["city"]
            lon, lat = self.nodes[i]["lon"], self.nodes[i]["lat"]
            lons.append(lon)
            lats.append(lat)
            citys.append(city)
        return idx, citys, lons, lats

    def dense_to_sparse_numpy(self, adj_matrix):
        row, col = np.nonzero(adj_matrix)
        edge_index = np.vstack((row, col))
        edge_attr = adj_matrix[row, col]
        return edge_index, edge_attr

    def generate_edges(self):
        coords = []
        for i in self.nodes:
            coords.append([self.nodes[i]["lon"], self.nodes[i]["lat"]])
        dist = distance.cdist(coords, coords, "euclidean")
        adj = np.zeros((self.node_num, self.node_num), dtype=np.uint8)
        adj[dist <= self.dist_thres] = 1
        assert adj.shape == dist.shape
        dist = dist * adj
        edge_index, dist = self.dense_to_sparse_numpy(dist)

        direc_arr = []
        dist_kilometer = []
        for i in range(edge_index.shape[1]):
            src, dest = edge_index[0, i], edge_index[1, i]
            src_lat, src_lon = self.nodes[src]["lat"], self.nodes[src]["lon"]
            dest_lat, dest_lon = self.nodes[dest]["lat"], self.nodes[dest]["lon"]
            src_location = (src_lat, src_lon)
            dest_location = (dest_lat, dest_lon)
            dist_km = geodesic(src_location, dest_location).kilometers
            v, u = src_lat - dest_lat, src_lon - dest_lon

            u = u * units.meter / units.second
            v = v * units.meter / units.second
            direc = mpcalc.wind_direction(u, v)._magnitude

            direc_arr.append(direc)
            dist_kilometer.append(dist_km)

        direc_arr = np.stack(direc_arr)
        dist_arr = np.stack(dist_kilometer)
        attr = np.stack([dist_arr, direc_arr], axis=-1)

        return edge_index, attr

    def update_edges(self):
        edge_index = []
        edge_attr = []
        for i in range(self.edge_index.shape[1]):
            src, dest = self.edge_index[0, i], self.edge_index[1, i]
            src_lat, src_lon = self.nodes[src]["lat"], self.nodes[src]["lon"]
            dest_lat, dest_lon = self.nodes[dest]["lat"], self.nodes[dest]["lon"]
            src_x, src_y = self.lonlat2xy(src_lon, src_lat, True)
            dest_x, dest_y = self.lonlat2xy(dest_lon, dest_lat, True)
            points = np.asarray(list(bresenham(src_y, src_x, dest_y, dest_x))).transpose((1, 0))
            altitude_points = self.altitude[points[0], points[1]]
            altitude_src = self.altitude[src_y, src_x]
            altitude_dest = self.altitude[dest_y, dest_x]
            if (
                np.sum(altitude_points - altitude_src > self.alti_thres) < 3
                and np.sum(altitude_points - altitude_dest > self.alti_thres) < 3
            ):
                edge_index.append(self.edge_index[:, i])
                edge_attr.append(self.edge_attr[i])

        self.edge_index = np.stack(edge_index, axis=1)
        self.edge_attr = np.stack(edge_attr, axis=0)


if __name__ == "__main__":
    # Example usage - update paths as needed
    station_fp = "path/to/stations.csv"
    altitude_fp = "path/to/altitude.npy"
    graph = Graph(station_fp=station_fp, altitude_fp=altitude_fp)
