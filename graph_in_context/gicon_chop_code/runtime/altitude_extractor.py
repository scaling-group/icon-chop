"""
Altitude extraction utility for spatial grids.

This module provides functionality to extract altitude values from a spatial grid
using station coordinates through interpolation.
"""

from pathlib import Path

import numpy as np
import pandas as pd


class AltitudeExtractor:
    """
    Extract altitude values from spatial grid using station coordinates.

    The spatial grid is assumed to be a regular lat-lon grid where:
    - Grid shape: (lat_points, lon_points)
    - Grid covers the specified geographic region
    """

    def __init__(
        self,
        altitude_grid_path: str,
        lat_bounds: tuple[float, float] = (15.0, 55.0),  # Default: China latitude bounds
        lon_bounds: tuple[float, float] = (70.0, 140.0),  # Default: China longitude bounds
    ):
        """
        Initialize the altitude extractor.

        Args:
            altitude_grid_path: Path to the altitude.npy file
            lat_bounds: (min_lat, max_lat) bounds of the grid
            lon_bounds: (min_lon, max_lon) bounds of the grid
        """
        self.altitude_grid_path = Path(altitude_grid_path)
        self.lat_bounds = lat_bounds
        self.lon_bounds = lon_bounds

        # Load altitude grid
        self.altitude_grid = np.load(str(self.altitude_grid_path))
        self.grid_shape = self.altitude_grid.shape

        # Calculate grid resolution
        self.lat_resolution = (lat_bounds[1] - lat_bounds[0]) / (self.grid_shape[0] - 1)
        self.lon_resolution = (lon_bounds[1] - lon_bounds[0]) / (self.grid_shape[1] - 1)

        print(f"Loaded altitude grid: {self.grid_shape}")
        print(f"Latitude bounds: {lat_bounds}")
        print(f"Longitude bounds: {lon_bounds}")
        print(f"Grid resolution: {self.lat_resolution:.4f}° lat, {self.lon_resolution:.4f}° lon")

    def coord_to_grid_index(self, lat: float, lon: float) -> tuple[int, int]:
        """
        Convert geographic coordinates to grid indices.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees

        Returns:
            (lat_idx, lon_idx): Grid indices
        """
        # Convert to grid coordinates
        lat_idx = int((self.lat_bounds[1] - lat) / self.lat_resolution)  # Reverse for typical grid layout
        lon_idx = int((lon - self.lon_bounds[0]) / self.lon_resolution)

        # Clamp to valid range
        lat_idx = max(0, min(self.grid_shape[0] - 1, lat_idx))
        lon_idx = max(0, min(self.grid_shape[1] - 1, lon_idx))

        return lat_idx, lon_idx

    def extract_altitude_nearest(self, lat: float, lon: float) -> float:
        """
        Extract altitude using nearest neighbor interpolation.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees

        Returns:
            Altitude value in meters
        """
        lat_idx, lon_idx = self.coord_to_grid_index(lat, lon)
        return float(self.altitude_grid[lat_idx, lon_idx])

    def extract_altitude_bilinear(self, lat: float, lon: float) -> float:
        """
        Extract altitude using bilinear interpolation.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees

        Returns:
            Interpolated altitude value in meters
        """
        # Convert to continuous grid coordinates
        lat_coord = (self.lat_bounds[1] - lat) / self.lat_resolution
        lon_coord = (lon - self.lon_bounds[0]) / self.lon_resolution

        # Get surrounding grid points
        lat_idx0 = int(np.floor(lat_coord))
        lat_idx1 = min(lat_idx0 + 1, self.grid_shape[0] - 1)
        lon_idx0 = int(np.floor(lon_coord))
        lon_idx1 = min(lon_idx0 + 1, self.grid_shape[1] - 1)

        # Clamp to valid range
        lat_idx0 = max(0, lat_idx0)
        lon_idx0 = max(0, lon_idx0)

        # Get fractional parts
        lat_frac = lat_coord - lat_idx0
        lon_frac = lon_coord - lon_idx0

        # Get corner values
        z00 = self.altitude_grid[lat_idx0, lon_idx0]
        z01 = self.altitude_grid[lat_idx0, lon_idx1]
        z10 = self.altitude_grid[lat_idx1, lon_idx0]
        z11 = self.altitude_grid[lat_idx1, lon_idx1]

        # Bilinear interpolation
        z0 = z00 * (1 - lon_frac) + z01 * lon_frac
        z1 = z10 * (1 - lon_frac) + z11 * lon_frac
        altitude = z0 * (1 - lat_frac) + z1 * lat_frac

        return float(altitude)

    def extract_altitude(self, lat: float, lon: float, method: str = "bilinear") -> float:
        """
        Extract altitude using specified interpolation method.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            method: Interpolation method ('nearest' or 'bilinear')

        Returns:
            Altitude value in meters
        """
        if method == "nearest":
            return self.extract_altitude_nearest(lat, lon)
        elif method == "bilinear":
            return self.extract_altitude_bilinear(lat, lon)
        else:
            raise ValueError(f"Unknown interpolation method: {method}")

    def extract_altitudes_from_coordinates(
        self, coordinates: list[tuple[float, float]], method: str = "bilinear"
    ) -> np.ndarray:
        """
        Extract altitudes for a list of coordinates.

        Args:
            coordinates: List of (lat, lon) tuples
            method: Interpolation method ('nearest' or 'bilinear')

        Returns:
            Array of altitude values
        """
        altitudes = []
        for lat, lon in coordinates:
            altitude = self.extract_altitude(lat, lon, method)
            altitudes.append(altitude)

        return np.array(altitudes, dtype=np.float32)

    def extract_altitudes_from_stations(
        self, stations_csv_path: str, lat_col: str = "lat", lon_col: str = "lon", method: str = "bilinear"
    ) -> np.ndarray:
        """
        Extract altitudes for all stations in a CSV file.

        Args:
            stations_csv_path: Path to stations CSV file
            lat_col: Name of latitude column
            lon_col: Name of longitude column
            method: Interpolation method ('nearest' or 'bilinear')

        Returns:
            Array of altitude values for each station
        """
        # Load stations data
        stations_df = pd.read_csv(stations_csv_path)

        if lat_col not in stations_df.columns or lon_col not in stations_df.columns:
            raise ValueError(f"Stations CSV must contain '{lat_col}' and '{lon_col}' columns")

        coordinates = [(row[lat_col], row[lon_col]) for _, row in stations_df.iterrows()]
        return self.extract_altitudes_from_coordinates(coordinates, method)

    def save_station_altitudes(
        self,
        stations_csv_path: str,
        output_path: str,
        lat_col: str = "lat",
        lon_col: str = "lon",
        method: str = "bilinear",
    ) -> None:
        """
        Extract and save station altitudes to a numpy file.

        Args:
            stations_csv_path: Path to stations CSV file
            output_path: Path to save the altitude array
            lat_col: Name of latitude column
            lon_col: Name of longitude column
            method: Interpolation method ('nearest' or 'bilinear')
        """
        altitudes = self.extract_altitudes_from_stations(stations_csv_path, lat_col, lon_col, method)
        np.save(output_path, altitudes)

        # Print summary
        stations_df = pd.read_csv(stations_csv_path)
        print(f"Extracted altitudes for {len(altitudes)} stations")
        print(f"Altitude range: {altitudes.min():.1f} - {altitudes.max():.1f} meters")
        print(f"Mean altitude: {altitudes.mean():.1f} meters")
        print(f"Saved to: {output_path}")

        # Show sample of stations with altitudes if station info available
        if len(stations_df) > 0:
            sample_size = min(5, len(stations_df))
            sample_df = stations_df.head(sample_size).copy()
            sample_df["altitude"] = altitudes[:sample_size]

            display_cols = [lat_col, lon_col, "altitude"]
            # Add additional columns if they exist
            for col in ["station_id", "station_name", "city"]:
                if col in sample_df.columns:
                    display_cols.insert(-1, col)

            print("\nSample stations with extracted altitudes:")
            print(sample_df[display_cols])


if __name__ == "__main__":
    # Example usage
    import os

    # Get the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    # Example: Extract altitudes for station coordinates
    extractor = AltitudeExtractor(altitude_grid_path=os.path.join(project_root, "data/air/altitude.npy"))

    # Extract single coordinate
    beijing_lat, beijing_lon = 39.9042, 116.4074
    altitude = extractor.extract_altitude(beijing_lat, beijing_lon)
    print(f"Beijing altitude: {altitude:.1f} meters")

    # Extract from coordinate list
    coordinates = [(39.9042, 116.4074), (31.2304, 121.4737)]  # Beijing, Shanghai
    altitudes = extractor.extract_altitudes_from_coordinates(coordinates)
    print(f"Altitudes: {altitudes}")
