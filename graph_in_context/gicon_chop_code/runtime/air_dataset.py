from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import metpy.calc as mpcalc
import numpy as np
import xarray as xr
from metpy.units import units


METEO_VARS = ["t2m", "d2m", "tp", "sp", "blh", "msdwswrf", "u100", "v100"]


def coerce_time(value: str | list[Any] | tuple[Any, ...] | datetime | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    if isinstance(value, (list, tuple)):
        parts = value[0] if value and isinstance(value[0], (list, tuple)) else value
        return datetime(*[int(part) for part in parts])
    raise TypeError(f"Unsupported time value: {value!r}")


class GiconFramesQueryDataset:
    """Local KnowAir/GICON query dataset used by the standalone testbed."""

    def __init__(
        self,
        path: str | Path,
        dataset_name: str,
        quest_start_time: str | list[Any],
        quest_end_time: str | list[Any],
        context_start_time: str | list[Any],
        context_end_time: str | list[Any],
        data_start: str | list[Any],
        data_end: str | list[Any],
        meteo_var: list[str] | None = None,
        meteo_use: list[str] | None = None,
        delta_t: int = 24,
        window_size: int = 24,
    ) -> None:
        self.path = Path(path)
        self.dataset_name = dataset_name.lower()
        self.quest_start_time = coerce_time(quest_start_time)
        self.quest_end_time = coerce_time(quest_end_time)
        self.context_start_time = coerce_time(context_start_time)
        self.context_end_time = coerce_time(context_end_time)
        self.data_start = coerce_time(data_start)
        self.data_end = coerce_time(data_end)
        self.meteo_var = meteo_var or METEO_VARS
        self.meteo_use = meteo_use or METEO_VARS
        self.delta_t = int(delta_t)
        self.window_size = int(window_size)

        self.load_netcdf_data()
        self.generate_timestamp()
        self.process_time()
        self.process_feature()
        self.setup_time_ranges()
        self.feature = np.float32(self.feature)
        self.pollution = np.float32(self.pollution)
        self.calc_mean_std()
        self.create_gicon_pairs()
        self.norm()

    def load_netcdf_data(self) -> None:
        if self.dataset_name not in {"bthsa", "yrd"}:
            raise ValueError(f"Unknown dataset_name={self.dataset_name!r}; expected 'bthsa' or 'yrd'")
        nc_file = self.path / f"dataset_{self.dataset_name}.nc"
        if not nc_file.exists():
            raise FileNotFoundError(f"Missing dataset file: {nc_file}")

        self.ds = xr.open_dataset(nc_file, engine="netcdf4")
        self.feature = np.stack([self.ds[var].values for var in METEO_VARS], axis=-1)
        self.pollution = np.stack([self.ds["PM2.5"].values, self.ds["O3"].values], axis=-1)

        print(f"Loaded {self.dataset_name} dataset:")
        print(f"  Feature shape: {self.feature.shape}")
        print(f"  Target shape: {self.pollution.shape} (PM2.5 + O3)")
        print(f"  Time range: {self.ds.time.values[0]} to {self.ds.time.values[-1]}")
        print(f"  Number of stations: {len(self.ds.station)}")

    def generate_timestamp(self) -> None:
        self.timetable = [
            time_value.astype("datetime64[s]").astype(datetime)
            for time_value in self.ds.time.values
        ]

    def get_idx(self, time_value: datetime) -> int:
        if self.data_start is None:
            raise ValueError("data_start is required")
        return int((time_value - self.data_start).total_seconds() / 3600)

    def get_processed_idx(self, time_value: datetime | None) -> int:
        if time_value is None:
            return 0
        return int((time_value - self.processed_data_start).total_seconds() / 3600)

    def process_time(self) -> None:
        start_times = [
            time_value
            for time_value in [self.quest_start_time, self.context_start_time]
            if time_value is not None
        ]
        end_times = [
            time_value
            for time_value in [self.context_end_time]
            if time_value is not None
        ]
        if self.quest_end_time is not None:
            end_times.append(self.quest_end_time + timedelta(hours=self.delta_t))
        if not start_times and self.data_start is not None:
            start_times.append(self.data_start)
        if not end_times and self.data_end is not None:
            end_times.append(self.data_end)

        overall_start_time = min(start_times) if start_times else self.data_start
        overall_end_time = max(end_times) if end_times else self.data_end
        if overall_start_time is None or overall_end_time is None:
            raise ValueError("Unable to resolve dataset time range")
        if self.data_end is not None:
            overall_end_time = min(overall_end_time, self.data_end)

        start_idx = max(0, self.get_idx(overall_start_time))
        end_idx = min(len(self.pollution) - 1, self.get_idx(overall_end_time))
        self.pollution = self.pollution[start_idx : end_idx + 1]
        self.feature = self.feature[start_idx : end_idx + 1]
        self.timetable = self.timetable[start_idx : end_idx + 1]
        self.processed_data_start = self.timetable[0]
        self.processed_data_start_idx = start_idx

    def setup_time_ranges(self) -> None:
        if self.quest_start_time and self.quest_end_time:
            self.quest_start_idx = max(self.window_size - 1, self.get_processed_idx(self.quest_start_time))
            self.quest_end_idx = min(len(self.timetable) - 1, self.get_processed_idx(self.quest_end_time))
        else:
            self.quest_start_idx = self.window_size - 1
            self.quest_end_idx = len(self.timetable) - 1

        if self.context_start_time and self.context_end_time:
            self.context_start_idx = max(0, self.get_processed_idx(self.context_start_time))
            self.context_end_idx = min(len(self.timetable) - 1, self.get_processed_idx(self.context_end_time))
        else:
            self.context_start_idx = 0
            self.context_end_idx = len(self.timetable) - 1

        print(
            f"Quest time range: indices [{self.quest_start_idx}, {self.quest_end_idx}] "
            f"({self.timetable[self.quest_start_idx]} to {self.timetable[self.quest_end_idx]})"
        )
        print(
            f"Context time range: indices [{self.context_start_idx}, {self.context_end_idx}] "
            f"({self.timetable[self.context_start_idx]} to {self.timetable[self.context_end_idx]})"
        )
        print(f"Window size: {self.window_size} frames")

    def process_feature(self) -> None:
        if self.meteo_use:
            indices = [self.meteo_var.index(var) for var in self.meteo_use if var in self.meteo_var]
            if indices:
                self.feature = self.feature[:, :, indices]

        u = self.feature[:, :, -2] * units.meter / units.second
        v = self.feature[:, :, -1] * units.meter / units.second
        speed = 3.6 * mpcalc.wind_speed(u, v)._magnitude
        direction = mpcalc.wind_direction(u, v)._magnitude

        hours = np.asarray([time_value.hour for time_value in self.timetable], dtype=np.float32)
        hours = np.repeat(hours[:, None], self.feature.shape[1], axis=1)
        self.feature = np.concatenate(
            [self.feature, hours[:, :, None], speed[:, :, None], direction[:, :, None]],
            axis=-1,
        )

    def calc_mean_std(self) -> None:
        context_features = self.feature[self.context_start_idx : self.context_end_idx + 1]
        context_pollution = self.pollution[self.context_start_idx : self.context_end_idx + 1]
        self.feature_mean = context_features.mean(axis=(0, 1))
        self.feature_std = context_features.std(axis=(0, 1))
        self.pollution_mean = context_pollution.mean(axis=(0, 1))
        self.pollution_std = context_pollution.std(axis=(0, 1))
        print("Calculated normalization statistics from context data range:")
        print(f"  Context indices: [{self.context_start_idx}, {self.context_end_idx}]")
        print(
            f"  Context time range: {self.timetable[self.context_start_idx]} "
            f"to {self.timetable[self.context_end_idx]}"
        )

    def create_gicon_pairs(self) -> None:
        self.combined_data = np.concatenate([self.feature, self.pollution], axis=-1)
        self.num_stations = self.feature.shape[1]
        self.feature_dim = self.feature.shape[-1]
        self.pollution_dim = self.pollution.shape[-1]

    def norm(self) -> None:
        self.combined_mean = np.concatenate([self.feature_mean, self.pollution_mean])
        self.combined_std = np.maximum(
            np.concatenate([self.feature_std, self.pollution_std]),
            1e-8,
        )
        self.combined_data = (self.combined_data - self.combined_mean) / self.combined_std

    def get_window_data(self, end_time: int) -> np.ndarray:
        start_time = end_time - self.window_size + 1
        return self.combined_data[start_time : end_time + 1]

    def __len__(self) -> int:
        return sum(
            1
            for quest_time in range(self.quest_start_idx, self.quest_end_idx + 1)
            if quest_time + self.delta_t < len(self.combined_data)
        )
