import numpy as np
from pyproj import Transformer


class ENUConverter:
    def __init__(self, lat0, lon0):
        self.lat0 = lat0
        self.lon0 = lon0

        self.transformer = Transformer.from_crs(
            "epsg:4326",
            "epsg:32639",  # Tehran zone
            always_xy=True
        )

        # Reverse Projection: UTM Zone 39N -> WGS84
        self.inverse_transformer = Transformer.from_crs(
            "epsg:32639",
            "epsg:4326",
            always_xy=True
        )


        self.origin_x, self.origin_y = self.transformer.transform(lon0, lat0)

    def to_enu(self, lat, lon):
        x, y = self.transformer.transform(lon, lat)
        return np.array([x - self.origin_x, y - self.origin_y])

    def to_latlon(self, e, n):
        x = e + self.origin_x
        y = n + self.origin_y
        lon, lat = self.inverse_transformer.transform(x, y)
        return lat, lon