import os
import folium
import numpy as np
import pandas as pd
from src.geo.enu import ENUConverter
from src.scenario.anchors import AnchorGenerator


class LocalizationVisualizer:
    # ADDED: x_range and y_range parameters
    def __init__(self, lat0, lon0, x_range=(0, 1000), y_range=(0, 1000), results_path="results.csv"):
        self.converter = ENUConverter(lat0, lon0)
        self.results_path = results_path
        self.x_range = x_range
        self.y_range = y_range

        if not os.path.exists(results_path):
            raise FileNotFoundError(f"No results file found at {results_path}. Run experiments first!")
        self.df = pd.read_csv(results_path)

    def generate_run_map(self, run_id=0, output_filename=None):
        run_data = self.df[self.df["run_id"] == run_id]
        if run_data.empty:
            raise ValueError(f"Run ID {run_id} not found in dataset.")

        if output_filename is None:
            output_filename = f"map_run_{run_id}.html"

        anchor_count = run_data["anchor_count"].iloc[0]

        anchor_seed = 42 + run_id
        # UPDATED: Use dynamic bounds instead of (0, 1000)
        anchors_enu = AnchorGenerator(
            anchor_count, self.x_range, self.y_range, seed=anchor_seed
        ).generate()

        m = folium.Map(
            location=[self.converter.lat0, self.converter.lon0],
            zoom_start=15,
            control_scale=True,
            tiles="OpenTopoMap"  # <-- Add this line
        )

        corners_enu = [
            [0, 0],  # Bottom-Left
            [1000, 0],  # Bottom-Right
            [1000, 1000],  # Top-Right
            [0, 1000]  # Top-Left
        ]

        boundary_coords = [self.converter.to_latlon(c[0], c[1]) for c in corners_enu]

        folium.Polygon(
            locations=boundary_coords,
            color="crimson",
            weight=3,
            fill=True,
            fill_color="red",
            fill_opacity=0.06,
            popup="Simulation Boundaries (1000m x 1000m)"
        ).add_to(m)

        for idx, anchor in enumerate(anchors_enu):
            lat, lon = self.converter.to_latlon(anchor[0], anchor[1])
            folium.Marker(
                location=[lat, lon],
                popup=f"Anchor {idx}<br>E: {anchor[0]:.1f}m, N: {anchor[1]:.1f}m",
                icon=folium.Icon(color="blue", icon="signal", prefix="fa"),
            ).add_to(m)

        for target_id, target_group in run_data.groupby("target_id"):
            true_enu = (
                target_group["true_x"].iloc[0],
                target_group["true_y"].iloc[0],
            )
            true_lat, true_lon = self.converter.to_latlon(
                true_enu[0], true_enu[1]
            )

            # Plot True Position (Green Target Dot)
            folium.CircleMarker(
                location=[true_lat, true_lon],
                radius=8,
                color="green",
                fill=True,
                fill_color="green",
                popup=f"True Target {target_id}",
            ).add_to(m)

            # Plot estimates for each unique solver evaluated
            for _, row in target_group.iterrows():
                est_lat, est_lon = self.converter.to_latlon(
                    row["est_x"], row["est_y"]
                )
                solver_name = row["solver"]
                error_m = row["error"]

                color = "orange" if solver_name == "vanilla" else "purple"

                # Plot Estimated Position
                folium.CircleMarker(
                    location=[est_lat, est_lon],
                    radius=5,
                    color=color,
                    fill=True,
                    fill_color=color,
                    popup=f"Est: {solver_name}<br>Error: {error_m:.2f}m",
                ).add_to(m)

                folium.PolyLine(
                    locations=[[true_lat, true_lon], [est_lat, est_lon]],
                    color=color,
                    weight=2,
                    dash_array="5, 5",
                    tooltip=f"{solver_name} displacement: {error_m:.2f}m",
                ).add_to(m)

        m.save(output_filename)
        print(f"-> Map layer successfully generated with boundaries: {output_filename}")