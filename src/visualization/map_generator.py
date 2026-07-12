import os
import json
import folium
import numpy as np
import pandas as pd
from src.geo.enu import ENUConverter


class LocalizationVisualizer:
    # Fixed, distinct color per solver so every algorithm is visually distinguishable on the map
    SOLVER_COLORS = {
        "vanilla": "orange",
        "weighted": "purple",
        "ipopt": "blue",
        "weighted_ipopt": "darkred",
        "particle_filter": "darkgreen",
    }
    FALLBACK_COLORS = ["cadetblue", "black", "pink", "gray", "lightred"]

    # ADDED: x_range and y_range parameters
    def __init__(self, lat0, lon0, x_range=(0, 1000), y_range=(0, 1000), results_path="results.csv"):
        self.converter = ENUConverter(lat0, lon0)
        self.results_path = results_path
        self.x_range = x_range
        self.y_range = y_range

        if not os.path.exists(results_path):
            raise FileNotFoundError(f"No results file found at {results_path}. Run experiments first!")
        self.df = pd.read_csv(results_path)

    def _color_for_solver(self, solver_name):
        """Returns a stable color for a solver name, falling back gracefully for unknown solvers."""
        if solver_name in self.SOLVER_COLORS:
            return self.SOLVER_COLORS[solver_name]
        # Deterministic fallback so an unrecognized solver still gets a consistent color across markers
        idx = abs(hash(solver_name)) % len(self.FALLBACK_COLORS)
        return self.FALLBACK_COLORS[idx]

    def generate_run_map(self, run_id=0, output_filename=None):
        run_data = self.df[self.df["run_id"] == run_id]
        if run_data.empty:
            raise ValueError(f"Run ID {run_id} not found in dataset.")

        if output_filename is None:
            output_filename = f"map_run_{run_id}.html"

        anchor_count = run_data["anchor_count"].iloc[0]

        # UPDATED: anchors are no longer regenerated from a seed (anchor generation is
        # now random per run, so a seed can't reproduce it). Instead we read back the
        # exact anchor coordinates that were used and saved into results.csv for this run.
        if "anchors" not in run_data.columns:
            raise ValueError(
                "results.csv has no 'anchors' column. Re-run main.py with the updated "
                "task.py that persists anchor coordinates per run."
            )
        anchors_enu = np.array(json.loads(run_data["anchors"].iloc[0]))

        m = folium.Map(
            location=[self.converter.lat0, self.converter.lon0],
            zoom_start=15,
            control_scale=True,
            tiles="OpenTopoMap"  # <-- Add this line
        )

        # FIXED: previously hardcoded to (0,1000); now respects the actual configured bounds
        x_min, x_max = self.x_range
        y_min, y_max = self.y_range
        corners_enu = [
            [x_min, y_min],  # Bottom-Left
            [x_max, y_min],  # Bottom-Right
            [x_max, y_max],  # Top-Right
            [x_min, y_max]   # Top-Left
        ]

        boundary_coords = [self.converter.to_latlon(c[0], c[1]) for c in corners_enu]

        folium.Polygon(
            locations=boundary_coords,
            color="crimson",
            weight=3,
            fill=True,
            fill_color="red",
            fill_opacity=0.06,
            popup=f"Simulation Boundaries ({x_max - x_min:.0f}m x {y_max - y_min:.0f}m)"
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

                color = self._color_for_solver(solver_name)

                # Plot Estimated Position
                folium.CircleMarker(
                    location=[est_lat, est_lon],
                    radius=5,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.9,
                    popup=f"Est: {solver_name}<br>Error: {error_m:.2f}m",
                ).add_to(m)

                folium.PolyLine(
                    locations=[[true_lat, true_lon], [est_lat, est_lon]],
                    color=color,
                    weight=2,
                    dash_array="5, 5",
                    tooltip=f"{solver_name} displacement: {error_m:.2f}m",
                ).add_to(m)

        self._add_legend(m, run_data["solver"].unique())

        m.save(output_filename)
        print(f"-> Map layer successfully generated with boundaries: {output_filename}")

    def _add_legend(self, m, solver_names):
        """Adds a fixed-position HTML legend mapping each solver present to its marker color."""
        rows = "".join(
            f'<div style="margin:2px 0;">'
            f'<span style="display:inline-block;width:12px;height:12px;'
            f'background:{self._color_for_solver(name)};border-radius:50%;'
            f'margin-right:6px;"></span>{name}</div>'
            for name in sorted(solver_names)
        )
        legend_html = f"""
        <div style="
            position: fixed;
            bottom: 30px; left: 30px; z-index: 9999;
            background: white; padding: 10px 14px;
            border: 2px solid #444; border-radius: 6px;
            font-size: 13px; line-height: 1.3;
            box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
        ">
            <div style="font-weight:bold; margin-bottom:4px;">Solver</div>
            <div style="margin:2px 0;">
                <span style="display:inline-block;width:12px;height:12px;
                background:green;border-radius:50%;margin-right:6px;"></span>True Target
            </div>
            {rows}
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))