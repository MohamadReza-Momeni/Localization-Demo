import pandas as pd
df = pd.read_csv("results.csv")
df["dist_to_edge"] = df[["true_x", "true_y"]].apply(
    lambda row: min(row["true_x"], 1000 - row["true_x"], row["true_y"], 1000 - row["true_y"]),
    axis=1
)
print(df.groupby(pd.cut(df["dist_to_edge"], bins=5))["error"].mean())