# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from pathlib import Path
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# %%
data_path = Path("data")
N_DISTRICTS = 14
DISTRICT_COLORS = {
    -1: "#000000",
    0: "#e6194B",
    1: "#3cb44b",
    2: "#ffe119",
    3: "#4363d8",
    4: "#f58231",
    5: "#911eb4",
    6: "#42d4f4",
    7: "#f032e6",
    8: "#bfef45",
    9: "#fabed4",
    10: "#469990",
    11: "#dcbeff",
    12: "#9A6324",
    13: "#fffac8",
}
# %%
Data = pd.read_csv(data_path / "Blocks.csv", index_col="GEOID")
# %%
Blocks = Data[["Population", "Latitude", "Longitude"]]
Blocks = Blocks.assign(County=[str(x)[2:5] for x in Blocks.index])
# %%
X = Blocks[["Latitude", "Longitude"]].values
sample_weights = Blocks["Population"].values
# %%
kmeans = KMeans(
    n_clusters=N_DISTRICTS,
    init="k-means++",
    verbose=1,
    max_iter=10000,
    tol=1 * 10**-5,
).fit(X, sample_weight=sample_weights)
# %%
Blocks = Blocks.assign(Cluster=kmeans.labels_)


# %%
def plot_blocks_clusters(Blocks):
    # Scatter plot of latitude and longitude colored by clusters using DISTRICT_COLORS
    plt.figure(figsize=(10, 8))
    total_population = Blocks["Population"].sum()

    for cluster in set(Blocks["Cluster"]):
        cluster_data = Blocks[Blocks["Cluster"] == cluster]
        color = DISTRICT_COLORS.get(
            cluster, "#808080"
        )  # Default to gray if color not defined
        plt.scatter(
            cluster_data["Longitude"],
            cluster_data["Latitude"],
            label=f"Cluster {cluster}",
            c=color,
        )

        # Calculate and print the percentage of total population for each cluster
        cluster_population = cluster_data["Population"].sum()
        percentage = (cluster_population / total_population) * 100
        print(
            f"Cluster {cluster}: Population = {cluster_population}, Percentage = {percentage:.2f}%"
        )

    # Set labels and title
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("K-Means Clustering of Blocks")
    plt.legend()
    plt.show()


# Call the function with your Blocks DataFrame
plot_blocks_clusters(Blocks)

# %%
