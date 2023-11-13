
# %%
from __future__ import division, print_function
from pathlib import Path
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import pandas as pd
import random
#%%
data_path = Path("data")
N_DISTRICTS = 14
MIN_DISTRICT_POP = 400000
MAX_DISTRICT_POP = 1000000
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
    14: "#800000",
}
#%%
Blocks = gpd.read_file(data_path / "Blocks.geojson", index = False)
#%%
def extract_data(geo_df, population_col='Population', lat_col='Latitude', lon_col='Longitude', id_col='GEOID'):
    """
    Extracts data from a GeoDataFrame and formats it for the weighted_kmeans function.

    Parameters:
    - geo_df: GeoDataFrame containing the data.
    - population_col: Column name for population data.
    - lat_col: Column name for latitude data.
    - lon_col: Column name for longitude data.
    - id_col: Column name for the identifier (GEOID).

    Returns:
    - data: List of data points, each represented as a list of coordinates [population, latitude, longitude, GEOID].
    """
    data = []
    for index, row in geo_df.iterrows():
        population = row[population_col]
        latitude = row[lat_col]
        longitude = row[lon_col]
        geoid = row[id_col]
        data.append([population, latitude, longitude, geoid])

    return data

def euclidean(a, b):
    return np.linalg.norm(np.asarray(a) - np.asarray(b))

def has_converged(mu, old_mu, max_diff):
    if old_mu is not None:
        diff = 0
        for i in range(len(mu)):
            diff += euclidean(mu[i], old_mu[i])
        diff /= len(mu)
        return diff < max_diff
    return False

def cluster_points(data, mu, alpha):
    clusters = {i: [] for i in range(len(mu))}
    counts_per_cluster = [0 for _ in range(len(mu))]

    for index, x in enumerate(data):
        bestmukey = min([(i, alpha * euclidean(x, mu[i])) for i in range(len(mu))],
                        key=lambda t: t[1])[0]
        clusters[bestmukey].append(x)
        counts_per_cluster[bestmukey] += 1

    return clusters, counts_per_cluster

def reevaluate_centers(data, cluster_indices, k):
    new_mu = []
    for cluster_index in range(k):
        cluster_points = cluster_indices[cluster_index]
        new_mu.append(np.mean(cluster_points, axis=0))
    return new_mu

def update_scaling_factors(mu, alpha, beta, scaling_factor):
    scaling_factors = np.asarray([len(cluster)**alpha for cluster in mu])
    scaling_factors /= np.sum(scaling_factors)
    scaling_factor = (1 - beta) * scaling_factor + beta * scaling_factors
    return scaling_factor
# %%

def weighted_kmeans(data, k, alpha=0, beta=0, max_runs=200, max_diff=0.001, verbose=False, mu=None, dist=euclidean):
    if mu is None:
        mu = random.sample(data, k)
    old_mu = None
    scaling_factor = np.ones((k)) / k
    counts_per_cluster = [0 for _ in range(k)]
    runs = 0
    while not has_converged(mu, old_mu, max_diff) and runs < max_runs:
        if verbose:
            print('\nRun: ' + str(runs) + ', alpha: ' + str(alpha) + ', beta: ' + str(beta))
        old_mu = mu
        cluster_indices, counts_per_cluster = cluster_points(data, mu, alpha)
        mu = reevaluate_centers(data, cluster_indices, k)
        scaling_factor = update_scaling_factors(mu, alpha, beta, scaling_factor)
        runs += 1
    return cluster_indices
#%%
def plot_districts(result_clusters, geo_df, title='Clustered Districts'):
    """
    Plots the clustered districts on a map with a legend.

    Parameters:
    - result_clusters: Dictionary containing cluster indices for each data point.
    - geo_df: GeoDataFrame containing the geometries for each data point.
    - title: Title for the plot (default is 'Clustered Districts').
    """
    # Create a copy of the GeoDataFrame to avoid modifying the original
    plot_df = geo_df.copy()

    # Add a new column 'Cluster' to store the cluster indices
    plot_df['Cluster'] = [result_clusters[i] for i in range(len(plot_df))]

    # Plot the districts using GeoDataFrame.plot() with 'Cluster' as color
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_df.plot(column='Cluster', cmap=DISTRICT_COLORS, linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)

    # Create a legend
    handles = []
    labels = []
    for cluster_id in np.unique(result_clusters.values()):
        cluster_data = plot_df[plot_df['Cluster'] == cluster_id]
        cluster_population = cluster_data['Population'].sum()

        # Add legend entry for each cluster with color, ID, and population
        handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=ax.get_cmap(DISTRICT_COLORS)(cluster_id), markersize=10)
        label = f'Cluster {cluster_id}\nPopulation: {cluster_population}'
        handles.append(handle)
        labels.append(label)

    ax.legend(handles, labels, title='Cluster Legend', loc='upper left', bbox_to_anchor=(1, 1))

    # Customize the plot
    ax.set_title(title, fontdict={'fontsize': '15', 'fontweight': '3'})
    ax.set_axis_off()

    # Show the plot
    plt.show()
    plt.close()

#%%
data = extract_data(Blocks)
result_clusters = weighted_kmeans(data, k=14, alpha=0.5, beta=0.1, max_runs=100, verbose = True)
#%%
plot_districts(result_clusters,Blocks)
# %%
