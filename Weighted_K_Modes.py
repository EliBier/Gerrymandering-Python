# %%
from pathlib import Path
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

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
Data = pd.read_csv(data_path / "Blocks.csv")
# %%
Blocks = Data[["GEOID", "Population", "Latitude", "Longitude"]]
Blocks["GEOID"] = Blocks["GEOID"].astype(str)
Blocks["County"] = Blocks["GEOID"].str[2:5]
# %%
Clusters = pd.DataFrame(columns=["Latitude", "Longitude"])
Scaling_Factors = {key: 1 for key in range(N_DISTRICTS)}


# %%
def Distance_Haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in kilometers

    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Calculate differences in latitude and longitude
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c

    return distance


# %%
def Distance_Euclidean(lat1, lon1, lat2, lon2):
    # Euclidean distance formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    distance = np.sqrt(dlat**2 + dlon**2)
    return distance


# %%
def select_seed_blocks(df, n):
    # Ensure 'GEOID' column exists and is of type str
    if "GEOID" not in df.columns or df["GEOID"].dtype != "O":
        raise ValueError("The DataFrame must have a 'GEOID' column of type string.")

    # Randomly select 14 unique counties
    selected_counties = random.sample(list(Blocks["County"].unique()), n)

    # Select one random row from each of the selected 14 counties
    selected_rows = (
        df[df["GEOID"].str[2:5].isin(selected_counties)]
        .groupby("County", group_keys=False)
        .apply(lambda x: x.sample(1))
    )

    selected_rows = selected_rows.reset_index(drop=True)
    selected_coordinates = selected_rows[["Latitude", "Longitude"]]
    return selected_coordinates


# %%
def Calculate_Scaling_Factors(df, factors_dict, alpha, beta):
    old_factors = factors_dict
    # Get the populations of each of the clusters
    populations_by_cluster = df.groupby("Cluster").sum()
    populations_by_cluster = populations_by_cluster[["Population"]]

    # For each cluster raise its population by alpha and store it
    for index, population in populations_by_cluster.iterrows():
        factors_dict[index] = population["Population"] ** alpha

    total_sum = sum(factors_dict.values())

    # Divide each of the cluster populations by the sum of the scaling factors
    for key, value in factors_dict.items():
        factors_dict[key] = value / total_sum

    # The scaling_factors should now sum to one (This might be slightly off because of floating point operations)
    # print(sum(factors_dict.values()))

    # apply time-averaging on the scaling factor
    for key, value in factors_dict.items():
        factors_dict[key] = (1 - beta) * old_factors[key] + (beta) * (value + 0.1)

    # The scaling factors should still sum to about one
    # print(sum(factors_dict.values()))

    return factors_dict


# %%
def reevaluate_centers(Blocks, Clusters):
    new_Clusters = pd.DataFrame(columns=["Latitude", "Longitude"])
    mean_coordinates = (
        Blocks.groupby("Cluster")[["Latitude", "Longitude"]].mean().reset_index()
    )
    new_Clusters = mean_coordinates[["Latitude", "Longitude"]]
    return new_Clusters


# %%
def Cluster_Points(
    Blocks,
    Clusters,
    Scaling_Factors,
    N_DISTRICTS,
    distance_type="euclidean",
    alpha=0.5,
    beta=0.1,
):
    # If clusters are empty (meaning that initial points haven't been chosen) choose initial points
    if Clusters.empty:
        Clusters = pd.concat([Clusters, select_seed_blocks(Blocks, N_DISTRICTS)])

    ################################################################################
    # FIRST CREATE CLUSTERS USING K-MODES ##########################################
    ################################################################################

    for block_index, block in tqdm(
        Blocks.iterrows(), total=len(Blocks), desc="Clustering Progress"
    ):
        distance = {}
        for cluster_index, cluster in Clusters.iterrows():
            if distance_type == "haversine":
                distance[cluster_index] = (
                    Distance_Haversine(
                        block["Latitude"],
                        block["Longitude"],
                        Clusters.at[cluster_index, "Latitude"],
                        Clusters.at[cluster_index, "Longitude"],
                    )
                    * Scaling_Factors[cluster_index]
                )
            elif distance_type == "euclidean":
                distance[cluster_index] = (
                    Distance_Euclidean(
                        block["Latitude"],
                        block["Longitude"],
                        Clusters.at[cluster_index, "Latitude"],
                        Clusters.at[cluster_index, "Longitude"],
                    )
                    * Scaling_Factors[cluster_index]
                )
        min_cluster, min_distance = min(distance.items(), key=lambda x: x[1])
        Blocks.at[block_index, "Cluster"] = min_cluster

    Blocks["Cluster"] = Blocks["Cluster"].astype(int)

    ################################################################################
    # SECONDLY CALCULATE SCALING FACTOR FOR EACH CLUSTER ###########################
    ################################################################################
    Scaling_Factors = Calculate_Scaling_Factors(Blocks, Scaling_Factors, alpha, beta)

    return Blocks, Clusters, Scaling_Factors


# %%
def calculate_distance(block, cluster, scaling_factor, distance_type="euclidean"):
    if distance_type == "haversine":
        return (
            Distance_Haversine(
                block["Latitude"],
                block["Longitude"],
                cluster["Latitude"],
                cluster["Longitude"],
            )
            * scaling_factor
        )
    elif distance_type == "euclidean":
        return (
            Distance_Euclidean(
                block["Latitude"],
                block["Longitude"],
                cluster["Latitude"],
                cluster["Longitude"],
            )
            * scaling_factor
        )


def cluster_block(block, clusters, scaling_factors, distance_type="euclidean"):
    distance_dict = {}
    for cluster_index, cluster in clusters.iterrows():
        scaling_factor = scaling_factors[cluster_index]
        distance_dict[cluster_index] = calculate_distance(
            block, cluster, scaling_factor, distance_type
        )
    min_cluster, min_distance = min(distance_dict.items(), key=lambda x: x[1])
    return min_cluster


def parallel_cluster_points(args):
    block, clusters, scaling_factors, distance_type = args
    cluster_index = cluster_block(block, clusters, scaling_factors, distance_type)
    return cluster_index


def Cluster_Points_Parallel(
    Blocks,
    Clusters,
    Scaling_Factors,
    N_DISTRICTS,
    distance_type="euclidean",
    alpha=0.5,
    beta=0.1,
):
    if Clusters.empty:
        Clusters = pd.concat([Clusters, select_seed_blocks(Blocks, N_DISTRICTS)])

    with tqdm(total=len(Blocks), desc="Clustering Progress") as pbar:

        def update_pbar(*_):
            pbar.update()

        with ThreadPoolExecutor() as executor:
            args_list = [
                (Blocks.iloc[i], Clusters, Scaling_Factors, distance_type)
                for i in range(len(Blocks))
            ]
            futures = [
                executor.submit(parallel_cluster_points, args) for args in args_list
            ]

            # Add a callback to update the progress bar for each completed task
            for future in futures:
                future.add_done_callback(update_pbar)

            # Wait for all tasks to complete
            results = [future.result() for future in futures]

    Blocks["Cluster"] = results
    Blocks["Cluster"] = Blocks["Cluster"].astype(int)

    Scaling_Factors = Calculate_Scaling_Factors(Blocks, Scaling_Factors, alpha, beta)

    return Blocks, Clusters, Scaling_Factors


# %%
def plot_clusters(blocks, clusters):
    plt.figure(figsize=(10, 6))

    total_population_all_clusters = blocks["Population"].sum()
    legend_labels = []

    for cluster, color in DISTRICT_COLORS.items():
        cluster_points = blocks[blocks["Cluster"] == cluster]
        plt.scatter(
            cluster_points["Longitude"],
            cluster_points["Latitude"],
            c=color,
            label=f"Cluster {cluster}",
        )

        total_population_cluster = cluster_points["Population"].sum()
        percentage_population = (
            total_population_cluster / total_population_all_clusters
        ) * 100

        legend_labels.append(
            f"Cluster {cluster}: {total_population_cluster} population ({percentage_population:.2f}%)"
        )

    # Plot markers at the center of each cluster
    for cluster_index, cluster in clusters.iterrows():
        plt.scatter(
            cluster["Longitude"],
            cluster["Latitude"],
            marker="x",
            s=10,
            c="black",
            label=f"Center of Cluster {cluster_index}",
            alpha=0.8,
        )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Clustered Points with Cluster Centers")
    plt.legend(labels=legend_labels, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.show()


# %%
def find_centers(
    Blocks,
    Clusters,
    Scaling_Factors,
    N_DISTRICTS=14,
    max_iterations=1000,
    alpha=0.5,
    beta=0.1,
):
    runs = 0
    while runs < max_iterations:
        # Keep track of previous clusters (will be used for convergence check later)
        old_clusters = Clusters.copy()

        # assign points to cluster
        Blocks, Clusters, Scaling_Factors = Cluster_Points(
            Blocks, Clusters, Scaling_Factors, N_DISTRICTS
        )

        # Find new center
        Clusters = reevaluate_centers(Blocks, Clusters)

        # Increment runs to end loop
        plot_clusters(Blocks, Clusters)
        print("Current Run:" + str(runs))
        runs += 1
    return Blocks, Clusters, Scaling_Factors


# %%
def find_centers_test(
    Blocks,
    Clusters,
    Scaling_Factors,
    N_DISTRICTS=14,
    max_iterations=1000,
    alpha=0.5,
    beta=0.1,
):
    runs = 0
    while runs < max_iterations:
        # Keep track of previous clusters (will be used for convergence check later)
        old_clusters = Clusters.copy()

        # assign points to cluster
        Blocks, Clusters, Scaling_Factors = Cluster_Points(
            Blocks, Clusters, Scaling_Factors, N_DISTRICTS
        )

        # Find new center
        # Clusters = reevaluate_centers(Blocks, Clusters)

        # Increment runs to end loop
        plot_clusters(Blocks, Clusters)
        print("Current Run:" + str(runs))
        runs += 1
    return Blocks, Clusters, Scaling_Factors


# %%
def Find_Centers_Parallel(
    Blocks,
    Clusters,
    Scaling_Factors,
    N_DISTRICTS=14,
    max_iterations=1000,
    alpha=0.5,
    beta=0.1,
):
    runs = 0
    while runs < max_iterations:
        # Keep track of previous clusters (will be used for convergence check later)
        old_clusters = Clusters.copy()

        # assign points to cluster
        Blocks, Clusters, Scaling_Factors = Cluster_Points_Parallel(
            Blocks, Clusters, Scaling_Factors, N_DISTRICTS
        )

        # Find new center
        Clusters = reevaluate_centers(Blocks, Clusters)

        # Increment runs to end loop
        plot_clusters(Blocks, Clusters)
        print("Current Run:" + str(runs))
        runs += 1
    return Blocks, Clusters, Scaling_Factors


# %%
Small_Blocks = Blocks.copy()
# %%
Small_Blocks = Small_Blocks.head(10000)
# %%
Small_Blocks, Clusters, Scaling_Factors = Cluster_Points(
    Small_Blocks, Clusters, Scaling_Factors, 2
)
# %%
Small_Blocks, Clusters, Scaling_Factors = find_centers(
    Small_Blocks, Clusters, Scaling_Factors, 14, 50, 0.1, 0.5
)
# %%
Blocks, Clusters, Scaling_Factors = find_centers(
    Blocks, Clusters, Scaling_Factors, 14, 50, 3, 1
)
# %%
Blocks, Clusters, Scaling_Factors = find_centers_test(
    Blocks, Clusters, Scaling_Factors, 14, 50, 0.5, 1
)

# %%
