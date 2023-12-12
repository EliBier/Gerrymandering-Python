# %%
from random import random
import shelve
import threading
import multiprocessing
from tkinter.tix import MAX
from pathlib import Path
import numpy as np
import networkx as nx
import pandas as pd
import geopandas as gpd
import random
import matplotlib.pyplot as plt
from matplotlib import pylab
import pickle

# %%
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

LOGISTIC_K = 0.75
LOGISTIC_MIDPOINT = 700000
LOGISTIC_SCALE_FACTOR = 100000
LOGISTIC_MIDPOINT /= LOGISTIC_SCALE_FACTOR
# %%
Blocks2020_pd = pd.read_csv(data_path / "Blocks2020.csv")
Blocks2020 = gpd.GeoDataFrame(
    Blocks2020_pd, geometry=gpd.GeoSeries.from_wkt(Blocks2020_pd["geometry"])
)
Counties = pd.read_csv(data_path / "Counties.csv")
CountiesAdjMat = np.genfromtxt(data_path / "CountiesAdjMat.csv", delimiter=",")
# %%
TOTAL_POPULATION = Counties["Population"].sum()
POPULATION_PER_DISTRICT = round(TOTAL_POPULATION / N_DISTRICTS)


# %%
def init_nc_graph_blocks2020(Blocks2020_df):
    # Create a graph
    g = nx.Graph()

    # Iterate through the rows of the DataFrame
    for index, row in Blocks2020_df.iterrows():
        block_id = row["GEOID"]
        adjacent_blocks_str = row["ADJ_GEOIDS"]

        if adjacent_blocks_str:
            adjacent_blocks = [int(x.strip()) for x in adjacent_blocks_str.split(",")]

            for adj in adjacent_blocks:
                g.add_edge(block_id, adj)

    # Initialize district attributes
    init_district_attr = {}
    for index, row in Blocks2020_df.iterrows():
        block_id = row["GEOID"]
        init_district_attr[block_id] = {}
        init_district_attr[block_id]["district"] = -1
        init_district_attr[block_id]["population"] = int(row["DECENNIA_1"])
        init_district_attr[block_id]["latitude"] = float(row["INTPTLAT"])
        init_district_attr[block_id]["longitude"] = float(row["INTPTLON"])
        init_district_attr[block_id]["county"] = row["DECENNIALP"].split(", ")[-2]

    nx.set_node_attributes(g, init_district_attr)

    return g


# %%
g = init_nc_graph_blocks2020(Blocks2020)


# %%
def reset_district_assignments(graph):
    """
    Reset the district assignments for each block in the graph to -1.

    Parameters:
    - graph (networkx.Graph): The graph containing current district assignments.

    Returns:
    - None (modifies the graph in-place).
    """
    for block in graph.nodes:
        graph.nodes[block]["district"] = -1


# %%
# Function to find the closest district for a block
def find_closest_district(block_id, district_seeds, g):
    block_coords = (g.nodes[block_id]["latitude"], g.nodes[block_id]["longitude"])

    # Initialize variables to store the closest seed and its distance
    closest_seed = None
    closest_distance = float("inf")

    for seed in district_seeds:
        seed_coords = (g.nodes[seed]["latitude"], g.nodes[seed]["longitude"])
        distance = np.linalg.norm(np.array(block_coords) - seed_coords)

        # If this seed is closer than the current closest seed, update the closest values
        if distance < closest_distance:
            closest_seed = seed
            closest_distance = distance

    return g.nodes[closest_seed]["district"]


# Function to randomly select a block from each county as a district seed
def select_district_seeds(g, num_seeds=14):
    district_seeds = []
    counties = set(g.nodes[block]["county"] for block in g.nodes)
    district_counter = 0  # Initialize a counter for district creation order

    for county in counties:
        blocks_in_county = [
            block for block in g.nodes if g.nodes[block]["county"] == county
        ]
        seed = random.choice(blocks_in_county)
        district_seeds.append(seed)
        g.nodes[seed][
            "district"
        ] = district_counter  # Assign the creation order as the district ID
        district_counter += 1  # Increment the district creation order

        # If we've reached the maximum number of districts, break the loop
        if district_counter >= num_seeds:
            break

    return district_seeds


# Function to assign blocks to the closest district
def assign_blocks_to_districts(g, district_seeds):
    for block in g.nodes:
        if g.nodes[block]["district"] == -1:
            closest_district = find_closest_district(block, district_seeds, g)
            g.nodes[block]["district"] = closest_district


# %%
def compute_population_imbalance(district_populations):
    # Calculate the average population per district
    avg_population = sum(district_populations) / len(district_populations)

    # Calculate the maximum difference in population between districts
    max_difference = max(district_populations) - min(district_populations)

    # Define a weight for the population balance factor (you can adjust this weight as needed)
    weight_population_balance = 0.9  # Adjust this value as needed

    # Calculate the imbalance score as a weighted sum of max difference and population balance factor
    imbalance_score = weight_population_balance * max_difference + (
        1 - weight_population_balance
    ) * abs(avg_population - max_difference)

    return imbalance_score


# %%
# Function to compute the total population of each district
def compute_district_populations(graph, num_districts):
    district_populations = [0] * num_districts
    for block in graph.nodes:
        district = graph.nodes[block]["district"]
        population = graph.nodes[block]["population"]
        district_populations[district] += population
    return district_populations


# Function to swap two block assignments
def swap_blocks(g, block1, block2):
    district1 = g.nodes[block1]["district"]
    district2 = g.nodes[block2]["district"]
    g.nodes[block1]["district"] = district2
    g.nodes[block2]["district"] = district1


# %%
def simulated_annealing(
    graph,
    num_districts,
    max_iterations=1000,
    min_batch_size=100,
):
    score_threshold_1 = 100000
    score_threshold_2 = 300000
    Score_threshold_3 = 1000000
    current_district_populations = compute_district_populations(graph, num_districts)
    initial_imbalance = compute_population_imbalance(current_district_populations)

    for iteration in range(max_iterations):
        # Select a random district with a population greater than the average
        valid_districts = [
            district
            for district in range(num_districts)
            if current_district_populations[district] > POPULATION_PER_DISTRICT
        ]
        if not valid_districts:
            print("BROKEN1")
            break

        source_district = random.choice(valid_districts)

        # Find blocks on the boundary of the source district
        boundary_blocks = []
        for block in graph.nodes:
            if graph.nodes[block]["district"] == source_district:
                for neighbor in graph.neighbors(block):
                    if graph.nodes[neighbor]["district"] != source_district:
                        boundary_blocks.append(block)
                        break

        if not boundary_blocks:
            print("BROKEN2")
            break

        # Determine batch size based on the current imbalance score
        score = initial_imbalance
        if score > score_threshold_2:
            swap_batch_size = min_batch_size * 100
        elif score > score_threshold_1:
            swap_batch_size = min_batch_size * 10
        else:
            swap_batch_size = min_batch_size * 100

        # Check if reassigning blocks to districts is more beneficial than swapping
        for _ in range(swap_batch_size):
            # Randomly select a boundary block
            block_to_swap = random.choice(boundary_blocks)

            # Find the neighboring districts of the selected block
            neighbor_districts = set(
                graph.nodes[neighbor]["district"]
                for neighbor in graph.neighbors(block_to_swap)
            )

            # Calculate the new population balance if the block is reassigned
            new_district_populations = current_district_populations.copy()
            new_district_populations[source_district] -= graph.nodes[block_to_swap][
                "population"
            ]
            for neighbor_district in neighbor_districts:
                new_district_populations[neighbor_district] += graph.nodes[
                    block_to_swap
                ]["population"]

            new_imbalance = compute_population_imbalance(new_district_populations)

            # If reassigning the block reduces imbalance more, then reassign it
            if new_imbalance < initial_imbalance:
                initial_imbalance = new_imbalance
                current_district_populations = new_district_populations
                graph.nodes[block_to_swap]["district"] = random.choice(
                    list(neighbor_districts)
                )

        if iteration % 50 == 0:
            print(f"Iterations Completed: {iteration}")
            print_and_plot_district_populations(graph, num_districts, DISTRICT_COLORS)

    print(f"Simulated Annealing completed in {iteration} iterations.")


# %%
def create_pos_from_graph(graph):
    """
    Create a 'pos' array based on the positions of nodes in a NetworkX graph.

    Parameters:
        graph (networkx.Graph): The graph with node positions.

    Returns:
        numpy.ndarray: An array of node positions (geoid, latitude, longitude).
    """
    pos = {}
    for node in graph.nodes:
        if "latitude" in graph.nodes[node] and "longitude" in graph.nodes[node]:
            pos[node] = {
                "geoid": node,
                "latitude": graph.nodes[node]["latitude"],
                "longitude": graph.nodes[node]["longitude"],
            }

    pos_array = np.array([pos[node] for node in graph.nodes if node in pos])

    return pos_array


# def draw_districts(graph, attribute="district", alpha=1):
#     """
#     Draw district boundaries using alpha shapes.

#     Parameters:
#         graph (networkx.Graph): The graph containing block assignments and attributes.
#         attribute (str): The attribute representing district assignments.
#         alpha (float): The alpha parameter for alpha shape computation.

#     Returns:
#         matplotlib.pyplot.Axes: The plot with district boundaries.
#     """
#     # Create a 'pos' array from the NetworkX graph
#     pos = create_pos_from_graph(graph)

#     # Get the district assignments from the graph
#     districts = nx.get_node_attributes(graph, attribute)

#     # Create a dictionary to store points for each district
#     district_points = {}
#     for node, district in districts.items():
#         if district not in district_points:
#             district_points[district] = []
#         # Use the GEOID to find the position in 'pos'
#         matching_pos = next((item for item in pos if item["geoid"] == node), None)
#         if matching_pos:
#             district_points[district].append([matching_pos["latitude"], matching_pos["longitude"]])

#     # Create a plot
#     fig, ax = plt.subplots(figsize=(10, 10))

#     # Plot each district boundary using alpha shape
#     for district, points in district_points.items():
#         if len(points) >= 3:
#             # Compute the alpha shape for the district
#             alpha_points, alpha_edges = alphashape(np.array(points), alpha)

#             # Plot the alpha shape boundary
#             ax.plot(alpha_points[:, 0], alpha_points[:, 1], label=f'District {district}')

#     ax.set_aspect('equal')
#     ax.set_xlabel('Longitude')
#     ax.set_ylabel('Latitude')
#     ax.legend()

#     return ax


# %%
def assign_colors_to_dataframe(graph, dataframe, colors_dict):
    for node in graph.nodes(data=True):
        node_id, district = node
        if node_id in dataframe["GEOID"].values:
            # Find the index in the DataFrame where 'GEOID' matches the node ID
            index = dataframe.index[dataframe["GEOID"] == node_id].tolist()[0]
            # Get the color for the district from the colors_dict
            color = colors_dict[district["district"]]
            # Update the 'color' column with the assigned color
            dataframe.at[index, "color"] = color


# %%
def plot_geopandas_dataframe(dataframe, geometry_column, color_column):
    """
    Plots a GeoDataFrame using the specified geometry and color columns.

    Parameters:
    - dataframe: The GeoDataFrame to plot.
    - geometry_column: The name of the column containing geometry (geometries must be provided as GeoJSON or WKT).
    - color_column: The name of the column containing colors.

    Returns:
    - None (displays the plot).
    """
    if not isinstance(dataframe, gpd.GeoDataFrame):
        raise ValueError("The input dataframe must be a GeoDataFrame.")

    if geometry_column not in dataframe.columns:
        raise ValueError(f"'{geometry_column}' not found in the dataframe columns.")

    if color_column not in dataframe.columns:
        raise ValueError(f"'{color_column}' not found in the dataframe columns.")

    # Create a custom color array using the values from the color_column
    colors = dataframe[color_column]

    # Plot using the custom colors
    ax = dataframe.plot(legend=True, legend_kwds={"label": "Legend"}, color=colors)

    plt.title("Geospatial Plot")
    plt.show()


# %%
def plot_top_districts(dataframe, geometry_column, color_column, n=3):
    """
    Plots the top N districts with the greatest population in a GeoDataFrame using the color from the color column.

    Parameters:
    - dataframe: The GeoDataFrame to plot.
    - geometry_column: The name of the column containing geometry (geometries must be provided as GeoJSON or WKT).
    - color_column: The name of the column containing colors.
    - n: The number of top districts to plot (default is 3).

    Returns:
    - None (displays the plot).
    """
    if not isinstance(dataframe, gpd.GeoDataFrame):
        raise ValueError("The input dataframe must be a GeoDataFrame.")

    if geometry_column not in dataframe.columns:
        raise ValueError(f"'{geometry_column}' not found in the dataframe columns.")

    if color_column not in dataframe.columns:
        raise ValueError(f"'{color_column}' not found in the dataframe columns.")

    # Sort the dataframe by the population column in descending order
    sorted_dataframe = dataframe.sort_values(by="population", ascending=False)

    # Select the top N districts
    top_districts = sorted_dataframe.head(n)

    # Extract the colors from the color_column as a list
    colors = top_districts[color_column].tolist()

    # Plot the selected districts using the extracted colors
    top_districts.plot(
        column=None,  # No specific column for coloring
        facecolor=colors,  # Use the extracted colors
        legend=False,  # Turn off the legend as colors are used directly
    )
    plt.title(f"Top {n} Districts by Population")
    plt.show()


# %%
# Define a function to calculate and print the population of each district as a percentage of the total population
def print_and_plot_district_populations(
    g, num_districts, DISTRICT_COLORS, previous_population_stats=None
):
    district_populations = [0] * num_districts
    total_population = 0  # Initialize total population

    # Create a dictionary to keep track of the population changes for each district
    population_changes = {}

    for block in g.nodes:
        district = g.nodes[block]["district"]
        population = g.nodes[block]["population"]

        # Check if the assigned district is within the valid range
        if 0 <= district < num_districts:
            district_populations[district] += population
            total_population += population

        else:
            print(f"Warning: Block {block} has an invalid district ID {district}")

    # Create a list of district indices sorted by population
    sorted_district_indices = sorted(
        range(num_districts), key=lambda x: -district_populations[x]
    )

    # Calculate and print the population changes
    for i, district_id in enumerate(sorted_district_indices):
        percentage = (district_populations[district_id] / total_population) * 100

        # Check if the population increased, decreased, or stayed the same
        change_symbol = "="
        if previous_population_stats:
            prev_population = previous_population_stats.get(district_id, None)
            if prev_population is not None:
                if district_populations[district_id] > prev_population:
                    change_symbol = "+"
                elif district_populations[district_id] < prev_population:
                    change_symbol = "-"
            # Update the previous population statistics dictionary
            previous_population_stats[district_id] = district_populations[district_id]

        print(
            f"District {district_id}: Population = {district_populations[district_id]}, Percentage of Total = {percentage:.2f}%, Change: {change_symbol}"
        )

        # Update the population change dictionary
        population_changes[district_id] = change_symbol

    # Create a bar graph with custom colors
    colors = [DISTRICT_COLORS[i] for i in sorted_district_indices]
    percentages = [
        (district_populations[i] / total_population) * 100
        for i in sorted_district_indices
    ]
    plt.bar(
        range(num_districts),
        percentages,
        color=colors,
    )
    plt.xlabel("District")
    plt.ylabel("Percentage of Total Population")
    plt.title(
        "District Populations as Percentages of Total Population (Descending Order)"
    )
    plt.show()

    return previous_population_stats


# %%
# Function to save the districting to a pickle file
def save_districting_to_pickle(g, filename):
    districting = {}
    for block in g.nodes:
        districting[block] = g.nodes[block]["district"]

    with open(filename, "wb") as f:
        pickle.dump(districting, f)
    print(f"Districting saved to {filename}")


# %%
# Step 1: Select 14 random district seeds (one in each county)
district_seeds = select_district_seeds(g, num_seeds=14)
# %%
# Step 2: Assign each block to the closest district
assign_blocks_to_districts(g, district_seeds)
# %%
# Step 3: Draw districts
assign_colors_to_dataframe(g, Blocks2020, DISTRICT_COLORS)
plot_geopandas_dataframe(Blocks2020, "geometry", "color")
print_and_plot_district_populations(g, 14, DISTRICT_COLORS)
# %%
# Step 4: Anneal To Get Correct Population
# Define a threshold for population balance
population_balance_threshold = 1000  # Adjust as needed

# Initialize variables
iterations = 0

while True:
    # Run simulated annealing
    simulated_annealing(g, 14, 100)  # You may need to pass other parameters as well

    # Calculate the current population imbalance
    current_district_populations = compute_district_populations(g, 14)
    max_difference = max(current_district_populations) - min(
        current_district_populations
    )

    # Check if the population balance threshold is met
    if max_difference <= population_balance_threshold:
        break  # Exit the loop if the threshold is met

    iterations += 1

    # Optionally, you can add a condition to limit the number of iterations to avoid infinite loops
    if iterations >= 1000:
        print("Reached the maximum number of iterations.")
        break

print(f"Simulated Annealing completed in {iterations} iterations.")
# %%
# Step 5: Draw districts
assign_colors_to_dataframe(g, Blocks2020, DISTRICT_COLORS)
plot_geopandas_dataframe(Blocks2020, "geometry", "color")
# %%
assign_colors_to_dataframe(g, Blocks2020, DISTRICT_COLORS)
plot_top_districts(Blocks2020, "geometry", "color")
# %%
