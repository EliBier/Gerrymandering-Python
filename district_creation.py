# %%
from random import random
import shelve
import threading
import multiprocessing
from tkinter.tix import MAX
import scipy.io
import csv
import pprint as pp
from collections.abc import Callable, Iterable
from pathlib import Path
from queue import Queue
import numpy as np
import networkx as nx
import pandas as pd
import geopandas as gpd
import random
import time
import matplotlib.pyplot as plt
from matplotlib import pylab
from matplotlib.lines import Line2D

# %%
data_path = Path("data")
N_DISTRICTS = 13
MIN_DISTRICT_POP = 400000
MAX_DISTRICT_POP = 1000000

LOGISTIC_K = 0.75
LOGISTIC_MIDPOINT = 700000
LOGISTIC_SCALE_FACTOR = 100000
LOGISTIC_MIDPOINT /= LOGISTIC_SCALE_FACTOR

DISTRICT_COLORS = {
    -1: "k",
    0: "tab:blue",
    1: "tab:orange",
    2: "tab:green",
    3: "tab:red",
    4: "tab:purple",
    5: "tab:brown",
    6: "tab:pink",
    7: "tab:gray",
    8: "tab:olive",
    9: "tab:cyan",
    10: "yellow",
    11: "magenta",
    12: "maroon",
    13: "beige",
    14: "aquamarine",
}
# %%
Blocks = pd.read_csv(data_path / "Blocks.csv")
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
Blocks.sort_values(by="Longitude", ascending=True, inplace=True)
seed = Blocks.iloc[0]


# %%
def assign_colors_to_dataframe(graph, dataframe, colors_dict):
    for node in graph.nodes(data=True):
        node_id, district = node
        if node_id in dataframe["GEOID"].values:
            # Find the index in the DataFrame where 'GEOID' matches the node ID
            index = dataframe.index[dataframe["GEOID"] == node_id].tolist()[0]
            # Get the color for the district from the colors_dict
            color = colors_dict.get(district, "unknown")
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

    dataframe.plot(
        column=color_column,
        legend=True,
        legend_kwds={"label": "Legend"},
        cmap="viridis",
    )
    plt.title("Geospatial Plot")
    plt.show()


# %%
def save_graph(graph, file_name):
    # initialze Figure
    plt.figure(num=None, figsize=(20, 20), dpi=80)
    plt.axis("off")
    fig = plt.figure(1)
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, pos)
    nx.draw_networkx_edges(graph, pos)
    nx.draw_networkx_labels(graph, pos)

    cut = 1.00
    xmax = cut * max(xx for xx, yy in pos.values())
    ymax = cut * max(yy for xx, yy in pos.values())
    plt.xlim(0, xmax)
    plt.ylim(0, ymax)

    plt.savefig(file_name, bbox_inches="tight")
    pylab.close()
    del fig


def draw_graph(g, data, legend=True):
    fig = plt.figure(figsize=(20, 5))
    ax = fig.add_subplot(111)
    colors = []

    # Create a mapping from ID to district
    district_mapping = nx.get_node_attributes(g, "district")

    # Loop through the DataFrame to get node positions and colors
    node_positions = {}
    for index, row in data.iterrows():
        node_id = row["ID"]
        node_positions[node_id] = (row["Longitude"], row["Latitude"])
        colors.append(DISTRICT_COLORS[district_mapping[node_id]])

    nx.draw(g, pos=node_positions, ax=ax, node_color=colors, with_labels=False)

    ax.set_aspect("equal")

    if not legend:
        return ax

    district_pops = {}
    for node in g:
        d = g.nodes[node]["district"]
        pop = g.nodes[node]["population"]
        if d not in district_pops:
            district_pops[d] = 0
        district_pops[d] += pop
    legend_elements = []
    for key in range(-1, N_DISTRICTS):
        value = district_pops.get(key, 0)
        label = f"{key}: {int(value)}"
        c = DISTRICT_COLORS[key]
        legend_elements.append(
            Line2D(
                [0],
                [0],
                c=c,
                marker="o",
                color="w",
                label=label,
                markerfacecolor=c,
                markersize=15,
            )
        )
    ax.legend(
        handles=legend_elements,
        ncol=len(legend_elements) // 2,
        bbox_to_anchor=(0.5, 0),
        loc="center",
        fontsize=14,
    )

    return ax


# %%
def init_nc_graph(Counties_df):
    ### Make county graph
    g = nx.Graph()
    for index, row in Counties_df.iterrows():
        county_id = row["ID"]
        adjacent_counties_str = row["AdjacentCounties"]

        if adjacent_counties_str:
            adjacent_counties = [
                int(x.strip()) for x in adjacent_counties_str.split(",")
            ]

            for adj in adjacent_counties:
                g.add_edge(county_id, adj)

    init_district_attr = {}
    for index, row in Counties_df.iterrows():
        county_id = row["ID"]
        init_district_attr[county_id] = {}
        init_district_attr[county_id]["district"] = -1
        init_district_attr[county_id]["population"] = row["Population"]
        init_district_attr[county_id]["latitude"] = row["Latitude"]
        init_district_attr[county_id]["longitude"] = row["Longitude"]

    nx.set_node_attributes(g, init_district_attr)

    return g


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
def write_graph(g: nx.graph, path: Path):
    path = Path(path)
    nx.write_gpickle(g, path.stem + ".pickle")


def read_graph(path: Path):
    return nx.read_gpickle(path)


def default_neigh_order_fn(g: nx.graph, neigh_iter: Iterable):
    return neigh_iter


def min_neigh_order_fn(g: nx.graph, neigh_iter: Iterable):
    nlist = []
    plist = []
    for neigh in neigh_iter:
        nlist.append(neigh)
        plist.append(g.nodes[neigh]["population"])
    sort_idx = np.argsort(plist)
    return [nlist[x] for x in sort_idx]


def max_neigh_order_fn(g: nx.graph, neigh_iter: Iterable):
    nlist = []
    plist = []
    for neigh in neigh_iter:
        nlist.append(neigh)
        plist.append(g.nodes[neigh]["population"])
    sort_idx = np.argsort(plist)[::-1]
    return [nlist[x] for x in sort_idx]


# %%

### Make county graph
g = init_nc_graph(Counties)
draw_graph(g, Counties)
write_graph(g, "county_graph.pickle")
# %%
g = init_nc_graph_blocks2020(Blocks2020)
write_graph(g, "Block2020.pickle")
# %%


def get_logistic_weight(current_pop):
    x = current_pop / LOGISTIC_SCALE_FACTOR
    weight = 1 / (1 + np.exp(-LOGISTIC_K * (x - LOGISTIC_MIDPOINT)))
    return weight


def get_logistic_exit(current_pop):
    weight = get_logistic_weight(current_pop)
    return np.random.choice([True, False], size=(1,), p=[weight, 1 - weight])[0]


# %%
def district_filling_iterative(g, d, n, district_pops, target_pop, neigh_order_fn):
    """
    Iteratively assign nodes to districts using the county filling method.

    Arguments
    ---------
    g: nx.Graph
        Networkx graph.
    d: int
        District index.
    n: int
        Node index (seed node).
    district_pops: dict
        Dictionary holding current district populations.
    target_pop: int
        Target population for each district.
    neigh_order_fn: Callable
        Function for ordering the neighbor list.

    """
    # Initialize a queue with the seed node
    queue = set()
    queue.add(n)

    while queue:
        current_node = queue.pop()

        if g.nodes[current_node]["district"] != -1:
            raise Exception("Only input undeclared district")

        # Check if the district population exceeds the limits
        if district_pops[d] > MIN_DISTRICT_POP:
            if district_pops[d] > MAX_DISTRICT_POP:
                break

        # Assign the current node to the district and update its population
        g.nodes[current_node]["district"] = d
        district_pops[d] += g.nodes[current_node]["population"]

        # Process neighbor nodes
        for neigh in neigh_order_fn(g, g.neighbors(current_node), d):
            # Check if the neighbor is already in the queue or if it's None
            if neigh in queue or neigh is None:
                continue
            if g.nodes[neigh]["district"] == -1:
                # Check if adding the neighbor would exceed the district population limit
                if g.nodes[neigh]["population"] > (MAX_DISTRICT_POP - district_pops[d]):
                    continue
                else:
                    queue.append(neigh)


# %%


def district_filling(
    g: nx.graph,
    d: int,
    n: int,
    district_pops: dict,
    target_pop: int,
    neigh_order_fn: Callable,
):
    # add maximum allowable and minimum exit populations as arguments at some
    """
    Recursive function using county filling method to define district.

    Arguments
    ---------
    g: nx.graph
        Networkx graph
    d: int
        District index
    n: int
        Node index (this is the seed node)
    district_pops: dict
        Dictionary holding the current district populations
    target_pop: int
        Target population for each district
    neigh_order_fn: Callable
        Function for ordering the neighbor list. Argument to this function is
        the graph and iterator from g.neighbors(n).

    """
    if g.nodes[n]["district"] != -1:
        raise Exception("Only input undeclared district")

    if district_pops[d] > MIN_DISTRICT_POP:
        if district_pops[d] > MAX_DISTRICT_POP:
            return
        if get_logistic_exit(district_pops[d]):
            return

    g.nodes[n]["district"] = d
    district_pops[d] += g.nodes[n]["population"]
    for neigh in neigh_order_fn(g, g.neighbors(n), d):
        if neigh == None:
            return
        if g.nodes[neigh]["district"] == -1:
            if g.nodes[neigh]["population"] > (MAX_DISTRICT_POP - district_pops[d]):
                continue
            else:
                district_filling(g, d, neigh, district_pops, target_pop, neigh_order_fn)


# %%
def create_districts(
    g: nx.graph,
    target_pop: int,
    district_start_node_fn: Callable,
    neigh_order_fn: Callable,
):
    """
    This is the main function loop that runs the district_filling argument for the number of districts that need to be created as well as re-ordering the neighbor list whenever a new district needs to be created
    """
    district_pops = {}
    for i in range(N_DISTRICTS):
        district_pops[i] = 0
        # district_start_node = district_start_node_fn(g)
        # district_filling(g, i , seed_node...)
        # for district checking if the district is greater than maximum allowable_pop but county size is 1 it is allowed at the moment because no county splitting atm
        n = district_start_node_fn(g)
        if n != None:
            district_filling_iterative(
                g, i, n, district_pops, target_pop, neigh_order_fn
            )
    return district_pops


# %%
def district_start_node_fn(g):
    unassigned_nodes = []
    for node in g:
        if g.nodes[node]["district"] == -1:
            unassigned_nodes.append(node)
    # if there are no unassigned nodes then return None
    if not unassigned_nodes:
        return None
    left_most_node = unassigned_nodes[0]
    for node in unassigned_nodes:
        if g.nodes[node]["latitude"] < g.nodes[left_most_node]["latitude"]:
            left_most_node = node
    return left_most_node


# %%
def random_start_node_fn(g):
    unassigned_nodes = []
    for node in g:
        if g.nodes[node]["district"] == -1:
            unassigned_nodes.append(node)
    if len(unassigned_nodes) == 0:
        return None
    return np.random.choice(unassigned_nodes, (1,))[0]


# %%
# def most_adjacencies_neigh_order_fn(g: nx.graph, neigh_iter: Iterable, d: int):
#     neigh_list = list(neigh_iter)
#     valid_node_list = [None]
#     adj_list = [-1]
#     for n in neigh_list:
#         ### Cannot add if already assigned
#         if g.nodes[n]["district"] != -1:
#             continue
#         valid_node_list.append(n)
#         adj = 0
#         temp_neigh_list = list(g.neighbors(n))
#         for temp_n in temp_neigh_list:
#             if g.nodes[temp_n]["district"] == d:
#                 adj += 1
#         adj_list.append(adj)
#     max_adj = np.max(adj_list)
#     max_idx = np.where(np.array(adj_list) == max_adj)[0]
#     yield valid_node_list[np.random.choice(max_idx, size=(1,))[0]]


def most_adjacencies_neigh_order_fn(g, neigh_iter, d, probability=0.2):
    max_adj_count = -1
    best_neighbors = []
    less_adj_neighbors = []

    for n in neigh_iter:
        if g.nodes[n]["district"] == -1:
            adj_count = sum(
                1 for neighbor in g.neighbors(n) if g.nodes[neighbor]["district"] == d
            )

            if adj_count > max_adj_count:
                max_adj_count = adj_count
                best_neighbors = [n]
                less_adj_neighbors = []
            elif adj_count == max_adj_count:
                best_neighbors.append(n)
            elif adj_count == 1 and random.random() < probability:
                best_neighbors = [
                    n
                ]  # Automatically accept a neighbor with no other adjacencies
            elif random.random() < probability:
                less_adj_neighbors.append(n)

    if less_adj_neighbors:
        return random.choice(less_adj_neighbors)
    elif best_neighbors:
        return random.choice(best_neighbors)
    else:
        return None  # No valid neighbors found


# %%
def default_neigh_order_fn(g: nx.graph, neigh_iter: Iterable, d: int):
    return neigh_iter


def min_neigh_order_fn(g: nx.graph, neigh_iter: Iterable, d: int):
    ### Orders the neighboring nodes from minimum to maximum
    nlist = []
    plist = []
    for neigh in neigh_iter:
        nlist.append(neigh)
        plist.append(g.nodes[neigh]["population"])
    sort_idx = np.argsort(plist)
    return [nlist[x] for x in sort_idx]


def max_neigh_order_fn(g: nx.graph, neigh_iter: Iterable, d: int):
    ### Orders the neighboring nodes from maximum to minimum
    nlist = []
    plist = []
    for neigh in neigh_iter:
        nlist.append(neigh)
        plist.append(g.nodes[neigh]["population"])
    sort_idx = np.argsort(plist)[::-1]
    return [nlist[x] for x in sort_idx]


failure_conditions = [0, 0, 0, 0, 0, 0]


def is_valid_graph(g, district_pops):
    assigned_pop = 0
    for d in range(N_DISTRICTS):
        assigned_pop += district_pops[d]
    if assigned_pop != TOTAL_POPULATION:
        failure_conditions[3] += 1
        return False
    for d in range(N_DISTRICTS):
        if district_pops[d] == 0:
            failure_conditions[0] += 1
            return False
        elif district_pops[d] < MIN_DISTRICT_POP:
            failure_conditions[1] += 1
            return False
        elif district_pops[d] > MAX_DISTRICT_POP:
            failure_conditions[2] += 1
            return False
        assigned_pop += district_pops[d]

    # if assigned_pop != tot_pop[0]:
    #     failure_conditions[3] += 1
    #     return False
    return True


# %%
graph_array = []


def process_graph():
    attempts = 0
    while True:
        attempts += 1
        g = read_graph("Block2020.pickle")
        district_pops = create_districts(
            g,
            POPULATION_PER_DISTRICT,
            district_start_node_fn,
            most_adjacencies_neigh_order_fn,
        )
        print(attempts)
        if not is_valid_graph(g, district_pops):
            break
    graph_array.append(g)


# %%
NUM_PROCESSES = 32
NUM_THREADS_PER_PROCESS = 2

processes = []

for _ in range(NUM_PROCESSES):
    process = multiprocessing.Process(target=process_graph)
    threads = []

    for _ in range(NUM_THREADS_PER_PROCESS):
        thread = threading.Thread(target=process_graph)
        threads.append(thread)

    processes.append((process, threads))

# start all processes and threads
for process, threads in processes:
    process.start()
    for thread in threads:
        thread.start()

for process, threads in processes:
    process.join()
    for thread in threads:
        thread.join()

# %%
for graph in graph_array:
    assign_colors_to_dataframe(g, Blocks2020, DISTRICT_COLORS)
    plot_geopandas_dataframe(Blocks2020, "geometry", "color")
# %%
