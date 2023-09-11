# %%
from random import random
import shelve
import threading
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
Counties = pd.read_csv(data_path / "Counties.csv")
CountiesAdjMat = np.genfromtxt(data_path / "CountiesAdjMat.csv", delimiter=",")
# %%
TOTAL_POPULATION = Counties["Population"].sum()
POPULATION_PER_DISTRICT = round(TOTAL_POPULATION / N_DISTRICTS)
# %%
Blocks.sort_values(by="Longitude", ascending=True, inplace=True)
seed = Blocks.iloc[0]


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
        node_positions[node_id] = (row["longitude"], row["latitude"])
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
        county_id = row["CountyID"]
        adjacent_counties = row["AdjacentCounties"]
        if adjacent_counties:
            for adj in adjacent_counties:
                g.add_edge(county_id, adj)

    init_district_attr = {}
    for index, row in Counties_df.iterrows():
        county_id = row["CountyID"]
        init_district_attr[county_id] = {}
        init_district_attr[county_id]["district"] = -1
        init_district_attr[county_id]["population"] = row["Population"]
        init_district_attr[county_id]["latitude"] = row["Latitude"]
        init_district_attr[county_id]["longitude"] = row["Longitude"]

    nx.set_node_attributes(g, init_district_attr)

    return g


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


def get_logistic_weight(current_pop):
    x = current_pop / LOGISTIC_SCALE_FACTOR
    weight = 1 / (1 + np.exp(-LOGISTIC_K * (x - LOGISTIC_MIDPOINT)))
    return weight


def get_logistic_exit(current_pop):
    weight = get_logistic_weight(current_pop)
    return np.random.choice([True, False], size=(1,), p=[weight, 1 - weight])[0]


# %%
def district_filling_iterative(
    g: nx.graph,
    d: int,
    n: int,
    district_pops: dict,
    target_pop: int,
    neigh_order_fn: Callable,
):
    # add maximum allowable and minimum exit populations as arguments at some
    """
    iterative function using county filling method to define district.

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
    # Optimization (Add all blocks within the current county if the current county would not cause overage)
    queue = []
    queue.append(n)
    while len(queue) > 0:
        current_node = queue.pop()
        if g.nodes[current_node]["district"] != -1:
            raise Exception("Only input undeclared district")

        # Explain later
        if district_pops[d] > MIN_DISTRICT_POP:
            if district_pops[d] > MAX_DISTRICT_POP:
                break

        g.nodes[current_node]["district"] = d
        district_pops[d] += g.nodes[current_node]["population"]

        for neigh in neigh_order_fn(g, g.neighbors(current_node), d):
            # Needs to be faster way to determine if a node is already in a queue
            if neigh in queue or neigh == None:
                continue
            if g.nodes[neigh]["district"] == -1:
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
def most_adjacencies_neigh_order_fn(g: nx.graph, neigh_iter: Iterable, d: int):
    neigh_list = list(neigh_iter)
    valid_node_list = [None]
    adj_list = [-1]
    for n in neigh_list:
        ### Cannot add if already assigned
        if g.nodes[n]["district"] != -1:
            continue
        valid_node_list.append(n)
        adj = 0
        temp_neigh_list = list(g.neighbors(n))
        for temp_n in temp_neigh_list:
            if g.nodes[temp_n]["district"] == d:
                adj += 1
        adj_list.append(adj)
    max_adj = np.max(adj_list)
    max_idx = np.where(np.array(adj_list) == max_adj)[0]
    yield valid_node_list[np.random.choice(max_idx, size=(1,))[0]]


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
    print(failure_conditions)
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
