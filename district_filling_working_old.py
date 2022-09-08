#%%

from random import random
from tkinter.tix import MAX
import scipy.io
import csv
import pprint as pp
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import networkx as nx
import pandas as pd
import random

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

data_path = Path("data")
N_DISTRICTS = 13
MIN_DISTRICT_POP = 400000
MAX_DISTRICT_POP = 1400000

LOGISTIC_K = 0.75
LOGISTIC_MIDPOINT = 700000
LOGISTIC_SCALE_FACTOR = 100000
LOGISTIC_MIDPOINT /= LOGISTIC_SCALE_FACTOR

#%%

adj_mat = scipy.io.mmread(data_path / "adjMat.mtx")


raw = []
with open(data_path / "raw.csv", "r") as file:
    csvreader = csv.reader(file)
    for row in csvreader:
        raw.append(row)

df = pd.DataFrame(
    columns=[
        "BlockID",
        "Latitude",
        "Longitude",
        "County",
        "Voting District",
        "Census Block",
        "Population",
        "Neighbors",
        "Hubs",
    ],
    data=raw,
)
df["BlockID"] = df["BlockID"].astype(int)
df = df.set_index("BlockID")
df["BlockID"] = df.index.values.astype(int)
df["Longitude"] = df["Longitude"].values.astype(float)
df["Latitude"] = df["Latitude"].values.astype(float)
df["Population"] = df["Population"].values.astype(float)
df["County"] = df["County"].astype(int)

idx2blockid = {}
blockid2idx = {}
for i, blockid in enumerate(df.index):
    idx2blockid[i] = blockid
    blockid2idx[blockid] = i

county_centers = []
with open(data_path / "County_Centroids.csv", "r") as file:
    csvreader = csv.reader(file)
    next(csvreader)
    for row in csvreader:
        county_centers.append([row[0], row[2], row[1]])

county_centers.sort(key=lambda row: (row[0]))
county_idx_lookup = {}
county_centers_dict = {}
for idx, entry in enumerate(county_centers):
    county_centers_dict[int(entry[0])] = [float(entry[2]), float(entry[1])]
    county_idx_lookup[int(entry[0])] = idx

raw_county_borders = []
with open(data_path / "County_Borders2.csv", "r") as file:
    csvreader = csv.reader(file)
    next(csvreader)
    for row in csvreader:
        raw_county_borders.append([row[0], row[1], row[2]])

raw_county_borders.sort(key=lambda row: (row[0]))

raw_county_borders_dict = {}
for entry in raw_county_borders:
    county_idx = int(entry[0])

    # adjacencies are county index
    adjacencies = [int(x) for x in entry[1].split(",")]
    # boundaries are range [0,110405]
    # boundaries are for adjoining counties based on the number of shared borders
    boundaries = [int(x) for x in entry[2].split(",")]

    raw_county_borders_dict[county_idx] = {}
    raw_county_borders_dict[county_idx]["adj"] = adjacencies
    raw_county_borders_dict[county_idx]["b"] = boundaries

### Creating a matrix that contains all of the adjacencies for each county

n_counties = len(raw_county_borders_dict)
county_adj = np.zeros((n_counties, n_counties)).astype(int)

for county, county_info in raw_county_borders_dict.items():
    county_idx = county_idx_lookup[county]
    adj_idx = [county_idx_lookup[x] for x in county_info["adj"]]
    county_adj[county_idx][adj_idx] = county_info["b"]

county_adj = np.array(county_adj)
county_adj = county_adj.T

county_pops_dict = {}
county_pops = np.zeros((len(county_idx_lookup),))
for county, idx in county_idx_lookup.items():
    block_idx = np.where(df["County"].values == county)[0]
    # county_pops[county] = df["Population"].values[block_idx]
    county_pops[idx] = df["Population"].values[block_idx].astype(int).sum()
    county_pops_dict[county] = county_pops[idx]

### Calculate the sum of the population of the counties and then calculate the number of people that should be in each district

total = county_pops.astype(int).sum()
tot_pop = [total, round(total / N_DISTRICTS)]
# fix target population so that it has a factor applied to it

remaining_long = df[["Longitude", "BlockID"]]
remaining_long = remaining_long.sort_values("Longitude", ascending=True)


#%%

seed = remaining_long["BlockID"].values[0]
seedID = seed
seed_county = df.loc[seed]["County"]
print(seedID, seed_county)

#%%
# Colors need to be decided at some point
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


def draw_graph(g, legend=True):
    fig = plt.figure(figsize=(20, 5))
    ax = fig.add_subplot(111)
    colors = []
    label = nx.get_node_attributes(g, "district")
    for node in g:
        colors.append(DISTRICT_COLORS[g.nodes[node]["district"]])
    nx.draw(g, pos=county_centers_dict, ax=ax, node_color=colors, labels=label)
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


"""
Add political affiliation as a node property when the time comes
"""


def init_nc_graph():
    ### Make county graph
    g = nx.Graph()
    for key, value in raw_county_borders_dict.items():
        for adj in value["adj"]:
            g.add_edge(key, adj)

    init_district_attr = {}
    for node_idx in g:
        init_district_attr[node_idx] = {}
        init_district_attr[node_idx]["district"] = -1
        init_district_attr[node_idx]["population"] = county_pops_dict[node_idx]
        init_district_attr[node_idx]["latitude"] = county_centers_dict[node_idx][0]
        init_district_attr[node_idx]["longitude"] = county_centers_dict[node_idx][1]
    nx.set_node_attributes(g, init_district_attr)

    return g


def init_nc_block_graph():
    ### Make block based graph
    g = nx.from_scipy_sparse_matrix(adj_mat)
    init_district_attr = {}
    for node_idx in g:
        blockid = idx2blockid[node_idx]
        init_district_attr[node_idx] = {}
        init_district_attr[node_idx]["district"] = -1
        init_district_attr[node_idx]["population"] = df.loc[blockid]["Population"]
        init_district_attr[node_idx]["county"] = df.loc[blockid]["County"]
        init_district_attr[node_idx]["latitude"] = df.loc[blockid]["Latitude"]
        init_district_attr[node_idx]["longitude"] = df.loc[blockid]["Longitude"]
    nx.set_node_attributes(g, init_district_attr)

    return g


def write_graph(g: nx.graph, path: Path):
    path = Path(path)
    nx.write_gpickle(g, path.stem + ".pickle")


def read_graph(path: Path):
    import pickle5

    with open("block_graph.pickle", "rb") as pickle_file:
        g = pickle5.load(pickle_file)
    return g


def get_logistic_weight(current_pop):
    x = current_pop / LOGISTIC_SCALE_FACTOR
    weight = 1 / (1 + np.exp(-LOGISTIC_K * (x - LOGISTIC_MIDPOINT)))
    return weight


def get_logistic_exit(current_pop):
    weight = get_logistic_weight(current_pop)
    return np.random.choice([True, False], size=(1,), p=[weight, 1 - weight])[0]


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
            district_filling(g, i, n, district_pops, target_pop, neigh_order_fn)
    return district_pops


def district_start_node_fn(g):
    unassigned_nodes = []
    for node in g:
        if g.nodes[node]["district"] == -1:
            unassigned_nodes.append(node)
    if len(unassigned_nodes) == 0:
        return None
    left_most_node = unassigned_nodes[0]
    for node in unassigned_nodes:
        if g.nodes[node]["latitude"] < g.nodes[left_most_node]["latitude"]:
            left_most_node = node
    return left_most_node


def random_start_node_fn(g):
    unassigned_nodes = []
    for node in g:
        if g.nodes[node]["district"] == -1:
            unassigned_nodes.append(node)
    if len(unassigned_nodes) == 0:
        return None
    return np.random.choice(unassigned_nodes, (1,))[0]


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


def draw_block_graph(g: nx.graph):
    ax = plt.axes()
    for i in range(N_DISTRICTS):
        g.nodes


def is_valid_graph(g, district_pops):
    assigned_pop = 0
    for d in range(N_DISTRICTS):
        if district_pops[d] == 0:
            return False
        elif district_pops[d] < MIN_DISTRICT_POP:
            return False
        elif district_pops[d] > MAX_DISTRICT_POP:
            return False
        assigned_pop += district_pops[d]
    if assigned_pop != tot_pop[0]:
        return False
    return True


#%%

while True:
    g = init_nc_graph()
    district_pops = create_districts(
        g, tot_pop[1], district_start_node_fn, most_adjacencies_neigh_order_fn
    )
    if is_valid_graph(g, district_pops):
        break

draw_graph(g)
plt.show()
plt.close()


#%%


#%%

### Investigating logistical function for probability of exiting based on pop
### https://en.wikipedia.org/wiki/Logistic_function
klist = [0.5, 0.75, 1, 2, 5]
x0 = 700000 / 100000
x = np.arange(0, MAX_DISTRICT_POP, 10000) / 100000
fig = plt.figure()
ax = fig.add_subplot(111)
for k in klist:
    y = 1 / (1 + np.exp(-k * (x - x0)))
    ax.plot(x, y, label=f"{k}")
plt.legend()
plt.show()
plt.close()

### I think 0.75 looks good here

#%%

### Ensuring that implementation is correct
x = np.arange(0, MAX_DISTRICT_POP, 100000)
y = 1 / (1 + np.exp(-LOGISTIC_K * (x / LOGISTIC_SCALE_FACTOR - LOGISTIC_MIDPOINT)))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, y)
ntest = 1000
for current_pop in x:
    nexit = 0
    for _ in range(ntest):
        if get_logistic_exit(current_pop):
            nexit += 1
    nexit /= ntest
    ax.scatter(current_pop, nexit, c="k")
plt.show()
plt.close()

#%%
