#%%
"""
This is my own original work based off of the previously created data files from Amy's project and insights from rewriting her code
"""

import scipy.io
import csv
import pprint as pp
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import networkx as nx
import pandas as pd

import matplotlib.pyplot as plt

data_path = Path("data")
N_DISTRICTS = 13

#%%

"""
I separated the NoDonutData into its three separate parts to make it easier to import into Python. 
adj_mat contains the sparse matrix that describes the geographical neighors of each census block. 
This is stored in Coordinate List (COO) format meaning that it is stored as a series of tuples having the 
form (row, column) value.
"""

adj_mat = scipy.io.mmread(data_path / "adjMat.mtx")

### Is graph a more convenient data-type?
# g = nx.from_scipy_sparse_matrix(adj_mat)

"""
This handles getting raw part of the NoDonutData file which consists of a list of census blocks and their
attributes. These are then saved to a list which has the attributes in this specific order
{BlockID, Latitude, Longitude, County, Voting District, Census Block, Population, Neighbors, Hubs]

***Important***
All of the items inside this variable are stored as STRINGS
"""

raw = []
with open(data_path / "raw.csv", "r") as file:
    csvreader = csv.reader(file)
    for row in csvreader:
        raw.append(row)

### Some BlockID is larger than length of list
### This is likely because of the some of the blocks were combined due to donuts being formed
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

### Accessing columns of data:
###     df["Latitude"]

### Accessing rows of info:
###     df.loc["5"]

"""""
This handles importing the lookUp data and saving it as a list 
""" ""
lookUp = []
file = open(data_path / "lookUp.txt")
for row in file:
    lookUp.append(int(row.strip()))

#%%

"""
This handles importing the county centroid file, it ignores the first line which contains the name of the columns and then it sorts the list. It also saves the data in the order CountyID, Latitude, Longitude. Longitude and latitude are flipped from their order in the CSV file.
"""
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

#%%

"""
This handles importing the county borders2 file. It ignores the first row which contains the headers for the data. It then sorts the data by the CountyID
"""
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
    boundaries = [int(x) for x in entry[2].split(",")]

    raw_county_borders_dict[county_idx] = {}
    raw_county_borders_dict[county_idx]["adj"] = adjacencies
    raw_county_borders_dict[county_idx]["b"] = boundaries


#%%

"""
Creating a matrix that contains all of the adjacencies for each county
"""
n_counties = len(raw_county_borders_dict)
county_adj = np.zeros((n_counties, n_counties)).astype(int)

for county, county_info in raw_county_borders_dict.items():
    county_idx = county_idx_lookup[county]
    adj_idx = [county_idx_lookup[x] for x in county_info["adj"]]
    county_adj[county_idx][adj_idx] = county_info["b"]

county_adj = np.array(county_adj)
county_adj = county_adj.T

### With DataFrame, better access pattern should be
###     block_idx = np.where(df["County"] == "37001")[0]
### Remember that block_idx is not the same as BlockID because
###   there's some issue with current BlockIDs
###     block_ids = df.index[block_idx]

#%%

county_pops_dict = {}
county_pops = np.zeros((len(county_idx_lookup),))
for county, idx in county_idx_lookup.items():
    block_idx = np.where(df["County"].values == county)[0]
    # county_pops[county] = df["Population"].values[block_idx]
    county_pops[idx] = df["Population"].values[block_idx].astype(int).sum()
    county_pops_dict[county] = county_pops[idx]

#%%


### Calculate the sum of the population of the counties and then calculate the number of people that should be in each district

total = county_pops.astype(int).sum()
tot_pop = [total, round(total / N_DISTRICTS)]


#%%

### This is good pattern to know, but if we store redundant data
### in df, then access pattern is easier
# remaining_long = np.hstack(
#     [df["Longitude"].values.reshape(-1,1),
#      df.index.values.reshape(-1,1)]
# ).astype(float)

remaining_long = df[["Longitude", "BlockID"]]
remaining_long = remaining_long.sort_values("Longitude", ascending=True)


#%%

seed = remaining_long["BlockID"].values[0]
seedID = seed
seed_county = df.loc[seed]["County"]
print(seedID, seed_county)

#%%

DISTRICT_COLORS = {-1: "k", 0: "tab:blue", 1: "tab:green", 2: "tab:"}


def draw_graph(g):
    fig = plt.figure(figsize=(20, 5))
    ax = fig.add_subplot(111)
    colors = []
    for node in g:
        colors.append(DISTRICT_COLORS[g.nodes[node]["district"]])
    nx.draw(g, pos=county_centers_dict, ax=ax, node_color=colors)
    ax.set_aspect("equal")


"""
Add latitude and longitude as a node property so that I don't need to refer back to dataframe
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
    return nx.read_gpickle(path)


def district_filling(
    g: nx.graph,
    d: int,
    n: int,
    district_pops: dict,
    target_pop: int,
    neigh_order_fn: Callable,
):
    """
    Recursive function using county filling method to define district. Uses
    depth first traversal.

    Arguments
    ---------
    g: nx.graph
        Networkx graph
    d: int
        District index
    n: int
        Node index
    district_pops: dict
        Dictionary holding the current district populations
    target_pop: int
        Target population for each district
    neigh_order_fn: Callable
        Function for ordering the neighbor list. Argument to this function is
        the graph and iterator from g.neighbors(n).

    """
    # This doesn't work for bredth first search
    # Needs to fixed so neighbors
    if g.nodes[n]["district"] != -1:
        raise Exception("Only input undeclared district")
    if district_pops[d] > target_pop:
        return

    g.nodes[n]["district"] = d
    district_pops[d] += g.nodes[n]["population"]

    for neigh in neigh_order_fn(g, g.neighbors(n)):
        if g.nodes[neigh]["district"] == -1:
            district_filling(g, d, neigh, district_pops, target_pop, neigh_order_fn)


def default_neigh_order_fn(g: nx.graph, neigh_iter: Iterable):
    return neigh_iter


def min_neigh_order_fn(g: nx.graph, neigh_iter: Iterable):
    ### Orders the neighboring nodes from minimum to maximum
    nlist = []
    plist = []
    for neigh in neigh_iter:
        nlist.append(neigh)
        plist.append(g.nodes[neigh]["population"])
    sort_idx = np.argsort(plist)
    return [nlist[x] for x in sort_idx]


def max_neigh_order_fn(g: nx.graph, neigh_iter: Iterable):
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


#%%

### Make county graph
g = init_nc_graph()
draw_graph(g)
write_graph(g, "county_graph.pickle")

#%%

### Default neigh ordering
g = read_graph("county_graph.pickle")
district_pops = {}
for d in range(N_DISTRICTS):
    district_pops[d] = 0
district_filling(g, 0, seed_county, district_pops, tot_pop[1], default_neigh_order_fn)
draw_graph(g)

#%%

### min neigh ordering
g = read_graph("county_graph.pickle")
district_pops = {}
for d in range(N_DISTRICTS):
    district_pops[d] = 0
district_filling(g, 0, seed_county, district_pops, tot_pop[1], min_neigh_order_fn)
draw_graph(g)

#%%

### min neigh ordering
g = read_graph("county_graph.pickle")
district_pops = {}
for d in range(N_DISTRICTS):
    district_pops[d] = 0
district_filling(g, 0, seed_county, district_pops, tot_pop[1], max_neigh_order_fn)
draw_graph(g)


#%%
### make block graph
g = init_nc_block_graph()
write_graph(g, "block_graph.pickle")
#%%
"""
This will immediately crash the kernal because there are too many options (possibly because of recursion limit in python)
Before I run using the block graph I will need to create significant optimizations. I do not believe that only looking at the same county at the same time will be enough.
Some better optimization possibilites will be looking a some number of closest nodes within the same county, but the exact optimizations can be determined later.
"""
### block default neigh ordering
# g = read_graph("block_graph.pickle")
# district_pops = {}
# for d in range(N_DISTRICTS):
#     district_pops[d] = 0
# district_filling(g, 0, seed_county, district_pops, tot_pop[1], min_neigh_order_fn)
# draw_graph(g)

#%%

### Now, need scores and different traversal methods
### For example, instead of directly looping g.neighbors, find the best ordering of g.neighbors, and then use that to loop, example below:

#%%

"""
Ideas:
    - Might want to create a better draw method so that I can draw the block graph without it being a mess
        - scipy.spatial.convexhull based on latitude longitude (?)
            - Doesn't work well and doesn't create a good representation

    - Problem with current method
        - District assignment can fail if 
            - Run out of nodes before run out of districts
            - Run out of districts before run out of nodes
        - Although, this should be taken care of in an outter loop
            of district_filling

    - Depth first traversal
        - Before calling recursive, try to flip all neighbors of current node. Then call recursion. 
        
    - Random traversal
        - Random decision perform bredth first filling and then go to 
            neighborlist or to go directly to neighborlist
        - Use random ordering of neighbor list before recursive call
        - So the idea here is not that bredth first or depth first 
            will be observed with high probability, but that with 
            high probability something in-between these two will be
            observed
            
    - For input block graph
        - Order neighborlist before recursive calls using first the blocks in the same county ID and then blocks in the next lowest county ID
        - This is block filling method by county first
    - For deduplication of graphs
        - Reorder all districts based on graph node #, where if node #1 has district 10, then this district is renamed to district 1, and so on in an ordered way through the graph node #s. 
        
        Reordering Algorithm Example: 
        
        Step 1
            Nodes: 1 2 3 4 5 6 7 8 9 10
        Districts: 5 5 5 4 4 4 3 3 2 1
        
        Step 2
            Nodes: 1 2 3 4 5 6 7 8 9 10
        Districts: 1 1 1 4 4 4 3 3 2 5
        
        Step 3
            Nodes: 1 2 3 4 5 6 7 8 9 10
        Districts: 1 1 1 2 2 2 3 3 4 5
        
        - Then a hash of the tuple of the district numbers in the order of the node #s will be unqiue for the purposes of district assignment
        - Building a dictionary with tuples as keys will contain only unique graphs. 
            For example, the unique key above is (1 1 1 2 2 2 3 3 4 5)
    
    - Scoring function for graphs
        - What heuristics should be used? How to code these?
        - Edges of districts can be determined using:
            - Convex hull method in scipy(?)
            - Marching cubes in skimage(?)
            - Can use skewed factor of 2D shape with these edges
    
    - With scoring function, then Monte Carlo Search can be performed
        1. Collect all nodes on edge of each district
            - This determined by any node that shares an edge with a 
                node that has a district that is not it's own
        2. Uniformly randomly choose 1 of these nodes, flip the node 
            to a uniformly randomly chosen neighboring district and 
            evaluate the score
                - If the score is better than the current graph, then 
                    the new graph because the current state
                - If the score is worse than the current graph, then 
                    accept the new graph with probability equal to a 
                    function of worsening and the temperature of the simulation, such that if the temperature is high, then there's higher probability of accepting a given worsening
        3. Repeat this until some herustic of convergence, for example
            not finding a new graph within 1,000 attempts
    
"""

#%%

"""
What follows is now an old idea. It's much better/general to use a function 
to sort the neighbor list as an argument to district filling. 
"""


def district_filling_min(
    g: nx.graph, d: int, n: int, district_pops: dict, target_pop: int
):
    """
    Recursive function using county filling method to
    define district. Uses depth first traversal.

    Arguments
    ---------
    g: nx.graph
        Networkx graph
    d: int
        District index
    n: int
        Node index
    district_pops: dict
        Dictionary holding the current district populations
    target_pop: int
        Target population for each district
    """
    if g.nodes[n]["district"] != -1:
        raise Exception("Only input undeclared district")
    if district_pops[d] > target_pop:
        return

    g.nodes[n]["district"] = d
    district_pops[d] += g.nodes[n]["population"]

    neigh_list = []
    neigh_pop_list = []
    for neigh in g.neighbors(n):
        if g.nodes[neigh]["district"] == -1:
            neigh_list.append(neigh)
            neigh_pop_list.append(g.nodes[neigh]["population"])
    sort_idx = np.argsort(neigh_pop_list)
    neigh_list = [neigh_list[x] for x in sort_idx]

    for neigh in neigh_list:
        if g.nodes[neigh]["district"] == -1:
            district_filling_min(g, d, neigh, district_pops, target_pop)


g = init_nc_graph()
district_pops = {}
for d in range(N_DISTRICTS):
    district_pops[d] = 0
district_filling_min(g, 0, seed_county, district_pops, tot_pop[1])
draw_graph(g)

g = init_nc_graph()
district_pops = {}
for d in range(N_DISTRICTS):
    district_pops[d] = 0
district_filling_min(g, 0, 37001, district_pops, tot_pop[1])

for node in g:
    if g.nodes[node]["district"] == -1:
        district_filling_min(g, 1, node, district_pops, tot_pop[1])

draw_graph(g)
# %%
