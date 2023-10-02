# %%
import geopandas as gpd
import networkx as nx
from pathlib import Path
import csv
import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix

# %%
# Define the output file path
output_file = "adjacency.csv"

# Load your GeoDataFrame
data_path = Path("SBE_PRECINCTS_CENSUSBLOCKS_20210727")
gdf = gpd.read_file(data_path / "SBE_PRECINCTS_CENSUSBLOCKS_20210727.shp")
# %%
# Create a spatial index for faster spatial queries
gdf_sindex = gdf.sindex

# Create an empty dictionary to store adjacent polygons for each polygon
adjacency_dict = {}

# Define the batch size (number of polygons to process before saving progress)
batch_size = 1000
# %%


# Define a function to save progress
def save_progress(progress_dict, output_file):
    with open(output_file, "w", newline="") as csvfile:
        fieldnames = ["id", "Polygon", "Adjacent_Polygons"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for index, neighbors in progress_dict.items():
            writer.writerow(
                {
                    "id": gdf.loc[index, "id"],
                    "Polygon": index,
                    "Adjacent_Polygons": ", ".join(map(str, neighbors)),
                }
            )


# Check if a progress file exists, and if so, resume from it
if Path(output_file).is_file():
    print(f"Resuming from existing progress file: {output_file}")
    with open(output_file, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            polygon_index = int(row["Polygon"])
            adjacent_polygons = [int(x) for x in row["Adjacent_Polygons"].split(", ")]
            adjacency_dict[polygon_index] = adjacent_polygons

# Get the last processed index or start from the beginning
last_processed_index = max(adjacency_dict.keys(), default=-1)

# Iterate through the GeoDataFrame starting from where it left off
for index, row in gdf.iloc[last_processed_index + 1 :].iterrows():
    # Get the geometry of the current polygon
    geom = row["geometry"]

    # Use the spatial index to find neighboring polygons
    possible_matches_index = list(gdf_sindex.intersection(geom.bounds))

    # Filter potential neighbors by checking actual geometry intersection
    possible_neighbors = gdf.iloc[possible_matches_index]
    actual_neighbors = possible_neighbors[possible_neighbors.intersects(geom)]

    # Exclude the current polygon itself
    actual_neighbors = actual_neighbors[actual_neighbors.index != index]

    # Store the adjacent polygons in the dictionary
    adjacency_dict[index] = actual_neighbors.index.tolist()

    # Save progress periodically
    if len(adjacency_dict) % batch_size == 0:
        save_progress(adjacency_dict, output_file)
        print(f"Processed {len(adjacency_dict)} polygons and saved progress.")

# Save the final adjacency information to a CSV file
save_progress(adjacency_dict, output_file)

print(f"Adjacency information saved to {output_file}")

# %%
import pandas as pd
import networkx as nx

# Step 1: Read the CSV file
csv_file = "adjacency.csv"
df = pd.read_csv(csv_file)
# %%
# Step 2: Create an empty NetworkX graph
G = nx.Graph()
# %%
# Step 3: Parse the adjacency information and add edges to the graph
for index, row in df.iterrows():
    polygon = row["Polygon"]
    adjacent_polygons = row["Adjacent_Polygons"].split(", ")
    # Add edges for each adjacent polygon
    for adjacent_polygon in adjacent_polygons:
        G.add_edge(polygon, adjacent_polygon)

# Step 4: You now have a NetworkX graph (G) representing the adjacency relationships

# %%
art_points = []
for i in nx.articulation_points(G):
    art_points.append(i)
# %%
nx.number_connected_components(G)
# %%
[len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
# %%
filtered_df = df[df["Adjacent_Polygons"].isna() | (df["Adjacent_Polygons"] == "")]
# %%

# Assuming you have a pandas DataFrame 'df' with columns 'id', 'Polygon', and 'Adjacent_Polygons'


# Function to convert a comma-separated string of polygon IDs to a string of adjacent IDs
def convert_adjacent_polygons(adjacent_polygons_str):
    if pd.notna(adjacent_polygons_str):  # Check for NaN values
        adjacent_polygon_ids = adjacent_polygons_str.split(", ")
        id_list = []
        for polygon_id in adjacent_polygon_ids:
            matching_rows = df[df["Polygon"] == polygon_id]
            if not matching_rows.empty:
                id_value = matching_rows["id"].values[0]
                id_list.append(str(id_value))
        return ", ".join(id_list)
    else:
        return ""


# Apply the conversion function to create the 'adjacent_ids' column
df["adjacent_ids"] = df["Adjacent_Polygons"].apply(convert_adjacent_polygons)

# 'df' now contains an 'adjacent_ids' column with the comma-separated adjacent IDs

# %%
# %%
