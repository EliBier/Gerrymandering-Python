"""
This is a recreation of Amy's mathematica code for the whole county first into Python. 
To maintain continuity between the two files variable names have been kept the same and the code
has not been modified to the maximum extent
"""

import scipy.io
import csv


def importNoDonutData():
    """
    I separated the NoDonutData into its three separate parts to make it easier to import into Python.
    adjMat contains the sparse matrix that describes the geographical neighors of each census block.
    This is stored in Coordinate List (COO) format meaning that it is stored as a series of tuples having the
    form (row, column) value.
    """

    adjMat = scipy.io.mmread("adjMat.mtx")

    """
    This handles getting raw part of the NoDonutData file which consists of a list of census blocks and their
    attributes. These are then saved to a list which has the attributes in this specific order
    {BlockID, Latitude, Longitude, County, Voting District, Census Block, Population, Neighbors, Hubs]
     """
    raw = []
    with open("raw.csv", "r") as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            raw.append(row)
