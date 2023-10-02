#%%
"""
This is a recreation of Amy's mathematica code for the whole county first into Python. 
To maintain continuity between the two files variable names have been kept the same and the code
has not been modified to the maximum extent
"""

import scipy.io
import csv
import pprint as pp
import numpy as np
from pathlib import Path

data_path = Path("data")

#%%

"""
I separated the NoDonutData into its three separate parts to make it easier to import into Python. 
adjMat contains the sparse matrix that describes the geographical neighors of each census block. 
This is stored in Coordinate List (COO) format meaning that it is stored as a series of tuples having the 
form (row, column) value.
"""

adjMat = scipy.io.mmread(data_path / "adjMat.mtx")

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
countyCenters = []
with open(data_path / "County_Centroids.csv", "r") as file:
    csvreader = csv.reader(file)
    next(csvreader)
    for row in csvreader:
        countyCenters.append([row[0], row[2], row[1]])

countyCenters.sort(key=lambda row: (row[0]))


"""
This handles importing the county borders2 file. It ignores the first row which contains the headers for the data. It then sorts the data by the CountyID
"""
rawCountyBorders = []
with open(data_path / "County_Borders2.csv", "r") as file:
    csvreader = csv.reader(file)
    next(csvreader)
    for row in csvreader:
        rawCountyBorders.append([row[0], row[1], row[2]])

rawCountyBorders.sort(key=lambda row: (row[0]))

#%%

###### County Weighted Adjacency Matrix ######


"""
My best guess at what is beign done in this code is that I am taking the the data from the rawCountyBorders list and using it to create a list of which counties share a border
this is then saved into the countyBorders list in the form [[PrimaryCountyID],[Something, Something]]. I am not certain what this data means and could use more help understanding why these things are being done
"""
countyBorders = []
for county in rawCountyBorders:
    adjacencies = [int(x) for x in county[1].split(",")]
    boundaries = [int(x) for x in county[2].split(",")]

    combined = []
    if len(adjacencies) == len(boundaries):
        for i in range(len(adjacencies)):
            combined.append([boundaries[i], (int(adjacencies[i] - 37000 + 1) // 2)])

    else:
        print("Error")

    countyBorders.append([(int(county[0]) - 37000 + 1) // 2, combined])


#%%

"""
This is clearly creating some sort of adjacency matrix which is then transposed but I don't understand the purpose and why this matrix is being created
"""
countyAdj = []
for i in range(100):
    countyAdj.append([0 for x in range(100)])

countyList = []
for i in range(len(countyBorders)):
    countyList.append(countyBorders[i][0])

for i in range(100):
    adjacencies = countyBorders[i][1]
    for adjacentCounty in adjacencies:
        adjacentCountyIndex = adjacentCounty[1]
        countyAdj[i][adjacentCountyIndex - 1] = adjacentCounty[0]

countyAdj = np.array(countyAdj)
countyAdj = countyAdj.T


#%%

###### County Populations ######

"""
This code will diverge slightly from the original design because the raw data is currently stored as strings and I am not certain what type each field in raw should be.
I have chosen to keep them as strings and thus they will continue to be strings in the variable blocksByCounty. As necessary I will change the type of each field during later 
code to be used as in the original code.
"""

"""
The creation of the blocksByCounty variable was initially somewhat confusing but I eventually realized that it is creating a multidimensional list where each set of values is grouped by it's county variable
This is probably not the most efficient way to implement this functionality but I chose to create a three objects that will be used to help create blocksByCounty. First I instantiate temp list which will be used to hold a list of all of the rows that have the same county value,
I then create the previousCounty variable with an initial value equal to first county value that is stored in the raw list. I then begin iteratating through the rows of the raw list. The first thing done in each iteration is that the currentCounty variable will be assigned the value of county in the current row.
Then the currentCounty is checked against the previousCounty, if these two values are the same then the row is appended the temp list. If they are different the entire temp list is appended to the blocksByCounty list, the temp list is reset, the current row appended to the temp list and then previousCounty value is set to be equal to the currentCounty.
The final step is performed outside the loop and it appends the contents of the temp list to the blocksByCounty list. This is necessary because otherwise the last county will not be included inside the blocksByCounty list.
This should mimic the behaviour of the original mathematica code.
"""

blocksByCounty = []
temp = []
previousCounty = raw[0][3]
for row in raw:
    currentCounty = row[3]
    if currentCounty == previousCounty:
        temp.append(row)
    else:
        blocksByCounty.append(temp)
        temp = []
        temp.append(row)
        previousCounty = currentCounty

blocksByCounty.append(temp)

#%%

"""
This should sum all of the populations within each of the counties and assign it to a list called countyPops in the form [CountyID, CountyPopulation]
"""
countyPops = []

for countyBlock in blocksByCounty:
    total = 0
    for county in countyBlock:
        total += int(county[6])
    countyPops.append([int(countyBlock[0][3]), total])


#%%


###### County First Approach ######

totPop = []
total = 0
for row in raw:
    total += int(row[6])

totPop.append(total)
totPop.append(round(total / 13, 0))


#%%


"""
Initializes 13 empty districts to fill with full info for each block for all 13 districts
"""
districts = []
for i in range(13):
    districts.append([])

"""
districtCounties will contain a list of counties that are contained in each district as a number from 1 to 100. Split counties are listed as 100+countyID when partially used. Otherwise counties will just be listed by countyID when the district uses all remaining blocks in the county
"""
districtCounties = []
for i in range(13):
    districtCounties.append([])

"""
Createx a copy of raw and adjMat that will be used throughout the program
"""
workingData = raw
workingAdj = adjMat

"""
Partitions raw into 100 groups, 1 for each county. This is the same as the code that was run to create blocksByCounty in the County Population section so I will reuse that variable. I have set it equal to itself to remind me that this variable was created earlier
"""
blocksByCounty = blocksByCounty

"""
"""
remainingLong = []

#%%

"""
This code chooses a seed which is currently set as the leftmostblock. This code diverges slightly from the original Mathematica code for language purposes, but it maintains the same functionality. 
To decide the seed a variable that contains the Longitude and blockID of each block is created and named remainingLong. This list is then sorted in reverse order based on Longitude.
"""
remainingLong = []
[remainingLong.append([(row[2]), int(row[0])]) for row in workingData]
remainingLong.sort(key=lambda row: (row[0]), reverse=True)

"""
I was initially confused what the Mathematica code did to calculate the seed but it seems to be finding the first position in the workingData list that has the values of the first entry in remainingLong. I have replicated this functionality in the following python code
"""
search = remainingLong[0]
for i in range(len(workingData)):
    if workingData[i][2] == search[0]:
        seed = i
        break
"""
52524 -> This is the western most block in NC which is used  as a seed. Not sure why seedID prints out 53511. This is the same value as printed in mathematica.  *** This doesn't seem right based on comments ***
"""
seedID = workingData[seed][0]

print(seedID)

#%%


# District creating do loop

for districtNum in range(12):

    # Determine which county the seed is in
    seedCounty = int(workingData[seed][3])
    # Adds the current county as the first county to add with a shared border length of 0 as it is the only option so the length does not matter
    countiesToAdd = [[0, (seedCounty - 37000 + 1) // 2]]

    # initialize variables/lists
    countyTooBig = 0
    districtBlockIDs = []
    districtPop = 0

    print("~~~~~~~~~~~~~~~~~~~~")

    # Loop through counties to add until we find a county that is too big

    while len(countiesToAdd) != 0:

        # Looks at the next county in the counties to add list
        currentCounty = countiesToAdd[0][1]
        del countiesToAdd[0]

        # Determines if the whole county can be added (countyPops[currentCounty-1] needs to be adjusted by -1 because list indexing begins at 0 in python and 1 in mathematica)
        if (districtPop + countyPops[currentCounty - 1][1]) < totPop[1]:
            # True - Add the whole County

            # Update counties in district, blocks in district and district population with the whole county
            districtCounties[districtNum].append(currentCounty)
            districtBlockIDs.append(blocksByCounty[currentCounty - 1])
            districtPop += countyPops[currentCounty - 1][1]

            # Zero out remaining blocks and countyPops for the selected county
            blocksByCounty[currentCounty - 1] = []
            countyPops[currentCounty][1] = 0

            # Determine what counties to add to the list of counties to add
            newAdjacentSharedBorders = countyBorders[currentCounty - 1][1]

#%%
