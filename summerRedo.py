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

data_path = Path("data")
N_DISTRICTS = 13
MIN_DISTRICT_POP = 400000
MAX_DISTRICT_POP = 1000000

LOGISTIC_K = 0.75
LOGISTIC_MIDPOINT = 700000
LOGISTIC_SCALE_FACTOR = 100000
LOGISTIC_MIDPOINT /= LOGISTIC_SCALE_FACTOR

# %%
Blocks = pd.read_csv(data_path / "Blocks.csv")
Counties = pd.read_csv(data_path / "Counties.csv")
CountiesAdjMat = np.genfromtxt(data_path / "CountiesAdjMat.csv", delimiter=",")
