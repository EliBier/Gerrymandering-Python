#%%

import numpy as np

#%%

lookup_list = [x for x in range(10000)]
lookup_dict = {}
for x in lookup_list:
    lookup_dict[x] = True

#%%

testn = np.random.randint(0,200,(1000,))

#%%

def test(testn,lookup):
    for n in testn:
        if n in lookup:
            pass

# %timeit test(testn,lookup_list)
# %timeit test(testn,lookup_dict)

#%%

# load_ext line_profiler
# %lprun -f line_test line_test()

def line_test(n=1000):
    lookup_list = [x for x in range(n)]
    lookup_dict = {}
    for x in lookup_list:
        lookup_dict[x] = True
    
    testn = np.random.randint(0,200,(1000,))
    
    for n in testn:
        if n in lookup_list:
            pass
        
    for n in testn:
        if n in lookup_dict:
            pass


# %%

from IPython import embed

def line_test(n=1000):
    lookup_list = [x for x in range(n)]
    lookup_dict = {}
    for x in lookup_list:
        lookup_dict[x] = True
    
    testn = np.random.randint(0,n*1000,(1000,))
        
    for n in testn:
        if n in lookup_dict:
            pass
        else:
            ### Code failed! Start ipython with local variables to debug
            embed()
            exit()

line_test()
#%%

### Installing source directory as library (dev purposes):
### In directory with setup.py:
###     pip install -e .

### Installing into Python libraries:
### In directory with setup.py:
###     python setup.py install 

### Linting python file:
###     black *.py

#%%