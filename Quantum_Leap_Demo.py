
# coding: utf-8

# In[5]:


#Step 1 Draw a single graph

import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

a5 = nx.star_graph(4)
plt.subplot(121)

nx.draw(a5, with_labels=True, front_weight='bold')


# In[4]:


#Solve the graph minimum vertex cover on the CPU

from dimod.reference.samplers import ExactSolver
import dwave_networkx as dnx

sampler = ExactSolver()
print(dnx.min_vertex_cover(a5, sampler))


# In[6]:


#step 3 solve the graphs minimum vertex cover on the QPU

from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite

sampler = EmbeddingComposite(DWaveSampler())
print(dnx.min_vertex_cover(a5, sampler))


# In[7]:


#Step 4 try with bigger graph

w5 = nx.wheel_graph(5)
plt.subplot(121)
nx.draw(w5, with_labels=True, font_weights='bold')
print(dnx.min_vertex_cover(w5, sampler))

