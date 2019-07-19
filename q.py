#QUBO
#Import dimod
import dimod
#Exact solver finds energies for all states.
exactsolver = dimod.ExactSolver()
#Set up the QUBO. 
#Embedding explicitly expressed, with chainstrength variable.
#Chainstrength variable
chainstrength = 1
#Define Q-Matrix
Q = {(0, 0): 2, (1, 1): chainstrength, (5, 5): chainstrength, (0, 4): -2, (1, 4): 2, (0, 5): -2, (1, 5): -2 * chainstrength}
#Define results
results = exactsolver.sample_qubo(Q)
#For loop to print results.
for smpl, energy in results.data(['sample', 'energy']):
    print(smpl, energy)Minimum Vertex Cover

#Minimum Vertex Cover
#Step one: Draw a single graph
#Imports include numpy, matplotlib, and networkx
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
a5 = nx.star_graph(4)
plt.subplot(121)
nx.draw(a5, with_labels=True, front_weight=’bold’)
#Step two: Solve the graphs minimum vertex cover
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
#Define sampler
sampler = EmbeddingComposite(DWaveSampler())
print(dnx.min_vertex_cover(a5, sampler))

#Factor
#Set an integer to factor
P = 21
#Binary representation of P 
bP = "{:06b}".format(P)
print(bP)
#Convert the CSP into BQM
csp = dbc.factories.multiplication_circuit(3)
bqm = dbc.stitch(csp, min_classical_gap=.1)
from helpers import draw
draw.circuit_from(bqm)
#Create variables
p_vars = ['p0', 'p1', 'p2', 'p3', 'p4', 'p5']
#Convert P from decimal to binary
fixed_variables = dict(zip(reversed(p_vars), "{:06b}".format(P)))
fixed_variables = {var: int(x) for(var, x) in fixed_variables.items()}
#Fix product variables
for var, value in fixed_variables.items():
    bqm.fix_variable(var, value)
#Confirm that a P variable has been removed from the BQM
print("Variable p0 in BQM: ", 'p0' in bqm)
print("Variable a0 in BQM: ", 'a0' in bqm)
from helpers.solvers import default_solver
my_solver, my_token = default_solver()
from dwave.system.samplers import DWaveSampler
#Sampler
sampler = DWaveSampler(solver={'qpu': True}) 
from dwave.embedding import embed_bqm, unembed_sampleset
from helpers.embedding import embeddings
#Set a pre-calculated minor-embeding
embedding = embeddings[sampler.solver.id]
bqm_embedded = embed_bqm(bqm, embedding, target_adjacency, 3.0)
#Return num_reads solutions
kwargs = {}
if 'num_reads' in sampler.parameters:
    kwargs['num_reads'] = 50
if 'answer_mode' in sampler.parameters:
    kwargs['answer_mode'] = 'histogram'
response = sampler.sample(bqm_embedded, **kwargs)
#Map back to the BQM's graph
response = unembed_sampleset(response, embedding, source_bqm=bqm)
from helpers.convert import to_base_ten
#Select the first sample. 
sample = next(response.samples(n=1))
dict(sample)
a, b = to_base_ten(sample)
print("Given integer P={}, found factors a={} and b={}".format(P, a, b))





