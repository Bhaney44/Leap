#NEAL

import neal                                                                     

sampler = neal.SimulatedAnnealingSampler()                                      

h = {0: -1, 1: -1}                                                              

j = {(0, 1): -1}                                                                

response = sampler.sample_ising(h, j)                                           

for sample, energy in response.data(['sample', 'energy']): 
    print(sample, energy) 
