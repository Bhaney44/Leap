#Set list of numbers

S = [25, 7, 13, 31, 42, 17, 21, 10]
C = sum(S)

#Set up QUBO dictionary

Q={}
Q[(0,0)]=-14100
Q[(0,1)]=1400
Q[(0,2)]=2600
Q[(0,3)]=6200
Q[(0,4)]=8400
Q[(0,5)]=3400
Q[(0,6)]=4200

Q[(1,1)]=-4452
Q[(1,2)]=728
Q[(1,3)]=1736
Q[(1,4)]=2352
Q[(1,5)]=952
Q[(1,6)]=1176

Q[(2,2)]=-7956
Q[(2,3)]=3224
Q[(2,4)]=4368
Q[(2,5)]=1768
Q[(2,6)]=2184

Q[(3,3)]=-16740
Q[(3,4)]=10416
Q[(3,5)]=4216
Q[(3,6)]=5208

Q[(4,4)]=-20832
Q[(4,5)]=5712
Q[(4,6)]=7056

Q[(5,5)]=-10132
Q[(5,6)]=2856

Q[(6,6)]=-12180

#Print Dictionary
#print(Q)

#Set up QPU parameters

chainstrength = 8000

#Number of returns
numruns = 10

#Run the QUBO on the solver from your config file

from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite

response = EmbeddingComposite(DWaveSampler()).sample_qubo(Q, chain_strength=chainstrength, num_reads=numruns)

#Return Results
R = iter(response)
E = iter(response.data())

for line in response:
    sample = next(R)
    S1 = [S[i] for i in sample if sample[i]> 0]
    S0 = [S[i] for i in sample if sample[i]< 1]
    print("S0 Sum:", sum(S0), "\tS1 Sum: ", sum(S1), "\t", S0)

#Diagonals of matrix
for i in range(len(S)):
    Q[(i,i)]= -4*C*S[i]+4*S[i]*S[i]
    
#Off diagonal
for i in range(len(S)):
    for j in range(i+1, len(S)): 
        Q[(i,j)]= 8*S[i]*S[j]
        
