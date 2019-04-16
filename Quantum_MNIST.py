#NOTICE
#Code is buggy and does not work yet

import numpy as np
from sklearn.datasets import load_digits

# Load data set
dig = load_digits()

# Shuffle data for random selection for training and test data
idx = np.arange(len(dig.target))
np.random.shuffle(idx)

# Use 2/3 of data set for training, 1/3 for testing
idx_train = idx[:2*len(idx)//3]
idx_test = idx[2*len(idx)//3:]

# Set up data points for training and testing
X_train = dig.data[idx_train]
X_test = dig.data[idx_test]

# Set up labels for training and testing.
y_train = 2 * dig.target[idx_train] - 1  
y_test = 2 * dig.target[idx_test] - 1


#Dataset information

print("Training data size: \t%d samples with %d features" %(X_train.shape[0], X_train.shape[1]))
print("Testing data size: \t%d samples" %(X_test.shape[0]))
print('---------------------------------------')



#QBoost and QBoost Plus are quantum algorithms
#QBoost and QBoost Plus require DWave resources to successfully run
#QBoost Function

def QBoost(X_train, y_train, X_test, y_test):
    NUM_READS = 1000
    DW_PARAMS = {'num_reads': NUM_READS,
                 'auto_scale': True,
                 'num_spin_reversal_transforms': 10,
                 'postprocess': 'optimization',
                 }

    from dwave.system.samplers import DWaveSampler
    from dwave.system.composites import EmbeddingComposite

    dwave_sampler = DWaveSampler(solver={'qpu': True}) 
    emb_sampler = EmbeddingComposite(dwave_sampler)

    from qboost import WeakClassifiers, QBoostClassifier

    clf4 = QBoostClassifier(n_estimators=30, max_depth=2)
    clf4.fit(X_train, y_train, emb_sampler, lmd=1.0, **DW_PARAMS)
    y_train4 = clf4.predict(X_train)
    y_test4 = clf4.predict(X_test)

    from sklearn.metrics import accuracy_score

    print('Accuracy for training data: \t', (accuracy_score(y_train, y_train4)))
    print('Accuracy for test data: \t', (accuracy_score(y_test, y_test4)))
    
    return clf4
    
clf4 = QBoost(X_train, y_train, X_test, y_test)

#Print Results
print('QBoost: ')
clf4 = QBoost(X_train, y_train, X_test, y_test) 
print('---------------------------------------')

