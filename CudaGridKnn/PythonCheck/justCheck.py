from numpy import genfromtxt
import numpy as np
from sklearn.neighbors import NearestNeighbors

import os
import subprocess




corpus = genfromtxt('./points_arranged.csv', delimiter=',')
queries = genfromtxt('./queries_arranged.csv', delimiter=',')
nbrs = NearestNeighbors(n_neighbors=1).fit(corpus)
dist,indices = nbrs.kneighbors(queries)
#print(np.asarray(corpus[indices]))
a = np.vstack(corpus[indices])
#print(a)
#print(corpus[0])
knns = genfromtxt('./knn.csv',delimiter=',')
print(np.array_equal(a,knns))
for i in range(0,len(a)):
    if not(np.array_equal(a[i],knns[i])):
        print(a[i],knns[i])
        print(i)
#tmp = call("../mainProgram")
#print(tmp)
print("END")
##print(corpus)

