from numpy import genfromtxt
import numpy as np
from sklearn.neighbors import NearestNeighbors

import os
import subprocess



# current_dir = os.path.dirname(os.path.realpath(__file__))
# main_name = "../mainProgram"
# args = '10'
# output = subprocess.run([main_name , '19' , '4'], stdout = subprocess.PIPE)
# print(output.stdout.decode('utf-8'))

# corpus = genfromtxt('./points_arranged.csv', delimiter=',')
# queries = genfromtxt('./queries_arranged.csv', delimiter=',')
# nbrs = NearestNeighbors(n_neighbors=1).fit(corpus)
# dist,indices = nbrs.kneighbors(queries)
# #print(np.asarray(corpus[indices]))
# iterable = (corpus[i][0] for i in indices)
# a = np.vstack(corpus[indices])
# #print(a)
# #print(corpus[0])
# knns = genfromtxt('./knn.csv',delimiter=',')
# print(np.array_equal(a,knns))
# for i in range(0,len(a)):
#     if not(np.array_equal(a[i],knns[i])):
#         print(a[i],knns[i])
#         print(i)
# #tmp = call("../mainProgram")
# #print(tmp)
# print("END")
# ##print(corpus)

for i in range(15,19):
        for j in range(2,4):
                if(j*3 + 3 < i):
                        if(-i - j*3 < 10):
                                output = subprocess.run(["../mainProgram" , str(i) , str(j)], stdout = subprocess.PIPE)
                                corpus = genfromtxt('./points_arranged.csv', delimiter=',')
                                queries = genfromtxt('./queries_arranged.csv', delimiter=',')
                                nbrs = NearestNeighbors(n_neighbors=1).fit(corpus)
                                dist,indices = nbrs.kneighbors(queries)
                                a = np.vstack(corpus[indices])
                                knns = genfromtxt('./knn.csv',delimiter=',')
                                print(np.array_equal(a,knns))
                                for k in range(0,len(a)):
                                        if not(np.array_equal(a[k],knns[k])):
                                                print(i,j)
                                                print(a[k],knns[k])
                                                print(k)
                                                break

