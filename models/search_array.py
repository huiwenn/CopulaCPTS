import pickle
import time
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import itertools

from copulae.core import pseudo_obs

def empirical_copula_loss(x, data, epsilon):
    pseudo_data = pseudo_obs(data)
    return np.fabs(np.mean(np.all(np.less_equal(pseudo_data, np.array([x] * pseudo_data.shape[1])), axis=1)
                           ) - 1 + epsilon)

def empirical_copula_loss_array(x, data, epsilon):
    pseudo_data = pseudo_obs(data)
    z= np.fabs(np.mean(np.all(np.less_equal(pseudo_data, x), axis=1)
                           ) - 1 + epsilon)
    return z


gr = np.arange(0.93,0.999,0.01)
gr_array = np.expand_dims(gr,1)
all_combs_old = gr_array
permut = itertools.permutations(gr, len(gr))

for _ in range(6):
    all_combs = []
    for g in all_combs_old:
        #print('g',g)
        gg = np.expand_dims(g,0)
        #print(gg)
        repeat = np.repeat(gg, gr_array.shape[0], axis=0)
        #print('repeat', repeat)
        #print(gr_array)
        all_combs.append(np.hstack([repeat, gr_array]))
        
    all_combs_old = np.concatenate(all_combs)

x_candidates = all_combs_old
print('made cadidates')



with open('../trained/covidmlp_nonconform.p','rb') as f:
    nonconformity = pickle.load(f)

epsilon=0.1
all_est = []

for d in range(10):

    alphas = nonconformity[:,d:7+d]
    #print(alphas.shape)
    mapping = {i: sorted(alphas[:, i].tolist()) for i in range(alphas.shape[1])}

    #x_candidates = np.mgrid[0.93:0.999:0.005, 0.93:0.999:0.005, 0.93:0.999:0.005, 0.93:0.999:0.005,0.93:0.999:0.005,0.93:0.999:0.005,0.93:0.999:0.005]
    #gr = gr.reshape(14**7, 7)

    start = time.time()

    x_fun = [empirical_copula_loss_array(x, alphas, epsilon) for x in x_candidates]
    print('x_fun', time.time()-start)
    
    #l = list(zip(x_fun, x_candidates))
    l = np.hstack([np.expand_dims(x_fun,-1), x_candidates]).tolist()
    print(l[:5])
    x_sorted = sorted(l)
    print(x_sorted[0])
    print('sort', time.time()-start)

    quantile = np.array([mapping[i][int(x_sorted[0][1+i] * alphas.shape[0])] for i in range(alphas.shape[1])])
    
    all_est.append(quantile)

    end = time.time()
    print('time', end - start)
    print(quantile)


