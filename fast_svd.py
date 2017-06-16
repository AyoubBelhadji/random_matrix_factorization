#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 11:45:41 2017

@author: ayoubbelhadji1
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot
from scipy.stats import chi2
import pylab as mp


### Parameters
N = 1000 # Number of points
d = 2    # Dimension
s_n = 10  # Number of leverage sampling size
s_num_iterations = 1000 # Number of leverage sampling iterations 


mean = [0, 0]
cov = [[1, 0], [0, 1]]

### Data Generation
r_X = np.random.multivariate_normal(mean, cov, N).T
r_X_index = list(range(0,N))

### Computing leverage scores
leverage_scores_r_X = np.sum(r_X*r_X, axis=0)/(np.linalg.norm(r_X)**2)
leverage_sampling = np.zeros((s_n,s_num_iterations))
C = np.zeros((d,s_n))
delta_quadratic_norm_sum = 0
delta_matrix = np.zeros((d,d))
for l in range(1,s_num_iterations):
    ### Sampling according to leverage scores
    leverage_sampling[:,l] = np.random.choice(r_X_index, s_n, p=leverage_scores_r_X,replace=False)
    sqrt_p_vector = np.divide(np.ones((d,s_n)),np.sqrt(leverage_scores_r_X[np.ndarray.tolist(leverage_sampling[:,l].astype(int))]))
    C = (1/np.sqrt(s_n))*(np.multiply(r_X[:,np.ndarray.tolist(leverage_sampling[:,l].astype(int))],sqrt_p_vector))
    delta_quadratic_norm_sum = delta_quadratic_norm_sum + (np.linalg.norm(np.dot(C,C.T) - np.dot(r_X,r_X.T)))**2
    delta_matrix = delta_matrix + np.dot(C,C.T)-np.dot(r_X,r_X.T)
delta_quadratic_norm_sum = delta_quadratic_norm_sum/s_num_iterations
delta_matrix = delta_matrix/s_num_iterations

norm_sum_bound = (1/s_n)*np.linalg.norm(r_X)**4
print(delta_quadratic_norm_sum/norm_sum_bound)
print(np.linalg.norm(delta_matrix))
## Plots
#matplotlib.pyplot.scatter(C[0,:],C[1,:])
#matplotlib.pyplot.show()

#matplotlib.pyplot.scatter(r_X[leverage_sampling,0],r_X[leverage_sampling,1])
#matplotlib.pyplot.show()
##empirical_cov = np.cov(r_X.T)
##plot_ellipse(r_X,cov=empirical_cov)

##leverage_empirical_cov = np.cov(r_X[leverage_sampling,:].T)
##plot_ellipse(r_X[leverage_sampling,:],cov=leverage_empirical_cov)
