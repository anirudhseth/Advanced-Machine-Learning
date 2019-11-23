import pylab as pb
import numpy as np
from math import pi
from scipy.spatial.distance import  cdist
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import random

def plotGP(noise,length_scale,no_of_samples):
	X = np.linspace(-10.0,10.0,100) 
	X=X.reshape(-1,1)
	mu = np.zeros(X.shape[0])
	K =  noise*np.exp(-cdist(X, X, 'sqeuclidean')/(length_scale**2))
	Z = np.random.multivariate_normal(mu,K,no_of_samples)
	for i in range(no_of_samples):
		plt.plot(X[:],Z[i,:])
	plt.title('Gaussin Prior with length-scale:'+str(length_scale))
	plt.show()

noise=1
length_scale=[0.1,1,10,100]
no_of_samples=10
for l in  length_scale:
	plotGP(noise,l,no_of_samples)