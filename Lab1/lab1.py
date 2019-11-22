import pylab as pb 
import numpy as np
from math import pi
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt



def plotMultiGaussian(mean,covariance,gridSize):
    dist = multivariate_normal(mean, cov=covariance)
    x, y = gridSize
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y
    z = dist.pdf(pos)
    plt.contourf(x,y,z)
    plt.show()

prior_mean=np.array([0,0])
prior_cov=np.array([[0.5,.1],[.1,0.5]])
plotMultiGaussian(prior_mean,prior_cov,np.mgrid[-0.5:1.51:.1, -2.5:0:.1])