import pylab as pb
import numpy as np
from math import pi
from scipy.spatial.distance import  cdist
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import random
def plot2dGaussian(mu,cov,w0,w1,title):
	dist=multivariate_normal(mu,cov)
	X,Y=np.meshgrid(w0,w1)
	Z = np.zeros((len(w0),len(w1)))
	for i in range(len(w0)):
		for j in range(len(w1)):
			Z[i,j] = multivariate_normal(mu, cov).pdf([X[i,j],Y[i,j]])
	plt.contourf(X,Y,Z,cmap = 'jet')
	# plt.imshow(Z,cmap='jet',extent=(-1.5, 1.5, -1.5, 1.5))
	plt.xlabel('w1')
	plt.ylabel('w0')
	plt.title(title)
	plt.axis('scaled')
	plt.show()

def calPosterior(prior_mu,prior_cov,noise_mu,noise_cov,X_Label,T,points):
	temp=random.randint(0,X_Label.shape[0])
	X_observed=np.ones(points)
	T_observed=np.ones(points)
	for i in range(points):
		temp=random.randint(0,X_Label.shape[0])
		X_observed[i]=X_Label[temp,1]
		T_observed[i]=T[temp,0]
	X_observed=X_observed.reshape(-1,1)
	X_observed=np.hstack((X_observed,np.ones(X_observed.shape[0]).reshape(-1,1)))
	X_observed=np.flip(X_observed,axis=1)
	# print(X_observed)
	T_observed=T_observed.reshape(-1,1)
	print(T_observed)
	posterior_cov=np.linalg.inv((1./(noise_cov))*(np.matmul(X_observed.transpose(),X_observed))+np.linalg.inv(prior_cov))
	posterior_mean=np.matmul(posterior_cov,np.matmul(X_observed.transpose(), T_observed))/noise_cov
	posterior_mean=posterior_mean.flatten()
	# print(posterior_cov)
	# print(posterior_mean)
	return posterior_mean,posterior_cov
# Create Data
W=np.array([-1.5,0.5]).reshape(-1,1)
X_Label=np.arange(-1,1.01,.01)
X_Label=np.stack((X_Label,np.ones(len(X_Label))),axis = 1)
X_Label=np.flip(X_Label,axis=1)
noise_mu=0
noise_cov=0.2
noise = np.random.normal(noise_mu, noise_cov, X_Label.shape[0])
noise=noise.reshape(-1,1)
T=np.matmul(X_Label,W)+noise


# plot prior
prior_mu=np.array([0,0])
prior_cov=np.array([[0.25,0],[0,0.25]])
w0=np.arange(-2,2,0.1)
w1=np.arange(-2,2,0.1)
title='P(w):Prior Distribution over W'
plot2dGaussian(prior_mu,prior_cov,w0,w1,title)

# compute posterior for 1 data point
posterior_mean,posterior_cov=calPosterior(prior_mu,prior_cov,noise_mu,noise_cov,X_Label,T,100)
w0=np.arange(-2,2,0.1)
w1=np.arange(-2,2,0.1)
title='Poseterior after Observing one point'
plot2dGaussian(posterior_mean,posterior_cov,w0,w1,title)

# draw 5 samples from posterior and plot function

for i in range(5):
	w_samples = np.random.multivariate_normal(posterior_mean, posterior_cov, 5)
print('----------------')	
print(w_samples)
print('----------------')
p=np.arange(-1,1.01,.01)
plt.plot(p, 0.5*p-1.5)	
for i in range(5):
	# Plot line based on sampled parameters
	plt.plot(p, w_samples[i][1]*p + w_samples[i][0], color = "red", linestyle = "-", linewidth = 0.5)

plt.show()

# # compute posterior
# posterior_mean,posterior_cov=calPosterior(prior_mu,prior_cov,noise_mu,noise_cov,X_Label,T,100)
# w0=np.arange(-2,2,0.1)
# w1=np.arange(-2,2,0.1)
# title='Poseterior after Observing one point'
# plot2dGaussian(posterior_mean,posterior_cov,w0,w1,title)
