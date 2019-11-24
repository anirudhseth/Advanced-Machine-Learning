import pylab as pb
import numpy as np
from math import pi
import scipy
from scipy.spatial.distance import  cdist
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import random

def sqExpKernel(X,Y,length_scale):
	return np.exp(-cdist(X, Y, 'sqeuclidean')/(length_scale**2))

def data():
	X=np.array([-1.99,-1.1,-0.8,-0.38,0.03,0.89,1.2,1.7,2.1]).reshape(-1,1)
	# X=np.array([-2,-1.5,-1,-0.5,0,0.5,1,1.5,2])
	X=X*pi
	noise = np.random.normal(0, 0.2, X.shape[0]).reshape(-1,1)
	# noise = np.random.normal(0, 0.2, X.shape[0])
	# T=np.sin(X)+np.cos(X)+noise
	T=np.cos(X)**2
	return X,T
	
def plotGP(noise,length_scale,no_of_samples):
	X = np.linspace(-10.0,10.0,100) 
	X=X.reshape(-1,1)
	mu = np.zeros(X.shape[0])
	K =  noise*sqExpKernel(X,X,length_scale)
	Z = np.random.multivariate_normal(mu,K,no_of_samples)
	
	
	for i in range(no_of_samples):
		plt.plot(X[:],Z[i,:])
	plt.title('Length Scale:'+str(length_scale))
	plt.savefig('q10/q10_GP_'+str(length_scale)+'.png')
	plt.show()

# https://peterroelants.github.io/posts/gaussian-process-tutorial/
def GP_noise(X, T, X_star, σ_noise):
	length_scale=2
	# Kernel of the noisy observations
	Σ11 = exponentiated_quadratic(X, X,length_scale) + σ_noise * np.eye(X.shape[0])
	# Kernel of observations vs to-predict
	Σ12 = exponentiated_quadratic(X, X_star,length_scale)
	# Solve
	solved = scipy.linalg.solve(Σ11, Σ12, assume_a='pos').T
	# Compute posterior mean
	μ2 = solved @ T
	# Compute the posterior covariance
	Σ22 = exponentiated_quadratic(X_star, X_star,length_scale)
	Σ2 = Σ22 - (solved @ Σ12)
	return μ2, Σ2  # mean, covariance


# # plot gp
# for l in  [.01,.5,3,10]:
# 	plotGP(noise,l,no_of_samples)

X,T=data()
# X_star=np.array([[-4.28318531]])
X_star = np.linspace(-3*np.pi,3*np.pi, 500).reshape(-1,1)
pos_mean,pos_cov=GP_noise(X,T,X_star,0)
op=np.random.multivariate_normal(mean=pos_mean.flatten(), cov=pos_cov, size=9)
# plt.plot(X[:],op[:])
# print(pos_cov)
# print(pos_mean)
plt.plot(X, T,'ro')
# plt.plot(X_star,np.cos(X_star)**2, color = 'green',label='True Function')
plt.plot(X_star,pos_mean, color = 'blue',linewidth=0.8,label='GP Mean')
Z = np.random.multivariate_normal(np.reshape(pos_mean,(500,)),pos_cov,10)
for i in range(10):
	pb.plot(X_star[:],Z[i,:],linewidth=0.4)
# print(pos_cov.diagonal())
pos_mean=pos_mean.flatten()
upper = pos_mean + 2*np.sqrt(pos_cov.diagonal())
lower = pos_mean - 2*np.sqrt(pos_cov.diagonal())
ax = plt.gca()
plt.xlabel('Labels')
plt.ylabel('Predictions')
plt.legend()
print(pos_cov.diagonal().shape)
ax.fill_between(np.linspace(-3*np.pi, 3*np.pi, 500), upper, lower, facecolor='cyan', interpolate=True, alpha=0.2)