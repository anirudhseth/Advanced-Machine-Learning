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
	# X=np.array([-1.99,-1.1,-0.8,-0.38,0.03,0.89,1.2,1.7,2.1]).reshape(-1,1)
	# X=np.array([-2,-1.5,-1,-0.5,0,0.5,1,1.5,2])
	X=np.array([-4,-3,-2,-1,0,2,3,5]).reshape(-1,1)
	
	noise = np.random.normal(0, 3, X.shape[0]).reshape(-1,1)
	# noise = np.random.normal(0, 0.2, X.shape[0])
	# T=np.sin(X)+np.cos(X)+noise
	T=(2+(0.5*X-1)**2)*np.sin(3*X)+noise

	return X,T
	
def plotGP():
	noise=1
	no_of_samples=10
	for l in [.01,.5,3,10]:
		X = np.linspace(-10.0,10.0,100) 
		X=X.reshape(-1,1)
		mu = np.zeros(X.shape[0])
		K =  noise*sqExpKernel(X,X,l)
		Z = np.random.multivariate_normal(mu,K,no_of_samples)
		for i in range(no_of_samples):
			plt.plot(X[:],Z[i,:])
		plt.title('Length Scale:'+str(l))
		# plt.savefig('q10/q10_GP_'+str(l)+'.png')
		plt.show()

# https://peterroelants.github.io/posts/gaussian-process-tutorial/
def GP_noise(X, T, X_star, σ_noise,length_scale):
	# Kernel of the noisy observations
	Σ11 = sqExpKernel(X, X,length_scale) + σ_noise * np.eye(X.shape[0])
	# Kernel of observations vs to-predict
	Σ12 = sqExpKernel(X, X_star,length_scale)
	# Solve
	solved = scipy.linalg.solve(Σ11, Σ12, assume_a='pos').T
	# Compute posterior mean
	μ2 = solved @ T
	# Compute the posterior covariance
	Σ22 = sqExpKernel(X_star, X_star,length_scale)
	Σ2 = Σ22 - (solved @ Σ12)
	return μ2, Σ2  # mean, covariance

def plot3(X,T,X_star,pos_mean,pos_cov,σ_noise,length_scale):
	plt.figure()
	plt.plot(X, T,'ro')
	plt.plot(X_star,(2+(0.5*X_star-1)**2)*np.sin(3*X_star), linewidth=0.2,color = 'green',label='True Function')
	plt.plot(X_star,pos_mean, color = 'blue',linewidth=0.8,label='GP Mean')
	for i in range(5):
		pb.plot(X_star[:],Z[i,:],linewidth=0.4)
	pos_mean=pos_mean.flatten()
	upper = pos_mean + 2*np.sqrt(pos_cov.diagonal())
	lower = pos_mean - 2*np.sqrt(pos_cov.diagonal())
	ax = plt.gca()
	plt.xlabel('Labels')
	plt.ylabel('Predictions')
	plt.legend()
	plt.title("Length Scale="+str(length_scale)+" and σ_noise="+str(σ_noise))
	plt.xlim(-10,10)
	plt.ylim(-18,20)
	ax.fill_between(np.linspace(-8, 8, 1000), upper, lower, facecolor='cyan', interpolate=True, alpha=0.2)
def plot1(X,X_star,σ_noise,length_scale):
	plt.figure()
	plt.plot(X, T,'ro')
	for i in range(10):
		pb.plot(X_star[:],Z[i,:],linewidth=0.4)
	ax = plt.gca()
	plt.xlabel('Labels')
	plt.xlim(-10,10)
	plt.ylim(-18,20)

	plt.title("Length Scale="+str(length_scale)+" and σ_noise="+str(σ_noise))
	plt.ylabel('Predictions')
	# plt.legend()
def plot2(X,T,X_star,pos_mean,pos_cov,σ_noise,length_scale):
	plt.figure()
	plt.plot(X, T,'ro')
	plt.plot(X_star,(2+(0.5*X_star-1)**2)*np.sin(3*X_star), linewidth=0.2,color = 'green',label='True Function')
	plt.plot(X_star,pos_mean, color = 'blue',linewidth=0.8,label='GP Mean')
	ax = plt.gca()
	plt.xlabel('Labels')
	plt.ylabel('Predictions')
	plt.xlim(-10,10)
	plt.ylim(-18,20)
	plt.title("Length Scale="+str(length_scale)+" and σ_noise="+str(σ_noise))
	plt.legend()
def plot4(X,T,X_star,pos_mean,pos_cov,σ_noise,length_scale):
	plt.figure()
	plt.plot(X, T,'ro')
	plt.plot(X_star,(2+(0.5*X_star-1)**2)*np.sin(3*X_star), color = 'green',label='True Function')
	plt.plot(X_star,pos_mean, color = 'blue',linewidth=0.8,label='GP Mean')
	for i in range(10):
		pb.plot(X_star[:],Z[i,:],linewidth=0.4)
	pos_mean=pos_mean.flatten()
	upper = pos_mean + 2*np.sqrt(pos_cov.diagonal())
	lower = pos_mean - 2*np.sqrt(pos_cov.diagonal())
	ax = plt.gca()
	plt.xlim(-10,10)
	plt.ylim(-18,20)
	plt.xlabel('Labels')
	plt.ylabel('Predictions')
	plt.title("Length Scale="+str(length_scale)+" and σ_noise="+str(σ_noise))
	plt.legend()
	ax.fill_between(np.linspace(-8, 8, 1000), upper, lower, facecolor='cyan', interpolate=True, alpha=0.2)
def plot5(X,T,X_star,pos_mean,pos_cov,σ_noise,length_scale):
	plt.figure()
	plt.plot(X, T,'ro')
	plt.plot(X_star,(2+(0.5*X_star-1)**2)*np.sin(3*X_star), color = 'green',label='True Function')
	plt.plot(X_star,pos_mean, color = 'blue',linewidth=0.8,label='GP Mean')
	for i in range(10):
		pb.plot(X_star[:],Z[i,:],linewidth=0.4)
	pos_mean=pos_mean.flatten()
	upper = pos_mean + 2*np.sqrt(pos_cov.diagonal())
	lower = pos_mean - 2*np.sqrt(pos_cov.diagonal())
	ax = plt.gca()
	plt.xlabel('Labels')
	plt.ylabel('Predictions')
	plt.title("Length Scale="+str(length_scale)+" and σ_noise="+str(σ_noise))
	plt.legend()
	plt.xlim(-10,10)
	plt.ylim(-18,20)
	ax.fill_between(np.linspace(-8, 8, 1000), upper, lower, facecolor='cyan', interpolate=True, alpha=0.2)




for length_scale in [2]:
	for σ_noise in [0,.3]:
		X,T=data()
		X_star = np.linspace(-8,8,1000).reshape(-1,1)
		pos_mean,pos_cov=GP_noise(X,T,X_star,σ_noise,length_scale)
		op=np.random.multivariate_normal(mean=pos_mean.flatten(), cov=pos_cov, size=9)
		Z = np.random.multivariate_normal(np.reshape(pos_mean,(1000,)),pos_cov,10)


		# plotGP()
		plot1(X,X_star,σ_noise,length_scale)
		plot2(X,T,X_star,pos_mean,pos_cov,σ_noise,length_scale)
		plot3(X,T,X_star,pos_mean,pos_cov,σ_noise,length_scale)
		plot4(X,T,X_star,pos_mean,pos_cov,σ_noise,length_scale)
		# plot5(X,T,X_star,pos_mean,pos_cov,σ_noise,length_scale)
