import numpy as np 
import scipy as sp
import scipy.optimize as opt
import matplotlib.pyplot as plt
import random

def plotTrueFunction(X,N,sigma):
	plt.scatter(X[:,0],X[:,1])
	plt.title("True X and Number of Points="+str(N)+" and $\sigma^2$="+str(sigma))
	plt.xlabel("$sin(x)-xcos(x)$")
	plt.ylabel("$cos(x)+xsin(x)$")
	plt.show()
def plotLearned(new_X,N,sigma):
	plt.scatter(new_X[:,0],new_X[:,1],c='r')
	plt.title("Learned X and Number of Points="+str(N)+" and $\sigma^2$="+str(sigma))
	plt.xlabel("$sin(x)-xcos(x)$")
	plt.ylabel("$cos(x)+xsin(x)$")
	plt.show()

def returnX(N):
	x =(np.linspace(0,4*np.pi,N))
	X=np.ones((N,2))
	X[:,0]=np.sin(x)-x*np.cos(x)
	X[:,1]=np.cos(x)+x*np.sin(x)
	# X=10*X
	return X
def f(W,*args):
	D=10
	N=args[0]
	sigma = args[2]
	W = np.reshape(W,(10,2))
	C = np.matmul(W, np.transpose(W)) + ((sigma**2) * np.identity(D))
	# print(C)
	S=args[1]
	# print(S)
	val1=N*D*0.5*np.log(2*np.pi)
	val2=N*0.5*np.log(np.linalg.det(C))
	val3=N*0.5*(np.trace(np.matmul(np.linalg.inv(C),S)))
	val=val1+val2+val3
	# print(val)
	return val
def returnA():
	A=np.random.normal(0, 1, (10,2))
	return A


A=np.ones((10,2))
def df(W, *args):
	N=args[0]
	D= 10
	sigma = args[2]
	W = np.reshape(W,(10,2))
	S=args[1]
	C = np.matmul(W, np.transpose(W)) + ((sigma**2) * np.identity(D))
	C_inv = np.linalg.inv(C)
	# df=np.ones((10,2))
	df = -N * (np.matmul(C_inv, np.matmul(S, np.matmul(C_inv, W)))- np.matmul(C_inv, W))
	return np.reshape(df, (20,))


for N in [50,200,1000,100,35]:
# for N in [50,100,200]:
	X=returnX(N)
	A=returnA()
	Y=np.matmul(X,A.transpose())
	for sigma in  [0.00001]:
	
		S=np.cov(Y,rowvar = False)
		# S=1/N*(np.matmul(np.transpose(Y-np.mean(Y,axis=0).reshape(-1,1)),(Y-np.mean(Y,axis=0).reshape(-1,1))))
		args =(N,S,sigma)
		# W = np.reshape(random.rand(20),(20,1))
		# W_init=np.asarray([np.random.normal(0,5*np.pi,20)]).transpose()
		W_init = np.asarray([np.random.normal(0,2, 1) for i in range(20)])
		# W_init=np.ones(20)
		W=opt.fmin_cg(f, W_init, fprime = df, args = args)
		W_new = np.reshape(W,(10,2))

		Wtr = np.transpose(W_new)
		M = np.dot(Wtr, W_new) + np.identity(2) * sigma
		Minv = np.linalg.inv(M)
		MinvWtr = np.matmul(Minv, Wtr)
		WtW = np.dot(np.transpose(W_new),W_new)
		inv = np.linalg.pinv(WtW)
		new_X = np.matmul(MinvWtr, np.transpose(Y)) 
		new_X = np.transpose(new_X)
		
		plotTrueFunction(X,N,sigma)
		plotLearned(new_X,N,sigma)

		plt.show()
