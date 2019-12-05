import numpy as np 
import scipy as sp
import scipy.optimize as opt
import matplotlib.pyplot as plt
import random

def plotTrueFunction(X,N):
	plt.scatter(X[:,0],X[:,1])
	plt.title("True X and Number of Points="+str(N))
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
	sigma = 1
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
	sigma = 1
	W = np.reshape(W,(10,2))
	S=args[1]
	C = np.matmul(W, np.transpose(W)) + ((sigma**2) * np.identity(D))
	C_inv = np.linalg.inv(C)
	# df=np.ones((10,2))
	df = -N * (np.matmul(C_inv, np.matmul(S, np.matmul(C_inv, W)))- np.matmul(C_inv, W))
	return np.reshape(df, (20,))

# random.seed(9340247)
for N in [200]:
# for N in [50,100,200]:
	X=returnX(N)
	A=returnA()
	Y=np.matmul(X,A.transpose())
	
	S=np.cov(Y,rowvar = False)
	args =(N,S)
	# W = np.reshape(random.rand(20),(20,1))
	# W_init=np.asarray([np.random.normal(0,5*np.pi,20)]).transpose()
	W_init = np.asarray([np.random.normal(0,2, 1) for i in range(20)])
	# W_init=np.ones(20)
	W=opt.fmin_cg(f, W_init, fprime = df, args = args)
	W_new = np.reshape(W,(10,2))
	WtW = np.dot(np.transpose(W_new),W_new)
	inv = np.linalg.pinv(WtW)
	plotTrueFunction(X,N)
	# X = np.dot(Y, np.dot(W_new,WtW))*(np.pi**4)
	X_new = np.dot(Y, np.matmul(W_new,WtW))
	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	Y_new=np.matmul(X_new,A.transpose())
	ax1.scatter(X_new[:,0],X_new[:,1],c='r')
	# ax1.scatter(X[:,0],X[:,1],c='b')

	# plt.scatter(X_new[:,0],X_new[:,1])
	# plt.title("Learned X and Number of Points="+str(N))
	plt.show()
