import numpy as np 
import scipy as sp
import scipy.optimize as opt
import matplotlib.pyplot as plt


def plotTrueFunction(X,N):
	plt.scatter(X[:,0],X[:,1])
	plt.title("True X and Number of Points="+str(N))
	plt.show()
def returnX(N):
	x =(np.linspace(0,4*np.pi,N))
	X=np.ones((N,2))
	X[:,0]=np.sin(x)-x*np.cos(x)
	X[:,1]=np.cos(x)+x*np.sin(x)
	return X
def f(W,*args):
	D=10
	N=args[0]
	sigma = 1
	W = np.reshape(W,(10,2))
	# print(W)
	# print('XXXXXXXXXXXXXXXXXXXX')
	C = np.matmul(W, np.transpose(W)) + ((sigma**2) * np.identity(D))
	# print(C)
	S=args[1]
	# print(S)
	val1=N*D*0.5*np.log(2*np.pi)
	val2=N*0.5*np.log(np.linalg.det(C))
	val3=np.trace(np.matmul(np.linalg.inv(C),S))
	val=val1+val2+val3
	# print(val)
	return val
def returnA():
	A=np.random.normal(0, 1, (10,2))
	return A
	
# A=np.ones((10,2))
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

random.seed(9001)
for N in [25,50,100,200,1000]:
	X=returnX(N)
	A=returnA()
	Y=np.dot(X,A.transpose())
	S=np.cov(Y,rowvar = False)
	args =(N,S)
	W = np.reshape(random.rand(20),(20,1))
	W=opt.fmin_cg(f, W_init, fprime = df, args = args)
	W_new = np.reshape(W,(10,2))
	WtW = np.dot(np.transpose(W_new),W_new)
	inv = np.linalg.pinv(WtW)
	plotTrueFunction(X,N)
	# X = np.dot(Y, np.dot(W_new,WtW))
	X_new = np.dot(Y, np.dot(W_new,WtW))*(np.pi**4)
	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	
	ax1.scatter(X_new[:,0],X_new[:,1],c='r')
	# ax1.scatter(X[:,0],X[:,1],c='b')

	# plt.scatter(X_new[:,0],X_new[:,1])
	# plt.title("Learned X and Number of Points="+str(N))
	plt.show()
