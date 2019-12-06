import itertools as it
from math import exp,sqrt,pi
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy import random, linalg

def plotPerformance(idx,result):
    plt.plot(result[idx,3],'g',linewidth=0.6, label= "P(D|$M_3$)")
    plt.plot(result[idx,2],'r', linewidth=0.3,label= "P(D|$M_2$)")
    plt.plot(result[idx,1],'b', linewidth=0.3,label= "P(D|$M_1$)")
    plt.plot(result[idx,0],'m--',linewidth=0.3, label ="P(D|$M_0$)")
    plt.xlabel('All data sets, D')
    plt.ylabel('Evidence')
    plt.legend()
    plt.show()
def plotPerformance_max(idx,result):
    plt.plot(result[idx,3],'g',linewidth=0.6, label= "P(D|$M_3$)")
    plt.plot(result[idx,2],'r', linewidth=0.3,label= "P(D|$M_2$)")
    plt.plot(result[idx,1],'b', linewidth=0.3,label= "P(D|$M_1$)")
    plt.plot(result[idx,0],'m--',linewidth=0.3, label ="P(D|$M_0$)")
    plt.xlabel('Subset of possible data sets, D')
    plt.ylabel('Evidence')
    plt.xlim(0,80)
    plt.legend()
    plt.show()
def indexAlgo(result):
    dist = distance.squareform(distance.pdist(result.reshape(-1,1), 'euclidean'))
    np.fill_diagonal(dist, np.inf)
    L = []  
    D = list(range(result.shape[0]))
    L.append(result.argmin())
    D.remove(L[-1])
    
    while len(D) > 0:
        # add d if dist from d to all other points in D
        # is larger than dist from d to L[-1]
        N = [d for d in D if dist[d, D].min() > dist[d, L[-1]]]
        
        if len(N) == 0:
            L.append(D[dist[L[-1],D].argmin()])
        else:
            L.append(N[dist[L[-1],N].argmax()])
        
        D.remove(L[-1])
    
    # reverse the resulting index array
    return np.array(L)[::-1]
def generateD():
    x=list(it.product([-1,1], repeat=9))
    D = []
    for d in x:
        D.append(np.reshape(np.asarray(d), (3, 3)))
    return D 
def visualizeData(arr):
    for i in range(3):
        for j in range(3):
            if arr[i][j] == -1:
                print("O", end=" ")
            else:
                print("X", end=" ")
        print()
    print()   
def modelsCal(model,theta,t):
    p=1
 
    if(model==0):
        p= 1/512
    elif(model==1):
        for i in range(3):
            for j in range(3):
                e = np.exp(-t[i, j]*(theta[0]*(j-1)))
                p = p * 1/(1+e)
    elif(model==2):
        for i in range(3):
          for j in range(3):
                e = np.exp(-t[i, j]*(theta[0]*(j-1) + theta[1]*(1-i)))
                p = p * 1/(1+e)
    elif(model==3):
        for i in range(3):
          for j in range(3):
                e = np.exp(-t[i, j]*(theta[0]*(j-1) + theta[1]*(1-i)+theta[2]))
                p = p * 1/(1+e)

    return p
def getTheta(model,S):
    theta=[]
    if(model==0):
        return theta
    else:
        mu=np.zeros(model)
        cov=np.eye(model)*1000
        return np.random.multivariate_normal(mu, cov, S)
def getThetaComplex(model,S):
    theta=[]
    if(model==0):
        return theta
    else:
        mu=np.ones(model)*5
        A = random.rand(model,model)
        cov = np.dot(A,A.transpose())*1000
        return np.random.multivariate_normal(mu, cov, S)
def computeEvidence(model,sample,data,theta):
    p=0
    for i in range(sample):
        p+=modelsCal(model,theta[i],data)
    return p/sample


S=1000
thetaM1=getTheta(1,S)
thetaM2=getTheta(2,S)
thetaM3=getTheta(3,S)
D=generateD()
result=np.zeros([512,4])


for m in range(4):
    for i in range(512):
        if (m==0):
            result[i][m]=1/512
        elif (m==1):
            result[i][m]=computeEvidence(m,S,D[i],thetaM1)
        elif (m==2):
            result[i][m]=computeEvidence(m,S,D[i],thetaM2)
        elif (m==3):
            result[i][m]=computeEvidence(m,S,D[i],thetaM3)  
sum_evid=np.sum(result,axis=0)
print('Samples used:'+str(S))
print('***Sum of the Evidence***')
for i in range(len(sum_evid)):
    print('Model '+str(i)+' :'+str(sum[i]))
max = np.argmax(result,axis=0)
min = np.argmin(result,axis=0)
print()
# print(max)
for i in range(4):
    print('Best Performance by Model: ' +str(i))
    print('Probability Mass is: ' +str(result[max[i]][i]))
    visualizeData(D[max[i]])
    print('Worst Performance by Model: ' +str(i))
    print('Probability Mass is: ' +str(result[min[i]][i]))
    visualizeData(D[min[i]])
       
idx=indexAlgo(np.sum(result,axis=1))
plotPerformance(idx,result)
plotPerformance_max(idx,result)





