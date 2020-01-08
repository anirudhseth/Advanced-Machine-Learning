import matplotlib.pyplot as plot
from scipy.stats import norm,gamma
import numpy as np

def generateData(N):
    D = np.random.rand(N)
    D = (D - np.mean(D))
    D = D/(np.var(D)**0.5)
    return D

def VIPosterior(a_N,b_N,mu_N,lambda_N,mu,tau):
    mean_mu =mu_N
    var_mu = 1/lambda_N
    mu_pdf = norm(mean_mu, var_mu).pdf(mu)
    tao_pdf = gamma(a_N, loc=0, scale=1/b_N).pdf(tau)
    return mu_pdf*tao_pdf

"Exact Posterior ref:https://en.wikipedia.org/wiki/Normal-gamma_distribution"
def ExactPosterior(a_0,b_0,mu0,lambda0,mu,tau,x,N):
    mu_mean = (mu0*l_0 + np.sum(x))/(lambda0+N)
    mu_var = 1/((lambda0+N)*tau)
    mu_pdf = norm(mu_mean, mu_var).pdf(mu)
    a_tr = a_0 + 0.5*N
    b_tr = b_0 + 0.5*lambda0*mu0*mu0 + 0.5*np.sum(x**2)
    tao_pdf = gamma(a_tr, loc=0, scale=1/b_tr).pdf(tau)
    return mu_pdf*tao_pdf

def getB(b_0,x2,l_0,mew_0,N,mew_N,l_N,x):
    term1=b_0 + 0.5*(x2+l_0*mew_0*mew_0)
    term2=0.5*(N+l_0)*(mew_N*mew_N+1/l_N)
    term3=-(x+l_0*mew_0)*mew_N
    return term1+term2+term3

div = 100
m=np.linspace(-1,1,div)
t=np.linspace(0.01,3,div)
[xx,yy] = np.meshgrid(m,t)
N = 10
iter = 10
D= generateData(N)

"setting mu_0,lamba_0,a_0,b_0"
a_0 = 0
b_0 = 0
l_0 = 0
mew_0 = 0


l_N=np.zeros(iter)
b_N=np.zeros(iter)

# data 1 #
# a_N_plot=2
# a_N=5
# b_N[0]=np.array([1])
# l_N[0]=np.array([5])
# mew_N=.5
# mew_N_plot=.5

# data 2 #
a_N_plot=7
a_N=7
b_N[0]=np.array([8])
l_N[0]=np.array([20])
mew_N=0
mew_N_plot=0

# data 3 #
#change N=20 and xlim and ylim for better plot
# a_N_plot=5
# a_N=5
# b_N[0]=np.array([5])
# l_N[0]=np.array([0.5])
# mew_N=0.5
# mew_N_plot=0.5

print('True Parameters  [a,b,lambda,mu]:',[a_0,b_0,l_0,mew_0])
print('VI Parameters    [a,b,lambda,mu]:',[a_N,b_N[0],l_N[0],mew_N])

for i in range(iter):
    if(i==0):
        continue
    a_N = a_0 + (N+1)/2
    b_N[i] = getB(b_0,np.sum(D**2),l_0,mew_0,N,mew_N,l_N[i-1],np.sum(D))
    mew_N = (l_0*mew_0 + N* np.mean(D))/(l_0 + N)
    l_N[i] = ((l_0+N)*a_N/b_N[i])


"True posterior"
Plot_ExactPosterior = np.zeros((div,div),dtype=float)
for i in range(div):
    for j in range(div):
        Plot_ExactPosterior[i,j]=ExactPosterior(a_0,b_0,mew_0,l_0,m[j],t[i],D,N)

"Posterior before running VI"
Posterior_BeforeVI = np.zeros((div,div),dtype=float)
for i in range(div):
    for j in range(div):
        Posterior_BeforeVI[i,j]=VIPosterior(a_N_plot,b_N[0],mew_N_plot,l_N[0],m[j],t[i])

"Posterior after running VI"
VIPosterior_MaxIter = np.zeros((div,div),dtype=float)
for i in range(div):
    for j in range(div):
        VIPosterior_MaxIter[i,j]=VIPosterior(a_N,b_N[iter-1],mew_N,l_N[iter-1],m[j],t[i])

plot1 = plot.contour(xx,yy,Plot_ExactPosterior,colors='g')
plot2 = plot.contour(xx,yy,Posterior_BeforeVI,colors='b')
plot.title('Posterior before running VI')
plot.xlim(-.5,.5)
plot.xlabel('$\mu$')
plot.ylabel('tao')
plot.show()

plot1 = plot.contour(xx,yy,Plot_ExactPosterior,colors='g')
plot2 = plot.contour(xx,yy,VIPosterior_MaxIter,colors='b')
plot.title('Posterior after running VI for '+str(iter)+' iterations.')
plot.xlabel('$\mu$')
plot.xlim(-.5,.5)
plot.ylabel('tao')
plot.show()

