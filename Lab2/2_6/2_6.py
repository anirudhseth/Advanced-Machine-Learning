import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import multivariate_normal, poisson


def generate_data(n_data, means, covariances, weights, rates):
    n_clusters, n_features = means.shape
    data = np.zeros((n_data, n_features))
    poission_data = np.zeros(n_data)
    colors = np.zeros(n_data, dtype='str')
    for i in range(n_data):
        # pick a cluster id and create data from this cluster
        k = np.random.choice(n_clusters, size=1, p=weights)[0]
        x = np.random.multivariate_normal(means[k], covariances[k])
        data[i] = x
        poission_data[i] = np.random.poisson(rates[k])
        if k == 0:
            colors[i] = 'red'
        elif k == 1:
            colors[i] = 'blue'
        elif k == 2:
            colors[i] = 'green'

    return data, poission_data, colors


# means, covs: means and covariances of Gaussians
# rates: rates of Poissons
# title: title of the plot defining which EM iteration
def plot_contours(X, S, means, covs, title, rates):
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], s=S)

    delta = 0.025
    k = means.shape[0]
    x = np.arange(-2.0, 7.0, delta)
    y = np.arange(-2.0, 7.0, delta)
    X, Y = np.meshgrid(x, y)
    col = ['green', 'red', 'indigo']
    for i in range(k):
        mean = means[i]
        cov = covs[i]
        sigmax = np.sqrt(cov[0][0])
        sigmay = np.sqrt(cov[1][1])
        sigmaxy = cov[0][1] / (sigmax * sigmay)
        pos=np.dstack((X,Y))
        rv=multivariate_normal(mean,cov)
        plt.contour(X, Y, rv.pdf(pos), colors=col[i], linewidths=rates[i], alpha=0.1)

    plt.title(title)
    plt.tight_layout()


class EM:

    def __init__(self, n_components, n_iter, tol, seed):
        self.n_components = n_components
        self.n_iter = n_iter
        self.tol = tol
        self.seed = seed

    def fit(self, X, S):

        # data's dimensionality
        self.n_row, self.n_col = X.shape

        # initialize parameters
        np.random.seed(self.seed)
        chosen = np.random.choice(self.n_row, self.n_components, replace=False)
        self.means = X[chosen]
        self.weights = np.full(self.n_components, 1 / self.n_components)
        if self.n_components == 3:
            self.rates = (np.mean(S) * np.ones(self.n_components) / np.array([1, 2, 3])[np.newaxis]).flatten()
        elif self.n_components == 2:
            self.rates = (np.mean(S) * np.ones(self.n_components) / np.array([1, 2])[np.newaxis]).flatten()
        shape = self.n_components, self.n_col, self.n_col
        self.covs = np.full(shape, np.cov(X, rowvar=False))
        new_covs = []
        for c in self.covs:
            new_covs = np.append(new_covs, np.diag(np.diag(c))) # making the covariances diagonal (question assumption)
        self.covs = np.array(new_covs).reshape(self.n_components, 2, 2)
        self.r=np.zeros([X.shape[0],self.n_components])
        self.loglikeplot=[]
    
        log_likelihood = -np.inf
        self.converged = False

        for i in range(self.n_iter):
            self._do_estep(X, S)
            self._do_mstep(X, S)
            
            log_likelihood_new = self._compute_log_likelihood(X, S)
            self.loglikeplot.append(log_likelihood_new)
            if (log_likelihood_new-log_likelihood  ) <= self.tol:
                self.converged = True
                break
            log_likelihood = log_likelihood_new

        return self
    
    def _do_estep(self, X, S):
        """
        E-step
        """
        N=X.shape[0]
        K=self.n_components
        r=np.zeros([N,K])
        for n in range(N):
            norm=[multivariate_normal(self.means[j],self.covs[j]) for j in range(K)]
            pois =[poisson(self.rates[j]) for j in range(K)]
            for k in range(K):
                r[n][k]=self.weights[k]*norm[k].pdf(X[n])*pois[k].pmf(S[n])
            r[n,:] = r[n,:]/np.sum(r[n,:])
        self.r=r
        return self

    def _do_mstep(self, X, S):
        """M-step, update parameters"""
        N=X.shape[0]
        K=self.n_components
        for k in range(K):
            self.weights[k]=np.sum(self.r[:,k])/N
            temp=self.r[:,k].reshape(-1,1)
            # print(temp)
            self.means[k]=np.sum(temp*X,axis =0) / np.sum(self.r[:,k])
            # qw=np.sum(temp*np.square(X-self.means[k]),axis=0)/np.sum(self.r[:,k])
            # print(self.means[k])
            qw=(np.dot((X-self.means[k]).T,temp*(X-self.means[k])))/ np.sum(self.r[:,k])
            # print(qw)
            qw[0][1]=0
            qw[1][0]=0
            self.covs[k]=qw
            # self.covs[k][0][0]=qw[0]
            # self.covs[k][1][1]=qw[1]
            self.rates[k]=np.sum(temp*S.reshape(-1,1))/np.sum(self.r[:,k])
        return self

    def _compute_log_likelihood(self, X, S):
        """compute the log likelihood of the current parameter"""
        N=X.shape[0]
        K=self.n_components
        log_likelihood = 0
        for n in range(N):
            tmp=0
            norm=[multivariate_normal(self.means[j],self.covs[j]) for j in range(K)]
            pois =[poisson(self.rates[j]) for j in range(K)]
            for k in range(K):
                tmp+=self.weights[k]*norm[k].pdf(X[n])*pois[k].pmf(S[n])
            log_likelihood+=np.log(tmp)

        return log_likelihood

###############################################################################################

# params for 3 clusters
means = np.array([
    [5, 0],
    [1, 1],
    [0, 5]
])

covariances = np.array([
    [[.5, 0.], [0, .5]],
    [[.92, 0], [0, .91]],
    [[.5, 0.], [0, .5]]
])

weights = [1 / 4, 1 / 2, 1 / 4]

# params for 2 clusters
means_2 = np.array([
    [5, 0],
    [1, 1]
])

covariances_2 = np.array([
    [[.5, 0.], [0, .5]],
    [[.92, 0], [0, .91]]
])

weights_2 = [1 / 4, 3 / 4]
seed=3
np.random.seed(seed)

rates = np.random.uniform(low=.2, high=20, size=3)
print("Poisson rates for 3 components:")
print(rates)

rates_2 = np.random.uniform(low=.2, high=20, size=2)
print("Poisson rates for 2 components:")
print(rates_2)

# generate data
X, S, colors = generate_data(100, means, covariances, weights, rates)
plt.scatter(X[:, 0], X[:, 1], s=S, c=colors) # the Poisson data is shown through size of the points: s
plt.show()

X_2, S_2, colors_2 = generate_data(100, means_2, covariances_2, weights_2, rates_2)
plt.scatter(X_2[:, 0], X_2[:, 1], s=S_2, c=colors_2) # the Poisson data is shown through size of the points: s
plt.show()

###############################################################################################

em = EM(n_components=3, n_iter=1, tol=1e-4, seed=1)
em.fit(X, S)
# plot: call plot_contours and give it the params updated from EM with 3 components (after 1 iteration)
plot_contours(X,S,em.means,em.covs,'Params updated from EM with 3 components (after 1 iteration)',em.rates)
plt.show()

em = EM(n_components=3, n_iter=50, tol=1e-4, seed=1)
em.fit(X, S)
print(em.loglikeplot)
# plot: call plot_contours and give it the params updated from EM with 3 components (after 50 iterations)
plot_contours(X,S,em.means,em.covs,'Params updated from EM with 3 components (after 50 iterations)',em.rates,)
plt.show()
plt.plot(em.loglikeplot)
plt.ylabel("Log-Likelihood")
plt.title('Log-Likelihood for 3 Components Data')
plt.xlabel("Iterations")
plt.show()

em_2 = EM(n_components=2, n_iter=1, tol=1e-4, seed=1)
em_2.fit(X_2, S_2)
plot_contours(X_2,S_2,em_2.means,em_2.covs,'Params updated from EM with 2 components (after 1 iteration)',em_2.rates)
# plot: call plot_contours and give it the params updated from EM with 2 components (after 1 iteration)
plt.show()


em_2 = EM(n_components=2, n_iter=50, tol=1e-4, seed=1)
em_2.fit(X_2, S_2)
plot_contours(X_2,S_2,em_2.means,em_2.covs,'Params updated from EM with 2 components (after 50 iterations)',em_2.rates)
plt.show()
plt.plot(em_2.loglikeplot)
plt.title('Log-Likelihood for 2 Components Data')
plt.ylabel("Log-Likelihood")
plt.xlabel("Iterations")

# plot: call plot_contours and give it the params updated from EM with 2 components (after 50 iterations)
plt.show()



