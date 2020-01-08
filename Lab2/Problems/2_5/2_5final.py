""" This file is created as a template for question 2.5 in DD2434 - Assignment 2.

    Please keep the fixed parameters in the function templates as is (in 2_5.py file).
    However if you need, you can add parameters as default parameters.
    i.e.
    Function template: def em_algorithm(seed_val, samples, k, num_iter=10):
    You can change it to: def em_algorithm(seed_val, samples, k, num_iter=10, new_param_1=[], new_param_2=123):

    You can write helper functions however you want.

    You do not have to implement the code for finding a maximum spanning tree from scratch. We provided two different
    implementations of Kruskal's algorithm and modified them to return maximum spanning trees as well as the minimum
    spanning trees. However, it will be beneficial for you to try and implement it. You can also use another
    implementation of maximum spanning tree algorithm, just do not forget to reference the source (both in your code
    and in your report)!

    We also provided an example regarding the Robinson-Foulds metric (see Phylogeny.py).

    If you want, you can use the class structures provided to you (Node, Tree and TreeMixture classes in Tree.py file),
    and modify them as needed. In addition to the sample files given to you, it is very important for you to test your
    algorithm with your own simulated data for various cases and analyse the results.

    For those who do not want to use the provided structures, we also saved the properties of trees in .txt and .npy
    format.

    Note that the sample files are tab delimited with binary values (0 or 1) in it.
    Each row corresponds to a different sample, ranging from 0, ..., N-1
    and each column corresponds to a vertex from 0, ..., V-1 where vertex 0 is the root.
    Example file format (with 5 samples and 4 nodes):
    1   0   1   0
    1   0   1   0
    1   0   0   0
    0   0   1   1
    0   0   1   1

    Also, I am aware that the file names and their extensions are not well-formed, especially in Tree.py file
    (i.e example_tree_mixture.pkl_samples.txt). I wanted to keep the template codes as simple as possible.
    You can change the file names however you want (i.e tmm_1_samples.txt).

    After all, we will test your code with commands like this one:
    %run 2_5.py "data/example_tree_mixture.pkl_samples.txt" "data/example_result" 3 --seed_val 123
    where
    "data/example_tree_mixture.pkl_samples.txt" is the filename of the samples
    "data/example_result" is the base filename of results (i.e data/example_result_em_loglikelihood.npy)
    3 is the number of clusters for EM algorithm
    --seed_val is the seed value for your code, for reproducibility.

    For this assignment, we gave you three different trees
    (q_2_5_tm_10node_20sample_4clusters, q_2_5_tm_10node_50sample_4clusters, q_2_5_tm_20node_20sample_4clusters).
    As the names indicate, the mixtures have 4 clusters with varying number of nodes and samples.
    We want you to run your EM algorithm and compare the real and inferred results in terms of Robinson-Foulds metric
    and the likelihoods.
    """
import argparse
import numpy as np
import matplotlib.pyplot as plt
import dendropy
from Tree import TreeMixture, Tree
import sys
from dendropy.calculate.treecompare import symmetric_difference as RfDist

def save_results(loglikelihood, topology_array, theta_array, filename):
    """ This function saves the log-likelihood vs iteration values,
        the final tree structure and theta array to corresponding numpy arrays. """

    likelihood_filename = filename + "_em_loglikelihood.npy"
    topology_array_filename = filename + "_em_topology.npy"
    theta_array_filename = filename + "_em_theta.npy"
    print("Saving log-likelihood to ", likelihood_filename, ", topology_array to: ", topology_array_filename,
          ", theta_array to: ", theta_array_filename, "...")
    np.save(likelihood_filename, loglikelihood)
    np.save(topology_array_filename, topology_array)
    np.save(theta_array_filename, theta_array)

def em_algorithm(seed_val, samples, num_clusters, max_num_iter=100):
    """
    This function is for the EM algorithm.
    :param seed_val: Seed value for reproducibility. Type: int
    :param samples: Observed x values. Type: numpy array. Dimensions: (num_samples, num_nodes)
    :param num_clusters: Number of clusters. Type: int
    :param max_num_iter: Maximum number of EM iterations. Type: int
    :return: loglikelihood: Array of log-likelihood of each EM iteration. Type: numpy array.
                Dimensions: (num_iterations, ) Note: num_iterations does not have to be equal to max_num_iter.
    :return: topology_list: A list of tree topologies. Type: numpy array. Dimensions: (num_clusters, num_nodes)
    :return: theta_list: A list of tree CPDs. Type: numpy array. Dimensions: (num_clusters, num_nodes, 2)

    You can change the function signature and add new parameters. Add them as parameters with some default values.
    i.e.
    Function template: def em_algorithm(seed_val, samples, k, max_num_iter=10):
    You can change it to: def em_algorithm(seed_val, samples, k, max_num_iter=10, new_param_1=[], new_param_2=123):
    """

    # Set the seed
    np.random.seed(seed_val)

    # TODO: Implement EM algorithm here.

    # Start: Example Code Segment. Delete this segment completely before you implement the algorithm.
    print("Running EM algorithm...")

    
    from Kruskal_v1 import Graph
    # return result in the method
    import sys

    tm = TreeMixture(num_clusters=num_clusters, num_nodes=samples.shape[1])
    tm.simulate_pi(seed_val=seed_val)
    tm.simulate_trees(seed_val=seed_val)
    tm.sample_mixtures(num_samples=samples.shape[0], seed_val=seed_val)
    eps=sys.float_info.min
    topology_list = []
    theta_list = []
    loglikelihood = []
    num_samples = samples.shape[0]
    num_nodes = samples.shape[1]
    for iter in range(max_num_iter):
        r = np.ones((num_samples, num_clusters))
        for i, sample in enumerate(samples):
            for j, t in enumerate(tm.clusters):
                visitedNodes = [t.root]
                r[i,j] *= tm.pi[j]
                while len(visitedNodes) != 0:
                    presentNode = visitedNodes[0]
                    visitedNodes = visitedNodes[1:]
                    if len(presentNode.descendants) != 0:
                        visitedNodes = visitedNodes + presentNode.descendants
                    if presentNode.ancestor == None:   #root node
                        r[i,j] *= presentNode.cat[sample[int(presentNode.name)]]
                    else:
                        r[i,j] *= presentNode.cat[sample[int(presentNode.ancestor.name)]][sample[int(presentNode.name)]] 

        r += eps
        rn = np.sum(r, axis=1).reshape(num_samples,1)
        r /= rn
        loglikelihood.append(np.sum(np.log(rn)))       
        tm.pi = np.sum(r,axis=0)/num_samples
        den = np.sum(r,axis=0)
        NominatorQk = np.zeros((num_nodes, num_nodes, 2, 2, num_clusters))
        for s in range(num_nodes):
            for t in range(num_nodes):
                for a in range(2):
                    for b in range(2):
                        matched_index = np.where((samples[:,(s,t)] == [a,b]).all(1))[0]
                        NominatorQk[s,t,a,b] = np.sum(r[matched_index], axis = 0)/den
        
        DenominatorQk = np.zeros((num_nodes, 2, num_clusters))
        for s in range(num_nodes):
            for a in range(2):
                matched_index = np.where((samples[:,s] == a))
                DenominatorQk[s,a] = np.sum(r[matched_index], axis = 0)/den
                
        Iqst = np.zeros((num_nodes, num_nodes, num_clusters))
        for s in range(num_nodes):
            for t in range(num_nodes):
                for a in range(2):
                    for b in range(2):
                        if (np.all(NominatorQk[s,t,a,b,:] > 0) ):
                            Iqst[s, t] += NominatorQk[s,t,a,b] * np.log((NominatorQk[s,t,a,b]/(DenominatorQk[s,a])) / DenominatorQk[t,b])
                        else:
                            Iqst[s,t] += 0             
        for k in range(num_clusters):
            g = Graph(num_nodes)
            for s in range(num_nodes):
                for t in range(s+1, num_nodes):
                    g.addEdge(s, t, Iqst[s, t, k])
                    
            mst_edges = np.array(g.maximum_spanning_tree())[:,[0,1]]
            topology_array = np.zeros(num_nodes)
            topology_array[0] = np.nan
            visitedNodes = [0]
            while len(visitedNodes) != 0:
                presentNode = visitedNodes[0]
                visitedNodes = visitedNodes[1:]   
                child_edges = np.array(np.where(mst_edges == [presentNode])).T
                for ind in child_edges:
                    child = mst_edges[ind[0]][1 - ind[1]]
                    topology_array[int(child)] = presentNode
                    visitedNodes.append(child)
                if np.size(child_edges) != 0:
                    mst_edges = np.delete(mst_edges, child_edges[:,0], 0)
            
            new_tree = Tree()
            new_tree.load_tree_from_direct_arrays(topology_array)
            new_tree.alpha = [1.0] * 2
            new_tree.k = 2
            
            visitedNodes = [new_tree.root]
            while len(visitedNodes) != 0:
                presentNode = visitedNodes[0]
                visitedNodes = visitedNodes[1:]
                    
                if len(presentNode.descendants) != 0:
                    visitedNodes = visitedNodes + presentNode.descendants
                    
                if presentNode.ancestor == None:
                    presentNode.cat = DenominatorQk[int(presentNode.name),:,k].tolist()
                else:
                    presentNode.cat = NominatorQk[int(presentNode.ancestor.name), int(presentNode.name),:,:,k]
                    presentNode.cat[0]=presentNode.cat[0]/np.sum(presentNode.cat[0])
                    presentNode.cat[1]=presentNode.cat[1]/np.sum(presentNode.cat[1])
                    presentNode.cat = [presentNode.cat[0], presentNode.cat[1]]

            tm.clusters[k] = new_tree

        for j, t in enumerate(tm.clusters):
            topology_list.append(t.get_topology_array())
            theta_list.append(t.get_theta_array())
    loglikelihood = np.array(loglikelihood)
    topology_list = np.array(topology_list)
    theta_list = np.array(theta_list)
    return loglikelihood, topology_list, theta_list

def truelikelihood(tree_cluster, samples, num_samples, num_clusters, pi):
    r = np.ones((num_samples, num_clusters))
    loglikelihood = [];
    for i, sample in enumerate(samples):
        for j, t in enumerate(tree_cluster):
            visitedNodes = [t.root]
            r[i,j] *= pi[j]
            while len(visitedNodes) != 0:
                presentNode = visitedNodes[0]
                visitedNodes = visitedNodes[1:]
                if len(presentNode.descendants) != 0:
                    visitedNodes = visitedNodes + presentNode.descendants
                if presentNode.ancestor == None:   #root node
                    r[i,j] *= presentNode.cat[sample[int(presentNode.name)]]
                else:
                    r[i,j] *= presentNode.cat[sample[int(presentNode.ancestor.name)]][sample[int(presentNode.name)]]
    eps=eps=sys.float_info.min
    r += eps
    rn = np.sum(r, axis=1).reshape(num_samples,1)
    r /= rn
    return np.sum(np.log(rn))


def likelihood_tree(tree_topology, theta, beta):
    numerator=1
    for i in range(beta.size):
        node=i
        parent_node=tree_topology[node]
        thetaNode=theta[node]
        if(np.isnan(parent_node)):
            numerator=numerator*thetaNode[beta[node]]
        else:
            numerator=numerator*thetaNode[beta[int(parent_node)]][beta[int(node)]]
    return numerator

def main():
    # Code to process command line arguments
    parser = argparse.ArgumentParser(description='EM algorithm for likelihood of a tree GM.')
    parser.add_argument('sample_filename', type=str,
                        help='Specify the name of the sample file (i.e data/example_samples.txt)')
    parser.add_argument('output_filename', type=str,
                        help='Specify the name of the output file (i.e data/example_results.txt)')
    parser.add_argument('num_clusters', type=int, help='Specify the number of clusters (i.e 3)')
    parser.add_argument('--seed_val', type=int, default=42, help='Specify the seed value for reproducibility (i.e 42)')
    parser.add_argument('--real_values_filename', type=str, default="",
                        help='Specify the name of the real values file (i.e data/example_tree_mixture.pkl)')
    # You can add more default parameters if you want.

    print("Hello World!")
    print("This file demonstrates the flow of function templates of question 2.5.")

    print("\n0. Load the parameters from command line.\n")

    args = parser.parse_args()
    print("\tArguments are: ", args)

    print("\n1. Load samples from txt file.\n")

    samples = np.loadtxt(args.sample_filename, delimiter="\t", dtype=np.int32)
    
    sample_filename=args.sample_filename
 
    customData=False
    if sample_filename == "data/q_2_5_tm_20node_20sample_4clusters.pkl_samples.txt":
        
        pi_file='data/q_2_5_tm_20node_20sample_4clusters.pkl_pi.npy'
        tree0='data/q_2_5_tm_20node_20sample_4clusters.pkl_tree_0_topology.npy'
        tree1='data/q_2_5_tm_20node_20sample_4clusters.pkl_tree_1_topology.npy'
        tree2='data/q_2_5_tm_20node_20sample_4clusters.pkl_tree_2_topology.npy'
        tree3='data/q_2_5_tm_20node_20sample_4clusters.pkl_tree_3_topology.npy'
        theta0='data/q_2_5_tm_20node_20sample_4clusters.pkl_tree_0_theta.npy'
        theta1='data/q_2_5_tm_20node_20sample_4clusters.pkl_tree_1_theta.npy'
        theta2='data/q_2_5_tm_20node_20sample_4clusters.pkl_tree_2_theta.npy'
        theta3='data/q_2_5_tm_20node_20sample_4clusters.pkl_tree_3_theta.npy'


    elif sample_filename== "data/q_2_5_tm_10node_50sample_4clusters.pkl_samples.txt":
       
        pi_file='data/q_2_5_tm_10node_50sample_4clusters.pkl_pi.npy'
        tree0='data/q_2_5_tm_10node_50sample_4clusters.pkl_tree_0_topology.npy'
        tree1='data/q_2_5_tm_10node_50sample_4clusters.pkl_tree_1_topology.npy'
        tree2='data/q_2_5_tm_10node_50sample_4clusters.pkl_tree_2_topology.npy'
        tree3='data/q_2_5_tm_10node_50sample_4clusters.pkl_tree_3_topology.npy'
        theta0='data/q_2_5_tm_10node_50sample_4clusters.pkl_tree_0_theta.npy'
        theta1='data/q_2_5_tm_10node_50sample_4clusters.pkl_tree_1_theta.npy'
        theta2='data/q_2_5_tm_10node_50sample_4clusters.pkl_tree_2_theta.npy'
        theta3='data/q_2_5_tm_10node_50sample_4clusters.pkl_tree_3_theta.npy'

    elif sample_filename=="data/q_2_5_tm_10node_20sample_4clusters.pkl_samples.txt":

        pi_file='data/q_2_5_tm_10node_20sample_4clusters.pkl_pi.npy'
        tree0='data/q_2_5_tm_10node_20sample_4clusters.pkl_tree_0_topology.npy'
        tree1='data/q_2_5_tm_10node_20sample_4clusters.pkl_tree_1_topology.npy'
        tree2='data/q_2_5_tm_10node_20sample_4clusters.pkl_tree_2_topology.npy'
        tree3='data/q_2_5_tm_10node_20sample_4clusters.pkl_tree_3_topology.npy'
        theta0='data/q_2_5_tm_10node_20sample_4clusters.pkl_tree_0_theta.npy'
        theta1='data/q_2_5_tm_10node_20sample_4clusters.pkl_tree_1_theta.npy'
        theta2='data/q_2_5_tm_10node_20sample_4clusters.pkl_tree_2_theta.npy'
        theta3='data/q_2_5_tm_10node_20sample_4clusters.pkl_tree_3_theta.npy'
      
    else:
        print('Testing with Custom File. Please provide true mixture for likelihood and RF Comparisiom')
        customData=True

    num_samples, num_nodes = samples.shape
    print("\tnum_samples: ", num_samples, "\tnum_nodes: ", num_nodes)
    print("\tSamples: \n", samples)

    print("\n2. Run EM Algorithm.\n")
    sieving=np.random.randint(20,999, size=10)
    good_seed=sieving[0]
    bestFit=-np.Infinity
    for sieve in sieving:
        loglikelihood, topology_array, theta_array = em_algorithm(sieve, samples, args.num_clusters,10)
        # print(loglikelihood)
        thisFit=loglikelihood[-1]
        if(thisFit>bestFit):
            bestFit=thisFit
            good_seed=sieve
    loglikelihood, topology_array, theta_array = em_algorithm(good_seed, samples, num_clusters=args.num_clusters)

    print("\n3. Save, print and plot the results.\n")
    num_clusters=args.num_clusters
    save_results(loglikelihood, topology_array, theta_array, args.output_filename)

    for i in range(args.num_clusters):
        print("\n\tCluster: ", i)
        print("\tTopology: \t", topology_array[i])
        print("\tTheta: \t", theta_array[i])
    
    plt.figure(figsize=(8, 3))
    plt.subplot(121)
    plt.plot(np.exp(loglikelihood), label='Estimated')
    plt.ylabel("Likelihood of Mixture")
    plt.xlabel("Iterations")
  
    plt.subplot(122)
    plt.title(sample_filename)
    plt.plot(loglikelihood, label='Estimated')
    plt.ylabel("Log-Likelihood of Mixture")
    plt.xlabel("Iterations")
    plt.legend(loc=(1.04, 0))
    plt.show()

    print("\n4. Retrieve real results and compare.\n")
    if args.real_values_filename != "" or True:
        if customData==False:
            print("\tComparing the results with real values...")

            print("\t4.1. Make the Robinson-Foulds distance analysis.\n")
            # TODO: Do RF Comparison      
            t0=Tree()
            t1=Tree()
            t2=Tree()
            t3=Tree()
            true_pi=np.load(pi_file)
            t0.load_tree_from_arrays(tree0,theta0)
            t1.load_tree_from_arrays(tree1,theta1)
            t2.load_tree_from_arrays(tree2,theta2)
            t3.load_tree_from_arrays(tree3,theta3)
            tns = dendropy.TaxonNamespace()
            t0_rf = dendropy.Tree.get(data=t0.get_tree_newick(), schema="newick", taxon_namespace=tns)
            t1_rf = dendropy.Tree.get(data=t1.get_tree_newick(), schema="newick", taxon_namespace=tns)
            t2_rf = dendropy.Tree.get(data=t2.get_tree_newick(), schema="newick", taxon_namespace=tns)
            t3_rf = dendropy.Tree.get(data=t3.get_tree_newick(), schema="newick", taxon_namespace=tns)
            t0_infer=Tree()
            t1_infer=Tree()
            t2_infer=Tree()
            t3_infer=Tree()
            t0_infer.load_tree_from_direct_arrays(topology_array[0],theta_array[0])
            t1_infer.load_tree_from_direct_arrays(topology_array[1],theta_array[1])
            t2_infer.load_tree_from_direct_arrays(topology_array[2],theta_array[2])
            t3_infer.load_tree_from_direct_arrays(topology_array[3],theta_array[3])
            t0_infer_rf = dendropy.Tree.get(data=t0_infer.get_tree_newick(), schema="newick", taxon_namespace=tns)
            t1_infer_rf = dendropy.Tree.get(data=t1_infer.get_tree_newick(), schema="newick", taxon_namespace=tns)
            t2_infer_rf = dendropy.Tree.get(data=t2_infer.get_tree_newick(), schema="newick", taxon_namespace=tns)
            t3_infer_rf = dendropy.Tree.get(data=t3_infer.get_tree_newick(), schema="newick", taxon_namespace=tns)
            print('File:',sample_filename)
            print('------Robinson-Foulds Distance------')
            rfTree0=[RfDist(t0_infer_rf, t0_rf),RfDist(t0_infer_rf, t1_rf),RfDist(t0_infer_rf, t2_rf),RfDist(t0_infer_rf, t3_rf)]
            rfTree1=[RfDist(t1_infer_rf, t0_rf),RfDist(t1_infer_rf, t1_rf),RfDist(t1_infer_rf, t2_rf),RfDist(t1_infer_rf, t3_rf)]
            rfTree2=[RfDist(t2_infer_rf, t0_rf),RfDist(t2_infer_rf, t1_rf),RfDist(t2_infer_rf, t2_rf),RfDist(t2_infer_rf, t3_rf)]
            rfTree3=[RfDist(t3_infer_rf, t0_rf),RfDist(t3_infer_rf, t1_rf),RfDist(t3_infer_rf, t2_rf),RfDist(t3_infer_rf, t3_rf)]
            print('------Real Trees------')
            print(t0.get_tree_newick())
            print(t1.get_tree_newick())
            print(t2.get_tree_newick())
            print(t3.get_tree_newick())
            print('------Inferred Trees------')
            print(t0_infer.get_tree_newick())
            print(t1_infer.get_tree_newick())
            print(t2_infer.get_tree_newick())
            print(t3_infer.get_tree_newick())
            print()
            print('RF Distance of Inferred Tree 0 with each Tree(true):',rfTree0)
            print('RF Distance of Inferred Tree 1 with each Tree(true):',rfTree1)
            print('RF Distance of Inferred Tree 2 with each Tree(true):',rfTree2)
            print('RF Distance of Inferred Tree 3 with each Tree(true):',rfTree3)
            print('------Robinson-Foulds Distance------')
            print("\t4.2. Make the likelihood comparison.\n")

            # TODO: Do Likelihood Comparison

            print('Log Likelihood of real mixture: '+str(truelikelihood([t0, t1, t2, t3], samples, num_samples,num_clusters, true_pi)))
            print('Log Likelihood of inferred mixture: '+str(loglikelihood[-1]))
        else:
            print('Testing with Custom Data')
            tns = dendropy.TaxonNamespace()
            if sample_filename == "newData/Data1/Dataset1.pkl_samples.txt":
                num_clusters = 5
                num_nodes = 30
                tm = TreeMixture(num_clusters, num_nodes)
                seed_val = 12
                tm.simulate_pi(seed_val=seed_val)
                tm.simulate_trees(seed_val)
                seed_val = 12
                num_samples = 20
                tm.sample_mixtures(num_samples, seed_val=seed_val)
                t0_rf = dendropy.Tree.get(data=tm.clusters[0].get_tree_newick(), schema="newick", taxon_namespace=tns)
                t1_rf = dendropy.Tree.get(data=tm.clusters[1].get_tree_newick(), schema="newick", taxon_namespace=tns)
                t2_rf = dendropy.Tree.get(data=tm.clusters[2].get_tree_newick(), schema="newick", taxon_namespace=tns)
                t3_rf = dendropy.Tree.get(data=tm.clusters[3].get_tree_newick(), schema="newick", taxon_namespace=tns)
                t4_rf = dendropy.Tree.get(data=tm.clusters[4].get_tree_newick(), schema="newick", taxon_namespace=tns)
                t0_infer=Tree()
                t1_infer=Tree()
                t2_infer=Tree()
                t3_infer=Tree()
                t4_infer=Tree()
                t0_infer.load_tree_from_direct_arrays(topology_array[0],theta_array[0])
                t1_infer.load_tree_from_direct_arrays(topology_array[1],theta_array[1])
                t2_infer.load_tree_from_direct_arrays(topology_array[2],theta_array[2])
                t3_infer.load_tree_from_direct_arrays(topology_array[3],theta_array[3])
                t4_infer.load_tree_from_direct_arrays(topology_array[4],theta_array[4])
                t0_infer_rf = dendropy.Tree.get(data=t0_infer.get_tree_newick(), schema="newick", taxon_namespace=tns)
                t1_infer_rf = dendropy.Tree.get(data=t1_infer.get_tree_newick(), schema="newick", taxon_namespace=tns)
                t2_infer_rf = dendropy.Tree.get(data=t2_infer.get_tree_newick(), schema="newick", taxon_namespace=tns)
                t3_infer_rf = dendropy.Tree.get(data=t3_infer.get_tree_newick(), schema="newick", taxon_namespace=tns)
                t4_infer_rf = dendropy.Tree.get(data=t4_infer.get_tree_newick(), schema="newick", taxon_namespace=tns)
                print('File:',sample_filename)
                print('------Robinson-Foulds Distance------')
                rfTree0=[RfDist(t0_infer_rf, t0_rf),RfDist(t0_infer_rf, t1_rf),RfDist(t0_infer_rf, t2_rf),RfDist(t0_infer_rf, t3_rf)]
                rfTree1=[RfDist(t1_infer_rf, t0_rf),RfDist(t1_infer_rf, t1_rf),RfDist(t1_infer_rf, t2_rf),RfDist(t1_infer_rf, t3_rf)]
                rfTree2=[RfDist(t2_infer_rf, t0_rf),RfDist(t2_infer_rf, t1_rf),RfDist(t2_infer_rf, t2_rf),RfDist(t2_infer_rf, t3_rf)]
                rfTree3=[RfDist(t3_infer_rf, t0_rf),RfDist(t3_infer_rf, t1_rf),RfDist(t3_infer_rf, t2_rf),RfDist(t3_infer_rf, t3_rf)]
                rfTree4=[RfDist(t4_infer_rf, t0_rf),RfDist(t4_infer_rf, t1_rf),RfDist(t4_infer_rf, t2_rf),RfDist(t4_infer_rf, t4_rf)]
                print('------Real Trees------')
                print(tm.clusters[0].get_tree_newick())
                print(tm.clusters[1].get_tree_newick())
                print(tm.clusters[2].get_tree_newick())
                print(tm.clusters[3].get_tree_newick())
                print(tm.clusters[4].get_tree_newick())
                print('------Inferred Trees------')
                print(t0_infer.get_tree_newick())
                print(t1_infer.get_tree_newick())
                print(t2_infer.get_tree_newick())
                print(t3_infer.get_tree_newick())
                print(t4_infer.get_tree_newick())
                print()
                print('RF Distance of Inferred Tree 0 with each Tree(true):',rfTree0)
                print('RF Distance of Inferred Tree 1 with each Tree(true):',rfTree1)
                print('RF Distance of Inferred Tree 2 with each Tree(true):',rfTree2)
                print('RF Distance of Inferred Tree 3 with each Tree(true):',rfTree3)
                print('RF Distance of Inferred Tree 4 with each Tree(true):',rfTree4)
                print('------Robinson-Foulds Distance------')
                print("\t4.2. Make the likelihood comparison.\n")
                print('Log Likelihood of real mixture: '+str(truelikelihood([tm.clusters[0],tm.clusters[1],tm.clusters[2],tm.clusters[3],tm.clusters[4]], samples, num_samples, num_clusters, tm.pi)))
                print('Log Likelihood of inferred mixture: '+str(loglikelihood[-1]))
            elif sample_filename== "newData/Data2/Dataset2.pkl_samples.txt":
                num_clusters = 3
                num_nodes = 50
                tm = TreeMixture(num_clusters, num_nodes)
                seed_val = 123
                tm.simulate_pi(seed_val=seed_val)
                tm.simulate_trees(seed_val)
                seed_val = 12
                num_samples = 100
                tm.sample_mixtures(num_samples, seed_val=seed_val)
                t0_rf = dendropy.Tree.get(data=tm.clusters[0].get_tree_newick(), schema="newick", taxon_namespace=tns)
                t1_rf = dendropy.Tree.get(data=tm.clusters[1].get_tree_newick(), schema="newick", taxon_namespace=tns)
                t2_rf = dendropy.Tree.get(data=tm.clusters[2].get_tree_newick(), schema="newick", taxon_namespace=tns)
                t0_infer=Tree()
                t1_infer=Tree()
                t2_infer=Tree()
                t0_infer.load_tree_from_direct_arrays(topology_array[0],theta_array[0])
                t1_infer.load_tree_from_direct_arrays(topology_array[1],theta_array[1])
                t2_infer.load_tree_from_direct_arrays(topology_array[2],theta_array[2])
                t0_infer_rf = dendropy.Tree.get(data=t0_infer.get_tree_newick(), schema="newick", taxon_namespace=tns)
                t1_infer_rf = dendropy.Tree.get(data=t1_infer.get_tree_newick(), schema="newick", taxon_namespace=tns)
                t2_infer_rf = dendropy.Tree.get(data=t2_infer.get_tree_newick(), schema="newick", taxon_namespace=tns)
                print('File:',sample_filename)
                print('------Robinson-Foulds Distance------')
                rfTree0=[RfDist(t0_infer_rf, t0_rf),RfDist(t0_infer_rf, t1_rf),RfDist(t0_infer_rf, t2_rf)]
                rfTree1=[RfDist(t1_infer_rf, t0_rf),RfDist(t1_infer_rf, t1_rf),RfDist(t1_infer_rf, t2_rf)]
                rfTree2=[RfDist(t2_infer_rf, t0_rf),RfDist(t2_infer_rf, t1_rf),RfDist(t2_infer_rf, t2_rf)]
                print('------Real Trees------')
                print(tm.clusters[0].get_tree_newick())
                print(tm.clusters[1].get_tree_newick())
                print(tm.clusters[2].get_tree_newick())
                print('------Inferred Trees------')
                print(t0_infer.get_tree_newick())
                print(t1_infer.get_tree_newick())
                print(t2_infer.get_tree_newick())

                print()
                print('RF Distance of Inferred Tree 0 with each Tree(true):',rfTree0)
                print('RF Distance of Inferred Tree 1 with each Tree(true):',rfTree1)
                print('RF Distance of Inferred Tree 2 with each Tree(true):',rfTree2)
                print('------Robinson-Foulds Distance------')
                print("\t4.2. Make the likelihood comparison.\n")
                print('Log Likelihood of real mixture: '+str(truelikelihood([tm.clusters[0],tm.clusters[1],tm.clusters[2]], samples, num_samples, num_clusters, tm.pi)))
                print('Log Likelihood of inferred mixture: '+str(loglikelihood[-1]))


if __name__ == "__main__":
    main()
