import numpy as np
from Tree import Tree
from Tree import Node
def calcProb(node):
    children=np.where(tree_topology==node)
    if(np.size(children)>0):
        left_child=children[0][0]
        right_child=children[0][1]
        calcProb_left=calcProb(left_child)
        calcProb_right=calcProb(right_child)
        left_likelihood=np.zeros(cat)
        for i in range(cat):
            left_likelihood[i]=theta[left_child][i].dot(calcProb_left.T)
        right_likelihood=np.zeros(cat)
        for i in range(cat):
            right_likelihood[i]=theta[right_child][i].dot(calcProb_right.T)
        return left_likelihood*right_likelihood
    else:
        likelihood=np.zeros(cat)
        likelihood[int(beta[node])]=1 
        return likelihood
def calculate_likelihood(tree_topology, theta, beta):
    cat=5

    likelihood=calcProb(0).dot(np.asarray(theta[0]).T)
    return likelihood
    
def main():
    for filename in ["data/q2_3_small_tree.pkl","data/q2_3_medium_tree.pkl","data/q2_3_large_tree.pkl"]:
        t = Tree()
        t.load_tree(filename)
        # t.print()
        print("\n2. Calculate likelihood of each FILTERED sample\n")
        print("\tFilename :",filename)
        print()
        for sample_idx in range(t.num_samples):
            beta = t.filtered_samples[sample_idx]
            # print(beta)
            # print("\n\tSample: ", sample_idx, "\tBeta: ", beta)
            sample_likelihood = calculate_likelihood(t.get_topology_array(), t.get_theta_array(), beta)
            print("\tLikelihood for Sample ",sample_idx," is:", sample_likelihood)


if __name__ == "__main__":
    main()
