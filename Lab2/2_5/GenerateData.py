from Tree import TreeMixture
# num_clusters = 5
# num_nodes = 30
# tm = TreeMixture(num_clusters, num_nodes)
# tm.print()

# seed_val = 12
# tm.simulate_pi(seed_val=seed_val)
# tm.simulate_trees(seed_val)
# tm.print()

# print("\n8. Simulate samples for tree mixture and print it:\n")
# seed_val = 12
# num_samples = 20
# tm.sample_mixtures(num_samples, seed_val=seed_val)
# tm.samples
# tm.print()

# print("\n9. Save the tree mixture:\n")
# filename = "newData/Data1/Dataset1.pkl"
# tm.save_mixture(filename, save_arrays=True)

num_clusters = 6
num_nodes = 10
tm = TreeMixture(num_clusters, num_nodes)
tm.print()

seed_val = 123
tm.simulate_pi(seed_val=seed_val)
tm.simulate_trees(seed_val)
tm.print()

print("\n8. Simulate samples for tree mixture and print it:\n")
seed_val = 12
num_samples = 25
tm.sample_mixtures(num_samples, seed_val=seed_val)
tm.samples
tm.print()

print("\n9. Save the tree mixture:\n")
filename = "newData/Data2/Dataset2.pkl"
tm.save_mixture(filename, save_arrays=True)