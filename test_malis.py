import numpy as np
import pdb
import time
import pyximport
from malis import build_tree
pyximport.install(setup_args={'include_dirs': [np.get_include()]})
from malis_cython import build_tree_cython

if __name__ == '__main__':
    for sz in range(5, 11):
        labels = np.empty((2 ** sz, 2 ** sz, 2 ** sz), dtype=np.uint32)
        weights = np.random.normal(size=(2 ** sz, 2 ** sz, 2 ** sz, 3)).astype(dtype=np.float32)
        neighborhood = np.zeros((3, 3), dtype=np.int32)
        neighborhood[0, ...] = [1, 0, 0]
        neighborhood[1, ...] = [0, 1, 0]
        neighborhood[2, ...] = [0, 0, 1]

        print("\nImage is of size: " + str(labels.shape))
        #print(labels.size * labels.itemsize / float(2**30), "Gbytes")

        # cython 
        start_time = time.time()
        edge_tree_cython = build_tree_cython(labels, weights, neighborhood)
        end_time = time.time()
        print("\ncython:")
        print("Time: " + str(end_time - start_time))
        print(edge_tree_cython[1])

        # regular python
        start_time = time.time()
        edge_tree = build_tree(labels, weights, neighborhood)
        end_time = time.time()
        print("\nregular python:")
        print("Time: " + str(end_time - start_time))
        print(edge_tree[1])

#        costs = compute_costs(labels, weights, neighborhood, edge_tree, "neg")
#        print("Sum of costs: " + str(np.sum(costs)))
