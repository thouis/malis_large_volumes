import numpy as np
import pdb
import time
import pyximport
import malis_python
pyximport.install(setup_args={'include_dirs': [np.get_include()]})
import malis_cython

if __name__ == '__main__':
    for sz in range(5, 6):
        labels = np.empty((2 ** sz, 2 ** sz, 2 ** sz), dtype=np.uint32)
        weights = np.random.normal(size=(2 ** sz, 2 ** sz, 2 ** sz, 3)).astype(dtype=np.float32)
        neighborhood = np.zeros((3, 3), dtype=np.int32)
        neighborhood[0, ...] = [1, 0, 0]
        neighborhood[1, ...] = [0, 1, 0]
        neighborhood[2, ...] = [0, 0, 1]

        print("\nImage is of size: " + str(labels.shape))
        #print(labels.size * labels.itemsize / float(2**30), "Gbytes")

        # cython 
        print("\ncython:")
        start_time = time.time()
        edge_tree_cython = malis_cython.build_tree(labels, weights, neighborhood)
        end_time = time.time()
        print("Tree computation time: " + str(end_time - start_time))
#        start_time = time.time()
#        costs = malis_python.compute_costs(labels, weights, neighborhood, edge_tree_cython, "neg")
#        end_time = time.time()
#        print("Cost computation time: " + str(end_time - start_time))
#        print("Sum of costs: " + str(np.sum(costs)))

        # regular python
        print("\nregular python:")
        start_time = time.time()
        edge_tree = malis_python.build_tree(labels, weights, neighborhood)
        end_time = time.time()
        print("Time: " + str(end_time - start_time))

        print("edge tree indices ")
        print(edge_tree[:10, 0])
        print("edge tree cython indices ")
        print(edge_tree_cython[:10, 0])
        print("Both indices in edge trees equal: " + str(np.all(edge_tree_cython[:, 0] == edge_tree[:, 0])))
        print("Both edge trees equal: " + str(np.all(edge_tree_cython == edge_tree)))
#        pdb.set_trace()
#        start_time = time.time()
#        costs = malis_python.compute_costs(labels, weights, neighborhood, edge_tree, "neg")
#        end_time = time.time()
#        print("Cost computation time: " + str(end_time - start_time))
#        print("Sum of costs: " + str(np.sum(costs)))

