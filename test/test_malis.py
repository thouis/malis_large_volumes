import numpy as np
import matplotlib.pyplot as plt
import pdb
import time
import malis_large_volumes
from test_tools import check_tree, malis_turaga
from malis_large_volumes import malis_cython, malis_python




if __name__ == '__main__':
    depth_size_range = [5]
#    max_stack_vec = np.zeros(len(depth_size_range))
    vol_size_vec = np.zeros(len(depth_size_range))
    height_and_width = 1000

    for i, depth_size in enumerate(depth_size_range):
        labels = np.ones((depth_size, height_and_width, height_and_width), dtype=np.uint32)
        labels[2:] = 2
        labels[int(depth_size/2):, int(height_and_width/2):] = 5
        vol_size_vec[i] = labels.size

        # we want the weights to alternate between high an low throughout the volume in order to 
        # create blobs. And we want the labels to (roughly) align with those blobs.
        weights = np.random.normal(size=labels.shape + (3,), loc=.5, scale=.1).astype(dtype=np.float32)
        for j in range(0, np.max(weights.shape), 100):
            try:
                weights[int(j/5)] -= .3 # this is hacky
                labels[:int(j/5)] *= 2
            except:
                pass
            try:
                weights[:, j] -= .3
                labels[:, :j] *= 3
            except:
                pass
            try:
                weights[:, :, j] -= .3
                labels[:, :, :j] *= 4
            except:
                pass
            
        neighborhood = np.array([[-1, 0, 0],
                                 [0, -1, 0],
                                 [0, 0, -1]], dtype=np.int32)

        print("\nImage is of size: " + str(labels.shape))
        print("\ncython:")
        start_time = time.time()
        edge_tree_cython = malis_cython.build_tree(labels, weights, neighborhood)
        end_time = time.time()
        print("Tree computation time: " + str(end_time - start_time))
#        check_tree(edge_tree_cython)
        start_time = time.time()
        pos_pairs, neg_pairs = malis_cython.compute_pairs(labels, weights, neighborhood, edge_tree_cython.copy())
#        max_stack_vec[i] = max_stack
#        print("Max stack: " + str(max_stack))
        end_time = time.time()
        print("Pair computation time: " + str(end_time - start_time))

        ######################################################################
        # Compare with S. Turagas malis implementation
        start_time = time.time()
        pos_pairs_2, neg_pairs_2 = malis_turaga(weights, labels, ignore_background=False)
        end_time = time.time()
        print("Turaga computation time: " + str(end_time - start_time))

#    plt.figure()
#    plt.plot(vol_size_vec, max_stack_vec)
#    plt.xlabel("Number of voxels")
#    plt.ylabel("Maximal tree depth")
#    plt.show()

        # test meta method that combines tree and pair computation
#        print("Testing convenience function pairs()")
#        pos_pairs, neg_pairs = pairs(labels, weights, neighborhood)

#        # regular python
#        print("\nregular python:")
#        start_time = time.time()
#        edge_tree_python = malis_python.build_tree(labels, weights, neighborhood)
#        end_time = time.time()
#        print("Time: " + str(end_time - start_time))
#        check_tree(edge_tree_python)
#
#        print("edge tree indices ")
#        print(edge_tree[:10, 0])
#        print("edge tree cython indices ")
#        print(edge_tree_cython[:10, 0])
#        print("Both indices in edge trees equal: " + str(np.all(edge_tree_cython[:, 0] == edge_tree[:, 0])))
#        print("Both edge trees equal: " + str(np.all(edge_tree_cython == edge_tree)))
#        pdb.set_trace()
#        start_time = time.time()
#        costs = malis_python.compute_costs(labels, weights, neighborhood, edge_tree_python, "neg")
#        end_time = time.time()
#        print("Cost computation time: " + str(end_time - start_time))
#        print("Sum of costs: " + str(np.sum(costs)))
