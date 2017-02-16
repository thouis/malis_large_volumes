import numpy as np
import pdb
import time
#import pyximport
#pyximport.install(setup_args={'include_dirs': [np.get_include()]})
#import malis_cython
#import malis_python
import malis_large_volumes
from malis_large_volumes import malis_cython, malis_python


def check_tree(edge_tree):
    # create alias
    et = edge_tree

    # check that no parent occurs more than once
    unique_idxes, unique_counts = np.unique(et[:, 0], return_counts=True)
    # -1 is allowed to occur more than once, so we set it to 1 here
    unique_counts[unique_idxes==-1] = 1
    assert(np.all(unique_counts == 1))

    # check that no child occurs more than once
    unique_idxes, unique_counts = np.unique(et[:, [1,2]], return_counts=True)
    # -1 is allowed to occur more than once, so we set it to 1 here
    unique_counts[unique_idxes==-1] = 1
    assert np.all(unique_counts == 1), "at least one child occurred more than once"

    # check that no non-parent (value -1) has children that are non-voxels (non -1)
    assert np.all(et[et[:, 0] == -1, 1] == -1), "non-parent has non-voxel children"
    assert np.all(et[et[:, 0] == -1, 2] == -1), "non-parent has non-voxel children"


    # traverse the tree non-recursively to check whether all indices are visited
    # and whether any index is visited twice
    visited_indices = set()
    index = len(et) -1
    while index >= 0:
        # choose new index
        if index in visited_indices or et[index, 0] == -1:
            index -= 1
            continue

        print("Traversing a tree!")
        depth = 1
        max_depth = 1
        neglected_second_children = []
        corresponding_depth = []
        while index != -1:

            # check that we haven't visited this index before
            assert index not in visited_indices, "index " + str(index) + " was visited twice"
            # add this index to visited
            visited_indices.add(index)

            if et[index, 1] != -1:
                # take the first child
                new_index = et[index, 1]
                # check if there was a second non-voxel child, if so add it to our list
                if et[index, 2] != -1:
                    neglected_second_children.append(et[index, 2])
                    corresponding_depth.append(depth)

            elif et[index, 2] != -1:
                # if there was a second child but no first one, take the second but
                # don't add it to the list of neglected children
                new_index = et[index, 2]
            else:
                if len(neglected_second_children) > 0:
                    new_index = neglected_second_children[-1]
                    depth = corresponding_depth[-1]
                    neglected_second_children.pop()
                    corresponding_depth.pop()
                else:
                    new_index = -1

            # update the index
            index = new_index
            depth += 1
            if depth > max_depth:
                max_depth = depth
    print("Finished traversing tree, maximum depth was: " + str(max_depth))

    # check that all elements were visited
    relevant_indices = np.where(et[:, 0] != -1)[0]
    assert set(relevant_indices.tolist()).issubset(visited_indices), "not all indices were visited while traversing the tree"

    print("Finished checking tree, no problems found")


if __name__ == '__main__':
    for depth_size in [2]:
        height_and_width = 128
        labels = np.empty((depth_size, height_and_width, height_and_width), dtype=np.uint32)
        weights = np.random.normal(size=(depth_size, height_and_width, height_and_width, 3)).astype(dtype=np.float32)
        neighborhood = np.array([[1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]], dtype=np.int32)

        print("\nImage is of size: " + str(labels.shape))
        #print(labels.size * labels.itemsize / float(2**30), "Gbytes")


        import pdb; pdb.set_trace()
        # cython
        print("\ncython:")
        start_time = time.time()
        edge_tree_cython = malis_cython.build_tree(labels, weights, neighborhood)
        end_time = time.time()
        print("Tree computation time: " + str(end_time - start_time))
        check_tree(edge_tree_cython)
        start_time = time.time()
        pos_pairs, neg_pairs = malis_python.compute_pairs(labels, weights, neighborhood, edge_tree_cython.copy())
        end_time = time.time()
        print("Pair computation time: " + str(end_time - start_time))

        # test meta method that combines tree and pair computation
        print("Testing convenience function pairs()")
        pos_pairs, neg_pairs = pairs(labels, weights, neighborhood)

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
