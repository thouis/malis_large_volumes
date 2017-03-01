import time
import numpy as np

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
