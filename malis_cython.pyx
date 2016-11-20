import numpy as np
cimport numpy as np
from libc.stdint cimport uint32_t
import pdb
cimport cython
from cython.view cimport array as cvarray
from argsort_int32 import qargsort32


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int chase(unsigned int [:] id_table, int idx) nogil:
    """ Terminates once it finds an index into id_table where the corresponding
        element in id_table is an index to itself """
    if id_table[idx] != idx:
        id_table[idx] = chase(id_table, id_table[idx])
    return id_table[idx]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int merge(unsigned int [:] id_table, int idx_from, int idx_to) nogil:
    cdef int old
    if id_table[idx_from] != idx_to:
        old = id_table[idx_from]
        id_table[idx_from] = idx_to
        merge(id_table, old, idx_to)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void my_unravel_index(int k, int [::1] shape, int[4] return_idxes) nogil:
    
    return_idxes[3] = k %  shape[3]
    k -= return_idxes[3]
    return_idxes[2] = k % shape[2]
    k -= return_idxes[2]
    return_idxes[1] = k % shape[1]
    k -= return_idxes[1]
    return_idxes[0] = k % shape[0]


@cython.boundscheck(False)
@cython.wraparound(False)
def build_tree_cython(labels, edge_weights, neighborhood):
    '''find tree of edges linking regions.
        labels = (D, W, H) integer label volume.  0s ignored
        edge_weights = (D, W, H, K) floating point values.
                  Kth entry corresponds to Kth offset in neighborhood.
        neighborhood = (K, 3) offsets from pixel to linked pixel.

        returns: edge tree (D * W * H, 3) (int32)
            array of: (linear edge index, child 1 index in edge tree, child 2 index in edge tree)
            If a child index is -1, it's a pixel, not an edge, and can be
                inferred from the linear edge index.
            Tree is terminated by linear_edge_index == -1

    '''
    cdef int [:, :] neighborhood_view = neighborhood
    cdef int D, W, H
    cdef unsigned int[:, :, :] merged_labels
    cdef unsigned int [:] merged_labels_raveled
    cdef int [:] region_parents
    cdef int [:, :] edge_tree
    cdef float [:] ew_flat
    ew_flat = edge_weights.ravel()

    # get shape information
    D = labels.shape[0]
    W = labels.shape[1]
    H = labels.shape[2]

    # this acts as both the merged label matrix, as well as the pixel-to-pixel
    # linking graph.
    merged_labels = np.arange(labels.size, dtype=np.uint32).reshape((D, W, H))
    merged_labels_raveled = np.asarray(merged_labels).ravel()

    # edge that last merged a region, or -1 if this pixel hasn't been merged
    # into a region, yet.
    region_parents = - np.ones_like(labels, dtype=np.int32).ravel()

    # edge tree
    edge_tree = - np.ones((D * W * H, 3), dtype=np.int32)
    
    # sort array and get corresponding indices
    cdef unsigned int [:] ordered_indices = qargsort32(np.asarray(ew_flat))[::-1]

    cdef int order_index = 0
    cdef int edge_idx
    cdef int d_1, w_1, h_1, k, d_2, w_2, h_2
    cdef int orig_label_1, orig_label_2, region_label_1, region_label_2, new_label
    cdef int [:] offset
    cdef int [::1] ew_shape = np.array(edge_weights.shape).astype(np.int32)
    cdef int return_idxes[4]
    cdef int n_loops = len(ordered_indices)
    cdef int i
    with nogil:
        for i in range(n_loops):
            edge_idx = ordered_indices[i]
            # the size of ordered_indices is k times bigger than the amount of 
            # voxels, but every voxel can be merged by only exactly one edge,
            # so this loop will run exactly n_voxels times.

            # get first voxel connected by the current edge
            my_unravel_index(edge_idx, ew_shape, return_idxes)
            d_1 = return_idxes[0]
            w_1 = return_idxes[1]
            h_1 = return_idxes[2]
            k = return_idxes[3]

            # get idxes of second voxel connected by this edge
            offset = neighborhood_view[k,:]
            d_2 = d_1 + offset[0]
            w_2 = w_1 + offset[1]
            h_2 = h_1 + offset[2]

            # ignore out-of-volume links
            if ((not 0 <= d_2 < D) or
                (not 0 <= w_2 < W) or
                (not 0 <= h_2 < H)):
                continue

            orig_label_1 = merged_labels[d_1, w_1, h_1]
            orig_label_2 = merged_labels[d_2, w_2, h_2]

            region_label_1 = chase(merged_labels_raveled, orig_label_1)
            region_label_2 = chase(merged_labels_raveled, orig_label_2)

            if region_label_1 == region_label_2:
                # already linked in tree, do not create a new edge.
                continue

            edge_tree[order_index, 0] = edge_idx
            edge_tree[order_index, 1] = region_parents[region_label_1]
            edge_tree[order_index, 2] = region_parents[region_label_2]

            # merge regions
            new_label = min(region_label_1, region_label_2)
            merge(merged_labels_raveled, orig_label_1, new_label)
            merge(merged_labels_raveled, orig_label_2, new_label)

            # store parent edge of region by location in tree
            region_parents[new_label] = order_index
            

            order_index += 1
    return np.asarray(edge_tree)


def compute_edge_cost(region_counts_1, region_counts_2, pos_neg_phase):
    cost = 0
    if pos_neg_phase == "pos":
        for key1, val1 in region_counts_1.iteritems():
            for key2, val2 in region_counts_2.iteritems():
                if key1 == key2:
                    cost += val1 * val2
    elif pos_neg_phase == "neg":
        for key1, val1 in region_counts_1.iteritems():
            for key2, val2 in region_counts_2.iteritems():
                if key1 != key2:
                    cost += val1 * val2
    else:
        raise("Specify pos_neg_phase as 'pos' or 'neg'")
    return cost


def compute_cost_recursive(labels, edge_weights, neighborhood, edge_tree, edge_tree_idx, pos_neg_phase, costs):
    linear_edge_index, child_1, child_2 = edge_tree[edge_tree_idx, ...]
    assert costs[edge_tree_idx] == 0.0  # this node shouldn't have been visited before
    assert linear_edge_index != -1  # also marks visited nodes
    if child_1 == -1:
        # first child is a voxel.  Compute its location
        d_1, w_1, h_1, k = np.unravel_index(linear_edge_index, edge_weights.shape)
        region_counts_1 = {labels[d_1, w_1, h_1]: 1}
    else:
        # recurse first child
        region_counts_1 = compute_cost_recursive(labels, edge_weights, neighborhood,
                                                 edge_tree, child_1, pos_neg_phase, costs)

    if child_2 == -1:
        # second child is a voxel.  Compute its location via neighborhood.
        d_1, w_1, h_1, k = np.unravel_index(linear_edge_index, edge_weights.shape)
        offset = neighborhood[k, ...]
        d_2, w_2, h_2 = (o + d for o, d in zip(offset, (d_1, w_1, h_1)))
        region_counts_2 = {labels[d_2, w_2, h_2]: 1}
    else:
        # recurse second child
        region_counts_2 = compute_cost_recursive(labels, edge_weights, neighborhood,
                                                 edge_tree, child_2, pos_neg_phase, costs)

    costs[edge_tree_idx] = compute_edge_cost(region_counts_1, region_counts_2, pos_neg_phase)

    # mark this edge as done so recursion doesn't hit it again
    edge_tree[edge_tree_idx, 0] = -1

    region_counts_1.update(region_counts_2)
    return region_counts_1


def compute_costs(labels, edge_weights, neighborhood, edge_tree, pos_neg_phase):
    costs = np.zeros(edge_tree.shape[0], dtype=np.float32)

    # save these for later.
    linear_edge_indices = edge_tree[:, 0].copy()

    # process tree from root (later in array) to leaves (earlier)
    for idx in range(edge_tree.shape[0] - 1, 0, -1):
        if edge_tree[idx, 0] == -1:
            continue
        assert costs[idx] == 0.0  # this node shouldn't have been visisted before
        compute_cost_recursive(labels, edge_weights, neighborhood,
                               edge_tree, idx, pos_neg_phase, costs)

    costs_array = np.zeros_like(edge_weights)

    # mask to actual edges, put costs in place
    costs = costs[linear_edge_indices > -1]
    linear_edge_indices = linear_edge_indices[linear_edge_indices > -1]
    costs_array.ravel()[linear_edge_indices] = costs

    return costs_array

