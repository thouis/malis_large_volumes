import numpy as np
import pdb


def chase(id_table, idx):
    if id_table[idx] != idx:
        id_table[idx] = chase(id_table, id_table[idx])
    return id_table[idx]


def merge(id_table, idx_from, idx_to):
    if id_table[idx_from] != idx_to:
        old = id_table[idx_from]
        id_table[idx_from] = idx_to
        merge(id_table, old, idx_to)


def build_tree(labels, edge_weights, neighborhood):
    '''find tree of edges linking regions.
        labels = (D, W, H) integer label volume.  0s ignored
        edge_weights = (D, W, H, K) floating point values.
                  Kth entry corresponds to Kth offset in neighborhood.
        neighborhood = (K, 3) offsets from pixel to linked pixel.

        returns: edge tree (D * W * H, 3) (int32)
            array of: (linear edge index, child 1 index in edge tree, child 2 index in edge tree)
            If a child index is -1, it's a pixel, not an edge, and can be inferred from the linear edge index.
            Tree is terminated by linear_edge_index == -1
    '''

    D, W, H = labels.shape
    ew_flat = edge_weights.ravel()

    # this acts as both the merged label matrix, as well as the pixel-to-pixel
    # linking graph.
    merged_labels = np.arange(labels.size, dtype=np.uint32).reshape(labels.shape)

    # edge that last merged a region, or -1 if this pixel hasn't been merged
    # into a region, yet.
    region_parents = - np.ones_like(labels, dtype=np.int32).ravel()

    # edge tree
    edge_tree = - np.ones((D * W * H, 3), dtype=np.int32)

    ordered_indices = ew_flat.argsort()[::-1]
    order_index = 0

    for edge_idx in ordered_indices:
        d_1, w_1, h_1, k = np.unravel_index(edge_idx, edge_weights.shape)
        offset = neighborhood[k, ...]
        d_2, w_2, h_2 = (o + d for o, d in zip(offset, (d_1, w_1, h_1)))

        # ignore out-of-volume links
        if ((not 0 <= d_2 < D) or
            (not 0 <= w_2 < W) or
            (not 0 <= h_2 < H)):
            continue

        orig_label_1 = merged_labels[d_1, w_1, h_1]
        orig_label_2 = merged_labels[d_2, w_2, h_2]

        region_label_1 = chase(merged_labels.ravel(), orig_label_1)
        region_label_2 = chase(merged_labels.ravel(), orig_label_2)

        if region_label_1 == region_label_2:
            # already linked in tree, do not create a new edge.
            continue

        edge_tree[order_index, 0] = edge_idx
        edge_tree[order_index, 1] = region_parents[region_label_1]
        edge_tree[order_index, 2] = region_parents[region_label_2]

        # merge regions
        new_label = min(region_label_1, region_label_2)
        merge(merged_labels.ravel(), orig_label_1, new_label)
        merge(merged_labels.ravel(), orig_label_2, new_label)

        # store parent edge of region by location in tree
        region_parents[new_label] = order_index
        order_index += 1
    return edge_tree


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
        # recurse first child
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

if __name__ == '__main__':
    for sz in range(4, 5):
        print(2 ** sz)
        labels = np.empty((2 ** sz, 2 ** sz, 2 ** sz), dtype=np.uint32)
        weights = np.empty((2 ** sz, 2 ** sz, 2 ** sz, 3), dtype=np.float32)
        neighborhood = np.zeros((3, 3))
        neighborhood[0, ...] = [1, 0, 0]
        neighborhood[1, ...] = [0, 1, 0]
        neighborhood[2, ...] = [0, 0, 1]
        print(labels.size * labels.itemsize / float(2**30), "Gbytes")
        edge_tree = build_tree(labels, weights, neighborhood)
        costs = compute_costs(labels, weights, neighborhood, edge_tree, "neg")
