import numpy as np
import pyximport


def chase(id_table, idx):
    if id_table[idx] != idx:
        id_table[idx] = chase(id_table, id_table[idx])
    return id_table[idx]


def build_tree(labels, edge_weights, neighborhood,
               stochastic_malis_param=None):
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

    D, W, H = labels.shape
    ew_flat = edge_weights.ravel()

    # this acts as both the merged label matrix, as well as the pixel-to-pixel
    # linking graph.
    merged_labels = np.arange(labels.size, dtype=np.uint32).reshape(labels.shape)

    # edge that last merged a region, or -1 if this pixel hasn't been merged
    # into a region, yet.
    region_parents = - np.ones_like(labels, dtype=np.int32).ravel()

    # edge tree
    edge_tree = - np.ones((labels.size, 3), dtype=np.int32)

    ordered_indices = ew_flat.argsort()[::-1].astype(np.uint32)
    order_index = 0

    for edge_idx in ordered_indices:
        # the size of ordered_indices is k times bigger than the amount of
        # voxels, but every voxel can be merged by only exactly one edge,
        # so this loop will run exactly n_voxels times.

        # get the positions of the two voxels linked by current edge
        k, d_1, w_1, h_1 = np.unravel_index(edge_idx, edge_weights.shape)
        offset = neighborhood[k, ...]
        d_2, w_2, h_2 = (int(o + d) for o, d in zip(offset, (d_1, w_1, h_1)))

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

        # choose one of the two labels to be the new label
        new_label = min(region_label_1, region_label_2)

        # store parent edge of region by location in tree
        region_parents[new_label] = order_index

        merged_labels[d_1, w_1, h_1] = new_label
        merged_labels[d_2, w_2, h_2] = new_label

        order_index += 1
    return edge_tree


def compute_pairs_iterative(labels, edge_weights, neighborhood, edge_tree, edge_tree_idx, pos_pairs, neg_pairs):

    stack = []
    stackentry_template = {
            "edge_tree_idx": 0,
            "child_1_status": 0,
            "child_2_status": 0,
            "region_counts_1": {}
    }

    # add first stackentry to stack
    stackentry = stackentry_template.copy()
    stackentry["edge_tree_idx"] = edge_tree_idx
    stack.append(stackentry)

    while len(stack) > 0:
        stackentry = stack[-1]
        linear_edge_index, child_1, child_2 = edge_tree[stackentry["edge_tree_idx"], ...]
        k, d_1, w_1, h_1 = np.unravel_index(linear_edge_index, edge_weights.shape)

        ########################################################################
        # Child 1
        if stackentry["child_1_status"] == 0:
            if child_1 == -1:
                stackentry["region_counts_1"] = {labels[d_1, w_1, h_1]: 1}
                stackentry["child_1_status"] = 2
            else:
                # recurse first child
                # add to stack
                next_stackentry = stackentry_template.copy()
                next_stackentry["edge_tree_idx"] = child_1
                stack.append(next_stackentry)
                stackentry["child_1_status"] = 1
                continue
        elif stackentry["child_1_status"] == 1:
            stackentry["region_counts_1"] = return_dict
            stackentry["child_1_status"] = 2

        ########################################################################
        # Child 2
        if stackentry["child_2_status"] == 0:

            if child_2 == -1:
                offset = neighborhood[k, ...]
                d_2, w_2, h_2 = (o + d for o, d in zip(offset, (d_1, w_1, h_1)))
                region_counts_2 = {labels[d_2, w_2, h_2]: 1}
            else:
                # recurse first child
                # add to stack
                next_stackentry = stackentry_template.copy()
                next_stackentry["edge_tree_idx"] = child_2
                stack.append(next_stackentry)
                stackentry["child_2_status"] = 1
                continue
        elif stackentry["child_2_status"] == 1:
            region_counts_2 = return_dict

        # syntactic sugar for below
        region_counts_1 = stackentry["region_counts_1"]

        # mark this edge as done so recursion doesn't hit it again
        edge_tree[stackentry["edge_tree_idx"], 0] = -1

        return_dict = {}
        for key1, counts1 in region_counts_1.items():
            for key2, counts2 in region_counts_2.items():
                if key1 == key2 \
                   and not key1 == 0 \
                   and not key2 == 0:
                    pos_pairs[k, d_1, w_1, h_1] += counts1 * counts2
                else:
                    neg_pairs[k, d_1, w_1, h_1] += counts1 * counts2

        for key1, counts1 in region_counts_1.items():
            return_dict[key1] = counts1

        for key2, counts2 in region_counts_2.items():
            if key2 in return_dict.keys():
                return_dict[key2] += counts2
            else:
                return_dict[key2] = counts2
        stack.pop()


def compute_pairs_with_tree(labels, edge_weights, neighborhood, edge_tree,
                            count_method=None, keep_objs_per_edge=None):
    pos_pairs = np.zeros((neighborhood.shape[0],) + labels.shape, dtype=np.uint32)
    neg_pairs = np.zeros((neighborhood.shape[0],) + labels.shape, dtype=np.uint32)

    # process tree from root (later in array) to leaves (earlier)
    for idx in range(edge_tree.shape[0] - 1, 0, -1):
        if edge_tree[idx, 0] == -1:
            continue
        compute_pairs_iterative(labels, edge_weights, neighborhood,
                                edge_tree, idx, pos_pairs, neg_pairs)

    return pos_pairs, neg_pairs
