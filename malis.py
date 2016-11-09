import numpy as np


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
        build_tree(labels, weights, neighborhood)
