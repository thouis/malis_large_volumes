from .malis_cython import build_tree
# DEBUG should later be malis_cython
from .malis_python import compute_pairs

def pairs(labels, edge_weights, neighborhood=None):
    """
    This function simply combines the build_tree and compute_pairs functions
    """

    if neighborhood is None:
        print("No neighboorhood provided, using 3d NN neighboorhood")
        neighborhood = np.array([[1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]], dtype=np.int32)
    edge_tree = build_tree(labels, edge_weights, neighborhood)
    return compute_pairs(labels, edge_weights, neighborhood, edge_tree)

