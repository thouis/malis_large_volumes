import numpy as np
#from .malis_cython import build_tree
# DEBUG should later be malis_cython
from . import malis_python
from . import malis_cython

def pairs_python(labels, edge_weights, neighborhood=None):
    """
    This function simply combines the build_tree and compute_pairs functions
    """

    if neighborhood is None:
        print("No neighboorhood provided, using 3d NN neighboorhood")
        neighborhood = np.array([[-1, 0, 0],
                                 [0, -1, 0],
                                 [0, 0, -1]], dtype=np.int32)
    edge_tree = malis_python.build_tree(labels, edge_weights, neighborhood)
    return malis_python.compute_pairs(labels, edge_weights, neighborhood, edge_tree)


def pairs_cython(labels, edge_weights, neighborhood=None):
    """
    This function simply combines the build_tree and compute_pairs functions
    """
    labels = labels.astype(np.int32)

    if neighborhood is None:
        print("No neighboorhood provided, using 3d NN neighboorhood")
        neighborhood = np.array([[-1, 0, 0],
                                 [0, -1, 0],
                                 [0, 0, -1]], dtype=np.int32)
    edge_tree = malis_cython.build_tree(labels, edge_weights, neighborhood)
    return malis_cython.compute_pairs(labels, edge_weights, neighborhood, edge_tree)
