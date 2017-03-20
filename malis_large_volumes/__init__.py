import numpy as np
from . import malis_python
from . import malis_cython

def get_pairs(labels, edge_weights, neighborhood=None,
              keep_objs_per_edge=20):
    """
    This function simply combines the build_tree and compute_pairs functions
    """
    labels = labels.astype(np.uint32)
    if neighborhood is None:
        neighborhood = np.array([[-1, 0, 0],
                                 [0, -1, 0],
                                 [0, 0, -1]], dtype=np.int32)
    edge_tree = malis_cython.build_tree(labels, edge_weights, neighborhood)
    return malis_cython.compute_pairs_with_tree(labels, edge_weights, neighborhood, edge_tree,
                                                keep_objs_per_edge=keep_objs_per_edge)

def get_pairs_python(labels, edge_weights, neighborhood=None,
                     keep_objs_per_edge=20):
    """
    This function simply combines the build_tree and compute_pairs functions
    """
    labels = labels.astype(np.uint32)
    if neighborhood is None:
        neighborhood = np.array([[-1, 0, 0],
                                 [0, -1, 0],
                                 [0, 0, -1]], dtype=np.int32)
    edge_tree = malis_python.build_tree(labels, edge_weights, neighborhood)
    return malis_python.compute_pairs_with_tree(labels, edge_weights, neighborhood, edge_tree,
                                                keep_objs_per_edge=keep_objs_per_edge)


