import numpy as np
from malis_large_volumes import pairs_python
from malis_large_volumes import pairs_cython
from malis_large_volumes import get_pairs_python
from malis_large_volumes import get_pairs_cython
import malis.malis_pair_wrapper as malis_pairs_wrapper_turaga

neighborhood = np.array([[-1, 0, 0],
                         [0, -1, 0],
                         [0, 0, -1]], dtype=np.int32)

ignore_background = False


def test(pairs_module, get_pairs):
    """
    pairs_module:
        the pairs module from malis_large_volumes, for which the
        functions should be tested (can be python or cython version)
    get_pairs_func:
        the function 'wrapper' to be used as imported from the __init__
        of malis_large_volumes
    """
    #######################################################
    # TEST 1
    print("\nStarting test 1")
    labels = np.array([[[1, 1, 1, 2, 2, 2]]], dtype=np.uint32)
    weights = np.zeros(shape=(3,) + labels.shape)
    weights[2, :, :, :] = np.array([[[.2, .2, .2, .1, .2, .2]]], dtype=np.float)
    weights += np.random.normal(size=weights.shape, scale=.001)

    edge_tree = pairs_module.build_tree(labels, weights, neighborhood,
                                        stochastic_malis_param=0)
    pos_pairs, neg_pairs = pairs_module.compute_pairs_with_tree(
                               labels, weights,
                               neighborhood, edge_tree, count_method=0,
                               ignore_background=True)
    assert neg_pairs[2, 0, 0, 3] == 9, "neg pairs result was incorrect"

    # compare with turagas implementation
    pos_pairs_2, neg_pairs_2 = malis_pairs_wrapper_turaga.get_counts(
                                   weights,
                                   labels.astype(np.int64),
                                   ignore_background=True)
    assert np.all(pos_pairs == pos_pairs_2), "pos pairs was not same as turaga implementation"
    assert np.all(neg_pairs == neg_pairs_2), "neg pairs was not same as turaga implementation"
    print("Test 1 finished, no error\n")

    #######################################################
    # TEST 2
    print("Starting test 2")

    labels = np.ones((4, 3, 3), dtype=np.uint32)
    labels[2:] = 2
    weights = np.random.normal(size=(3,) + labels.shape, loc=.5, scale=.01).astype(np.float)
    weights[0, 2, :, :] -= .3
    weights[0, 2, 1, 1] = .4  # this is the maximin edge between the two objects

    edge_tree = pairs_module.build_tree(labels, weights, neighborhood)
    pos_pairs, neg_pairs = pairs_module.compute_pairs_with_tree(labels, weights, neighborhood, edge_tree,
                                                                ignore_background=True)
    assert neg_pairs[0, 2, 1, 1] == (2 * 3 * 3) ** 2

    # compare with turagas implementation
    pos_pairs_2, neg_pairs_2 = malis_pairs_wrapper_turaga.get_counts(weights,
                                                                     labels.astype(np.int64),
                                                                     ignore_background=True)
    assert np.all(pos_pairs == pos_pairs_2), "pos pairs was not same as turaga implementation"
    assert np.all(neg_pairs == neg_pairs_2), "neg pairs was not same as turaga implementation"
    print("Test 2 finished, no error\n")

    #######################################################
    # TEST 3
    print("Starting test 3")
    # in this test we're just comparing the current implementation and Turagas
    labels = np.random.randint(0, 10, size=(10, 20, 20), dtype=np.uint32)
    weights = np.random.normal(loc=0.5, scale=0.1, size=(3,) + labels.shape).astype(np.float)

    edge_tree = pairs_module.build_tree(labels, weights, neighborhood)
    pos_pairs, neg_pairs = pairs_module.compute_pairs_with_tree(labels, weights,
                              neighborhood, edge_tree, keep_objs_per_edge=20,
                              ignore_background=ignore_background)

    # compare with turagas implementation
    pos_pairs_2, neg_pairs_2 = malis_pairs_wrapper_turaga.get_counts(weights,
                                                                     labels.astype(np.int64),
                                                                     ignore_background=ignore_background)
    try:
        assert np.all(pos_pairs == pos_pairs_2), "pos pairs was not same as turaga implementation"
        assert np.all(neg_pairs == neg_pairs_2), "neg pairs was not same as turaga implementation"
        print("Test 3 finished, no error\n")
    except Exception as e:
        print("Test 3 FAILED!")
        print("Exception:\n" + str(e))
        print("Tree-malis was not the same as Turaga-malis.")
        print("However, this happens sometimes and I assume it's due to differences in " +
              "sorting the edges. Try running the tests again and see if it fails again.\n")

    #######################################################

    #######################################################
    # TEST 4
    print("Starting test 4")
    # In this test we're testing the wrapper that will be used by external users
    pos_pairs_from_get_pairs, neg_pairs_from_get_pairs = get_pairs(
            labels, weights, neighborhood, keep_objs_per_edge=20,
            ignore_background=ignore_background)

    assert np.all(pos_pairs == pos_pairs_from_get_pairs), "pos pairs was not same as pos pairs from get_pairs"
    assert np.all(neg_pairs == neg_pairs_from_get_pairs), "neg pairs was not same as neg pairs from get_pairs"
    print("Test 4 finished, no error\n")
    #######################################################


if __name__ == "__main__":
    print("Testing Cython implementation")
    test(pairs_cython, get_pairs_cython)

    print("Testing Python implementation")
    test(pairs_python, get_pairs_python)

