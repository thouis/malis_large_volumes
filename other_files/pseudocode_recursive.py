def recursive_function(current_edge, counts_array):
    index, child_1, child_2 = current_edge

    if child_1.stopping_condition:
        # assign value for child 1
        child_1_value = "some_value"
    else:
        # recurse first child
        recursive_function(child_1, counts_array)

    if child_2.stopping_condition:
        # assign value for child 2
        child_2_value = "some_value"
    else:
        # recurse first child
        recursive_function(child_2, counts_array)

    # compute the return value for this edge
    edge_value = "result"  # contains counts from child_1 and child_2

    # compute total counts and save in counts_array
    counts_array[index] = "total_counts"  # function of child_1_value, child_2_value

    return edge_value
