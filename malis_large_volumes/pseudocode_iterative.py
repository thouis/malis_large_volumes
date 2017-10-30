def recursive_function(root_edge, counts_array):
    index, child_1, child_2 = root_edge

    stackentry_template = {
            "edge_tree_idx": 0,
            "child_1_status": 0,
            "child_2_status": 0,
            "region_counts_1": {}
    }


    # add first stackentry to stack
    stack = []
    stackentry = stackentry_template.copy()
    stackentry["edge_tree_idx"] = edge_tree_idx
    stack.append(stackentry)

    while len(stack) > 0:

        # get latest stackentry
        stackentry = stack[-1]
        index, child_1, child_2 = stackentry

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

        ########################################################################
        # Handle child-values (region_counts_1 and _2) and "return"

        # (NOT included in pseudocode)

