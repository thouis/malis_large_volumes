#cython: language_level=3, boundscheck=False, wraparound=False

import numpy as np
cimport numpy as np
from libc.stdint cimport uint32_t

# based on
# https://gist.github.com/zed/1257360/ac36097250845568205481b2c707b74785f3f7c1

cdef:
   int CUTOFF = 17  # when to switch to insertion sort

cdef inline void swap(uint32_t* a, int i, int j) nogil:
    a[i], a[j] = a[j], a[i]

cdef int argpartition(uint32_t *offsets,
                      float *values,
                      int start, int end) nogil:
    with gil:
        assert end > start, "end not > start"
    cdef int i = start, j = end - 1
    cdef float pivot = values[offsets[j]]

    while True:
        # invariant: all(x < pivot for x in values[offsets[start:i]])
        # invariant: all(x >= pivot for x in values[offsets[j:end]])

        while values[offsets[i]] < pivot:
            i += 1
        while i < j and pivot <= values[offsets[j]]:
            j -= 1
        if i >= j:
            break
        with gil:
            assert values[offsets[j]] < pivot <= values[offsets[i]], "problem1"
        swap(offsets, i, j)
        with gil:
            assert values[offsets[i]] < pivot <= values[offsets[j]], "problem2"
    with gil:
        assert i >= j, "i < j"
        assert i < end, "i >= end"
    swap(offsets, i, end - 1)
    with gil:
        assert values[offsets[i]] == pivot, "pivot not at pivot"
    # at end of loop: all(x < pivot for x in a[start:i])
    # at end of loop: all(x >= pivot for x in a[i:end])
    return i

cdef void insertion_argsort(uint32_t *offsets,
                            float *values,
                            int start, int end) nogil:
    cdef int i, j
    cdef int idxv
    cdef float v
    for i in range(start + 1, end):
        # invariant: values[offsets[start:i]] is sorted

        # v is index of value to be inserted
        idxv = offsets[i]
        v = values[idxv]

        j = i - 1
        while j >= start:
            # if current j indexes a value lower than v, we will insert idxv
            # just past j...
            if values[offsets[j]] <= v: break
            # otherwise, shift offsets[j] one to the right
            offsets[j + 1] = offsets[j]
            j -= 1
        offsets[j + 1] = idxv  # ... insert idxv here

cdef void qargsort(uint32_t[:]  offsets,
                   float[:] values,
                   int start, int end) nogil:
    if end - start < CUTOFF:
        insertion_argsort(&offsets[0], &values[0], start, end)
        return

    cdef int boundary = argpartition(&offsets[0], &values[0], start, end)
    qargsort(offsets, values, start, boundary)
    qargsort(offsets, values, boundary + 1, end)

def qargsort32(arr):
    assert arr.size < 2**32
    offsets = np.arange(arr.size, dtype=np.uint32)
    qargsort(offsets, arr, 0, arr.size)
    return offsets
