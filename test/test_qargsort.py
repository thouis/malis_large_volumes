# noqa: E402
import functools
import random
from time import time
import numpy as np

import pyximport
pyximport.install(setup_args={'include_dirs': [np.get_include()]})
from argsort_int32 import qargsort32  # noqa: E402


def measure(f):
    @functools.wraps(f)
    def wrapper(L):
        start = time()
        try:
            v = f(L)
        finally:
            return v, time() - start
    return wrapper


measure_qargsort = measure(qargsort32)


def run_tests():
    for n in np.r_[np.arange(2, 10), 11, 101, 100, 1000, 1e4, 1e5, 1e6]:
        L = np.random.random_sample(int(n)).astype(np.float32)
        t = float("+inf")
        for _ in range(10):
            random.shuffle(L)
            ordered, tnew = measure_qargsort(L)
            t = min(t, tnew)
            assert np.all(L[ordered] == L[np.argsort(L)])
        report_time(n, t)


def report_time(n, t):
    print("N=%09d took us %.2g\t%s" % (n, t*1000, "ms"))


if __name__=="__main__":
    run_tests()

