# Malis for large volumes

## What is this?
This program computes the object pair counts associated with the MALIS method as described in

SC Turaga, KL Briggman, M Helmstaedter, W Denk, HS Seung (2009). *Maximin learning of image segmentation*. _Advances in Neural Information Processing Systems (NIPS) 2009_.

http://papers.nips.cc/paper/3887-maximin-affinity-learning-of-image-segmentation

## Why another implementation?
There is already a fast c++ implementation at github.com/turagalab/malis. So why did we reimplement this?
The existing implementation is not as memory efficient as possible, and it becomes prohibitively memory-demanding
to compute malis for large volumes (hence the name malis_large_volumes here). This implementation computes malis with a
lower memory footprint, but unfortunately at the cost of more computation time.

## Usage
We assume the following layout for your variables:
- labels: (D, W, H)
- affinities: (K, D, W, H) (K is the number of neighbors that a voxel has affinities to)

If you use a normal nearest neighbor connectivity:
```python
import malis_large_volumes
pos_pairs, neg_pairs = malis_large_volumes.get_pairs(labels, affinities)
```
Otherwise you can also specify an arbitrary neighboorhood yourself.

For illustration purposes, let's use the nearest neighbor neighborhood (if you wouldn't specify it, 
this is what malis_large_volumes will do):
```python
import malis_large_volumes

neighborhood = np.array([[-1, 0, 0],
                         [0, -1, 0],
                         [0, 0, -1]], dtype=np.int32)
pos_pairs, neg_pairs = malis_large_volumes.get_pairs(labels, affinities, neighborhood)
```
There are more options available for expert users. For those you will have to check the docstrings at the relevant functions.
