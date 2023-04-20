import ctypes
import numpy.ctypeslib
del numpy
import numpy as np
import os

__all__ = ('DBSCAN',)

try:
    _mclib = np.ctypeslib.load_library('libdbscan', os.path.dirname(__file__))
except:
    _mclib = np.ctypeslib.load_library('dbscan', os.path.dirname(__file__))

_mclib_DBSCAN_argtypes = [ctypes.c_int,
                          ctypes.c_int,
                          np.ctypeslib.ndpointer(dtype=np.float64,
                                                 ndim=2,
                                                 flags=('C_CONTIGUOUS', 'ALIGNED')),
                          ctypes.c_double,
                          ctypes.c_int,
                          np.ctypeslib.ndpointer(dtype=np.bool_,
                                                 ndim=1,
                                                 flags=('C_CONTIGUOUS', 'ALIGNED', 'WRITEABLE')),
                          np.ctypeslib.ndpointer(dtype=np.intc,
                                                 ndim=1,
                                                 flags=('C_CONTIGUOUS', 'ALIGNED', 'WRITEABLE')),]

_mclib_DBSCAN = _mclib.DBSCAN
_mclib_DBSCAN.argtypes = _mclib_DBSCAN_argtypes

def DBSCAN(X, eps: float=0.5, min_samples: int=5):
    if len(X.shape) != 2:
        raise ValueError('DBSCAN: X must be a 2D array')

    n, dim = X.shape

    core_samples = np.empty(n, dtype=np.bool_)
    labels = np.empty(n, dtype=np.intc)

    err = _mclib_DBSCAN(
        dim,
        n,
        X,
        eps,
        min_samples,
        core_samples,
        labels
    )

    if err != 0:
        raise ValueError(f'DBSCAN: {dim} dimensions are not supported')

    return labels, core_samples
