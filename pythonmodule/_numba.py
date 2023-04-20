try:
    from ._dbscan import _mclib_DBSCAN
except:
    _mclib_DBSCAN = None

import numpy as np

if _mclib_DBSCAN is None:
    from ._dbscan import _mclib_DBSCAN_ptr

    # https://github.com/numba/numba/issues/7818
    import ctypes
    from ctypes import pythonapi
    #https://docs.python.org/3/c-api/capsule.html

    def capsule_name(capsule):
        pythonapi.PyCapsule_GetName.restype = ctypes.c_char_p
        pythonapi.PyCapsule_GetName.argtypes = [ctypes.py_object]
        return pythonapi.PyCapsule_GetName(capsule)

    def get_f2py_function_address(capsule):
        name = capsule_name(capsule)
        pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
        pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
        return pythonapi.PyCapsule_GetPointer(capsule, name)

    import numpy.ctypeslib
    del numpy

    _mclib_DBSCAN_argtypes = [ctypes.c_int,
                            ctypes.c_int,
                            # np.ctypeslib.ndpointer(dtype=np.float64,
                            #                         ndim=2,
                            #                         flags=('C_CONTIGUOUS', 'ALIGNED')),
                            ctypes.c_void_p,
                            ctypes.c_double,
                            ctypes.c_int,
                            # np.ctypeslib.ndpointer(dtype=np.bool_,
                            #                         ndim=1,
                            #                         flags=('C_CONTIGUOUS', 'ALIGNED', 'WRITEABLE')),
                            ctypes.c_void_p,
                            # np.ctypeslib.ndpointer(dtype=np.intc,
                            #                         ndim=1,
                            #                         flags=('C_CONTIGUOUS', 'ALIGNED', 'WRITEABLE')),
                            ctypes.c_void_p,]

    _mclib_DBSCAN_addr = get_f2py_function_address(_mclib_DBSCAN_ptr)
    _mclib_DBSCAN = ctypes.CFUNCTYPE(ctypes.c_int, *_mclib_DBSCAN_argtypes)(_mclib_DBSCAN_addr)

from numba.core.errors import TypingError
from numba import types
import numba.extending

from ._dbscan import DBSCAN as DBSCAN_py

@numba.extending.overload(DBSCAN_py)
def DBSCAN_overload(X: np.ndarray, eps: float=0.5, min_samples: int=5):
    if not isinstance(X, types.Array) or X.ndim != 2:
        raise TypingError("'X' must be a 2D array")
    elif not isinstance(eps, (types.Integer, types.Float, types.NoneType, int, float)):
        raise TypingError("'eps' must be a positive real")
    elif not isinstance(min_samples, (types.Integer, types.NoneType, int)):
        raise TypingError("'min_samples' must be a positive integer")

    def DBSCAN(X: np.ndarray, eps: float=0.5, min_samples: int=5):
        n, dim = X.shape

        core_samples = np.empty(n, dtype=np.bool_)
        labels = np.empty(n, dtype=np.intc)

        err = _mclib_DBSCAN(
            dim,
            n,
            X.ctypes,
            eps,
            min_samples,
            core_samples.ctypes,
            labels.ctypes
        )

        if err != 0:
            # TODO: there is very clearly a bug in Numba
            raise ValueError(f'DBSCAN: dimensions are not supported') # dim

        return labels, core_samples

    return DBSCAN

# import dbscan, numba
# import numpy as np
# @numba.njit
# def f(a):
#     return dbscan.DBSCAN(a)

# f(np.zeros((5,3)))
