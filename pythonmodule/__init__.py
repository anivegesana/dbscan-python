from ._dbscan import *
DBSCAN.__module__ = 'dbscan'

# Load version from _version.py if available
__all__ = ('DBSCAN_MIN_DIMS', 'DBSCAN_MAX_DIMS')
try:
    from ._version import version as __version__
    __all__ += ('__version__',)
except:
    pass

import sys

if sys.hexversion >= 0x030700f0:
    def __getattr__(name):
        if name == 'sklDBSCAN':
            import warnings
            warnings.warn('sklDBSCAN moved to dbscan.sklearn to make importing faster.', DeprecationWarning)
            from .sklearn import sklDBSCAN
            globals()['sklDBSCAN'] = sklDBSCAN
            return sklDBSCAN
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
else:
    try:
        from .sklearn import sklDBSCAN
    except:
        # scikit-learn might have not been installed correctly
        pass

del sys

def _init_numba_extension():
    from . import _numba
