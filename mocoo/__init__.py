from .agent import MoCoO

__all__ = ['MoCoO']

__version__ = '0.0.3'

# Optional evaluation subpackage -- imported lazily so that missing
# scientific-computing dependencies (numpy, sklearn, scipy) do not break the
# core MoCoO import.
try:
    from . import evaluation  # noqa: F401
    __all__ = __all__ + ['evaluation']
except ImportError:
    pass

# Optional visualization subpackage -- imported lazily so that missing
# plotting dependencies (matplotlib, seaborn) do not break the core import.
try:
    from . import visualization  # noqa: F401
    __all__ = __all__ + ['visualization']
except ImportError:
    pass

# Configs subpackage -- always available (pure-Python fallback when PyYAML
# is not installed).
try:
    from . import configs  # noqa: F401
    __all__ = __all__ + ['configs']
except ImportError:
    pass
