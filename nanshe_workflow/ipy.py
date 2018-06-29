__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Nov 10, 2015 17:09$"


try:
    from IPython.utils.shimmodule import ShimWarning
except ImportError:
    class ShimWarning(Warning):
        """Warning issued by IPython 4.x regarding deprecated API."""
        pass

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings('error', '', ShimWarning)

    try:
        # IPython 3
        from IPython.html.widgets import FloatProgress
        from IPython.parallel import Client
    except ShimWarning:
        # IPython 4
        from ipywidgets import FloatProgress
        from ipyparallel import Client

from IPython.display import display
