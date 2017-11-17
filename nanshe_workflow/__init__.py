__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Oct 26, 2015 16:03$"

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

import os
os.environ["DASK_CONFIG"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "dask_config.yaml"
)
del os
