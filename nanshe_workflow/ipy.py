__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Nov 10, 2015 17:09$"


import json
import re

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

import ipykernel
import notebook.notebookapp

import requests


def check_nbserverproxy():
    """
    Return the url of the current jupyter notebook server.
    """
    kernel_id = re.search(
        "kernel-(.*).json",
        ipykernel.connect.get_connection_file()
    ).group(1)
    servers = notebook.notebookapp.list_running_servers()
    for s in servers:
        response = requests.get(
            requests.compat.urljoin(s["url"], "api/sessions"),
            params={"token": s.get("token", "")}
        )
        for n in json.loads(response.text):
            if n["kernel"]["id"] == kernel_id:
                # Found server that is running this Jupyter Notebook.
                # Try to requests this servers port through nbserverproxy.
                url = requests.compat.urljoin(
                    s["url"], "proxy/%i" % s["port"]
                )
                # If the proxy is running, it will redirect.
                # If not, it will error out.
                try:
                    requests.get(url).raise_for_status()
                except requests.HTTPError:
                    return False
                else:
                    return True
