#!/usr/bin/env python


import os
import shutil
import sys


if not os.environ.get("TEST_NOTEBOOKS"):
    sys.exit(0)

for each in list(sys.argv[1:]) + [
    "data.tif",
    "data.h5",
    "data_traces.h5",
    "data_rois.h5",
    "data.zarr",
    "data_proj.html",
    "dask-worker-space"]:
    if os.path.isfile(each):
        os.remove(each)
    elif os.path.isdir(each):
        shutil.rmtree(each)
