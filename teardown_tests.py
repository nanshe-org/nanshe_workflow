#!/usr/bin/env python


import os
import shutil
import sys


if not os.environ.get("TEST_NOTEBOOKS"):
    sys.exit(0)

for each in list(sys.argv[1:]) + [
    "reg.h5",
    "reg_sub.h5",
    "reg_f_f0.h5",
    "reg_wt.h5",
    "reg_norm.h5",
    "reg_dict.h5",
    "reg_post.h5",
    "reg_traces.h5",
    "reg_rois.h5",
    "reg_proj.h5",
    "reg.zarr",
    "reg_sub.zarr",
    "reg_f_f0.zarr",
    "reg_wt.zarr",
    "reg_norm.zarr",
    "reg_dict.zarr",
    "reg_post.zarr",
    "reg_traces.zarr",
    "reg_rois.zarr",
    "reg_proj.zarr",
    "reg_proj.html"]:
    if os.path.isfile(each):
        os.remove(each)
    elif os.path.isdir(each):
        shutil.rmtree(each)
