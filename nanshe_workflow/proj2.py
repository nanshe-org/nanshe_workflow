__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Aug 14, 2017 09:52$"


import numpy


def compute_traces(imagestack, rois):
    """
        Constructs Dask computation of traces.

        Args:
            imagestack(array):      Dask Array of image data
            rois(array):            Dask Array of ROI masks

        Returns:
            Dask Array:             Traces for each ROI
    """

    if not issubclass(imagestack.real.dtype.type, numpy.floating):
        imagestack = imagestack.astype(float)

    traces = (
        rois[:, None].repeat(len(imagestack), axis=1) *
        imagestack[None].repeat(len(rois), axis=0)
    )
    traces = (
        traces.sum(axis=tuple(range(2, traces.ndim))) /
        rois.sum(axis=tuple(range(1, rois.ndim)))[:, None]
    )

    return traces
