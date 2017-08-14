__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Aug 14, 2017 09:52$"


import dask
import dask.array
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


def compute_min_projection(data):
    """
        Compute the minimum projection of the data.

        Args:
            data(array):      Dask Array of image data

        Returns:
            Dask Array:       Minimum projection
    """

    return data.min(axis=0)


def compute_max_projection(data):
    """
        Compute the maximum projection of the data.

        Args:
            data(array):      Dask Array of image data

        Returns:
            Dask Array:       Minimum projection
    """

    return data.max(axis=0)


def compute_moment_projections(data, num_moment):
    """
        Compute the moment projections of the data.

        Args:
            data(array):        Dask Array of image data
            num_moment(int):    number of moments to compute

        Returns:
            Dask Array:         Moment projections
    """

    if not issubclass(data.real.dtype.type, numpy.floating):
        data = data.astype(float)

    mom_range = dask.array.arange(num_moment, chunks=(1,))
    mom_range = mom_range[(Ellipsis,) + data.ndim * (None,)]

    moments = (data[None].repeat(num_moment, axis=0) ** mom_range).mean(axis=1)

    return moments
