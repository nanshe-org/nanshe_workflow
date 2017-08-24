from __future__ import division

__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Aug 14, 2017 09:52$"


import numbers

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

    traces = rois[:, None] * imagestack[None]
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


def compute_adj_harmonic_mean_projection(data):
    """
        Compute the adjusted harmonic mean projection of the data.

        Shifts the data positively to avoid issues with negative or zero
        values that could cause strange results.

        Args:
            data(array):        Dask Array of image data

        Returns:
            Dask Array:         Adjusted harmonic mean projection
    """

    if not issubclass(data.real.dtype.type, numpy.floating):
        data = data.astype(float)

    data_shift = data.min() - data.dtype.type(1)

    data_adj_inv = dask.array.reciprocal(data - data_shift)
    data_adj_inv_mean = data_adj_inv.mean(axis=0)
    data_adj_hmean = dask.array.reciprocal(data_adj_inv_mean) + data_shift

    return data_adj_hmean


def norm_layer(data, out_dtype):
    """
        Compute data rescaled using the bounds of the output type.

        Args:
            data(array):        Dask Array of image data
            out_dtype(type):    type of the data to output

        Returns:
            Dask Array:         Adjusted harmonic mean projection
    """

    out = data.astype(float)
    out_dtype = numpy.dtype(out_dtype)

    data_min = data.min()
    data_max = data.max()

    out_nan = float("nan")
    out_min = float(0)
    out_max = float(1)
    if issubclass(out_dtype.type, numbers.Integral):
        out_dtype_info = numpy.iinfo(out_dtype.type)

        out_nan = float(0)
        out_min = float(out_dtype_info.min)
        out_max = float(out_dtype_info.max)

    scale = (out_max - out_min) / (data_max - data_min)

    out -= data_min
    out *= scale
    out += out_min

    out = dask.array.clip(out, out_min, out_max)

    out_isnan = dask.array.isnan(out)
    out_isinf = dask.array.isinf(out)
    out_sign = dask.array.sign(out)

    out[out_isinf & (out_sign == -1)] = out_min
    out[out_isinf & (out_sign == 1)] = out_max

    out[out_isnan] = out_nan

    out = out.astype(out_dtype)

    return out
