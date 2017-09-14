from __future__ import division


__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Aug 09, 2017 17:12$"


import itertools

import numpy

import dask
import dask.array
import dask.array.linalg

from builtins import range as irange

from nanshe.imp.filters.wavelet import transform as _wavelet_transform
from nanshe.imp.segment import extract_f0 as _extract_f0


def extract_f0(new_data,
               half_window_size,
               which_quantile,
               temporal_smoothing_gaussian_filter_stdev,
               temporal_smoothing_gaussian_filter_window_size,
               spatial_smoothing_gaussian_filter_stdev,
               spatial_smoothing_gaussian_filter_window_size,
               bias=None,
               return_f0=False,
               **parameters):
    """
        Compute ``extract_f0`` on Dask Arrays.

        See the nanshe function for more details

        Returns:
            Dask Array:    A lazily computed result.
    """

    if "out" in parameters:
        raise TypeError("Got an unexpected keyword argument 'out'.")

    new_data = new_data.astype(numpy.float32)

    if bias is None:
        bias = 1 - new_data.min()

    depth = numpy.zeros((new_data.ndim,), dtype=int)

    depth[0] = int(numpy.ceil(
        temporal_smoothing_gaussian_filter_window_size *
        temporal_smoothing_gaussian_filter_stdev
    ))
    depth[1:] = int(numpy.ceil(
        spatial_smoothing_gaussian_filter_window_size *
        spatial_smoothing_gaussian_filter_stdev
    ))
    depth[0] = max(depth[0], half_window_size)

    depth = tuple(depth.tolist())

    boundary = len(depth) * ["none"]
    # Workaround for a bug in Dask with 0 depth.
    #
    # ref: https://github.com/dask/dask/issues/2258
    #
    for i in irange(len(depth)):
        if boundary[i] == "none" and depth[i] == 0:
            boundary[i] = "reflect"

    boundary = tuple(boundary)

    result = new_data.map_overlap(
        func=_extract_f0,
        depth=depth,
        boundary=boundary,
        dtype=numpy.float32,
        half_window_size=half_window_size,
        which_quantile=which_quantile,
        temporal_smoothing_gaussian_filter_stdev=temporal_smoothing_gaussian_filter_stdev,
        temporal_smoothing_gaussian_filter_window_size=temporal_smoothing_gaussian_filter_window_size,
        spatial_smoothing_gaussian_filter_stdev=spatial_smoothing_gaussian_filter_stdev,
        spatial_smoothing_gaussian_filter_window_size=spatial_smoothing_gaussian_filter_window_size,
        bias=bias,
        return_f0=return_f0,
        **parameters
    )

    return(result)


def wavelet_transform(im0,
                      scale=5):
    """
        Compute ``wavelet_transform`` on Dask Arrays.

        See the nanshe function for more details

        Returns:
            Dask Array:    A lazily computed result.
    """

    im0 = im0.astype(numpy.float32)

    try:
        scale_iter = enumerate(scale)
    except TypeError:
        scale_iter = enumerate(itertools.repeat(scale, im0.ndim))

    depth = list(itertools.repeat(0, im0.ndim))
    for i, each_scale in scale_iter:
        depth_i = 0
        for j in irange(1, 1 + each_scale):
            depth_i += 2 ** j

        depth[i] = depth_i

    depth = tuple(depth)

    boundary = len(depth) * ["none"]
    # Workaround for a bug in Dask with 0 depth.
    #
    # ref: https://github.com/dask/dask/issues/2258
    #
    for i in irange(len(depth)):
        if boundary[i] == "none" and depth[i] == 0:
            boundary[i] = "reflect"

    boundary = tuple(boundary)

    result = im0.map_overlap(
        func=_wavelet_transform,
        depth=depth,
        boundary=boundary,
        dtype=numpy.float32,
        scale=scale
    )

    return(result)


def zeroed_mean_images(new_data):
    """
        Compute ``zeroed_mean_images`` on Dask Arrays.

        See the nanshe function for more details

        Returns:
            Dask Array:    A lazily computed result.
    """

    new_data_means = new_data.mean(
        axis=tuple(irange(1, new_data.ndim)),
        keepdims=True
    )

    new_data_zeroed = new_data - new_data_means

    return new_data_zeroed


def renormalized_images(input_array, ord=2):
    """
        Compute ``renormalized_images`` on Dask Arrays.

        See the nanshe function for more details

        Returns:
            Dask Array:    A lazily computed result.
    """

    input_array_norms = dask.array.linalg.norm(
        input_array.reshape((len(input_array), -1)),
        ord=ord,
        axis=1
    )
    input_array_norms = input_array_norms[
        (slice(None),) + (input_array.ndim - input_array_norms.ndim) * (None,)
    ]

    input_array_renormed = dask.array.where(
        input_array_norms != 0, input_array / input_array_norms, input_array
    )

    return(input_array_renormed)


def normalize_data(new_data, **parameters):
    """
        Compute ``normalize_data`` on Dask Arrays.

        See the nanshe function for more details

        Returns:
            Dask Array:    A lazily computed result.
    """

    if "out" in parameters:
        raise TypeError("Got an unexpected keyword argument 'out'.")

    ord = parameters.get("renormalized_images", {}).get("ord", 2)

    if ord is None:
        ord = 2

    new_data_zeroed = zeroed_mean_images(new_data)

    new_data_norms = dask.array.linalg.norm(
        new_data_zeroed.reshape((len(new_data_zeroed), -1)),
        ord=ord,
        axis=1
    )
    new_data_norms = new_data_norms[
        (slice(None),) + (new_data_zeroed.ndim - new_data_norms.ndim) * (None,)
    ]

    new_data_renormed = dask.array.where(
        new_data_norms != 0, new_data_zeroed / new_data_norms, new_data_zeroed
    )

    return(new_data_renormed)
