from __future__ import division


__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Aug 09, 2017 17:12$"


import itertools

import numpy

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
        Compute halo for ``extract_f0`` given parameters.

        Notes:
            Shape and dtype refer to the data to be used as input. See
            ``extract_f0`` documentation for other parameters.

        Returns:
            tuple of ints:         Half halo shape to be tacked on to the data.
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
        Compute halo for ``wavelet_transform`` given parameters.

        Notes:
            Shape and dtype refer to the data to be used as input. See
            ``wavelet_transform`` documentation for other parameters.

        Returns:
            tuple of ints:         Half halo shape to be tacked on to the data.
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


def normalize_data(new_data, **parameters):
    """
        Compute halo for ``normalize_data`` given parameters.

        Notes:
            Shape and dtype refer to the data to be used as input. See
            ``wavelet_transform`` documentation for other parameters.

        Returns:
            tuple of ints:         Half halo shape to be tacked on to the data.
    """

    if "out" in parameters:
        raise TypeError("Got an unexpected keyword argument 'out'.")

    ord = parameters.get("renormalized_images", {}).get("ord", 2)

    if ord is None:
        ord = 2

    new_data_means = new_data.mean(
        axis=tuple(irange(1, new_data.ndim)),
        keepdims=True
    )

    new_data_zeroed = new_data - new_data_means

    if ord == 0:
        new_data_norms = (
            (new_data_zeroed != 0).astype(new_data_zeroed.dtype).sum(
                axis=tuple(irange(1, new_data.ndim)), keepdims=True
            )
        )
    elif ord == numpy.inf:
        new_data_norms = abs(new_data_zeroed).max(
            axis=tuple(irange(1, new_data.ndim)), keepdims=True
        )
    elif ord == -numpy.inf:
        new_data_norms = abs(new_data_zeroed).min(
            axis=tuple(irange(1, new_data.ndim)), keepdims=True
        )
    else:
        new_data_norms = (
            (abs(new_data_zeroed) ** ord).sum(
                axis=tuple(irange(1, new_data.ndim)), keepdims=True
            )
            ** (1.0 / ord)
        )

    new_data_renormed = new_data_zeroed / new_data_norms

    return(new_data_renormed)
