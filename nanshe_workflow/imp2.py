__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Aug 09, 2017 17:12$"


import numpy

from builtins import range as irange

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
