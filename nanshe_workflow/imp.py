__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Nov 10, 2015 16:28$"


import itertools

import numpy
import zarr

from builtins import range as irange

from nanshe.imp.segment import get_empty_neuron,  \
                               merge_neuron_sets, \
                               wavelet_denoising

from nanshe_workflow.data import DataBlocks
from nanshe_workflow.ipy import display, FloatProgress


def extract_f0_halo(data_shape, data_dtype,
                    half_window_size,
                    which_quantile,
                    temporal_smoothing_gaussian_filter_stdev,
                    temporal_smoothing_gaussian_filter_window_size,
                    spatial_smoothing_gaussian_filter_stdev,
                    spatial_smoothing_gaussian_filter_window_size,
                    bias=None,
                    out=None,
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

    halo = numpy.zeros((len(data_shape),), dtype=int)

    halo[0] = int(numpy.ceil(
        temporal_smoothing_gaussian_filter_window_size *
        temporal_smoothing_gaussian_filter_stdev
    ))
    halo[1:] = int(numpy.ceil(
        spatial_smoothing_gaussian_filter_window_size *
        spatial_smoothing_gaussian_filter_stdev
    ))
    halo[0] = max(halo[0], half_window_size)

    halo = tuple(halo.tolist())

    return(halo)


def wavelet_transform_halo(data_shape, data_dtype,
                           scale=5,
                           include_intermediates=False,
                           include_lower_scales=False,
                           out=None):
    """
        Compute halo for ``wavelet_transform`` given parameters.

        Notes:
            Shape and dtype refer to the data to be used as input. See
            ``wavelet_transform`` documentation for other parameters.

        Returns:
            tuple of ints:         Half halo shape to be tacked on to the data.
    """

    try:
        scale_iter = enumerate(scale)
    except TypeError:
        scale_iter = enumerate(itertools.repeat(scale, len(data_shape)))

    half_halo = list(itertools.repeat(0, len(data_shape)))
    for i, each_scale in scale_iter:
        half_halo_i = 0
        for j in irange(1, 1+each_scale):
            half_halo_i += 2**j

        half_halo[i] = half_halo_i

    half_halo = tuple(half_halo)

    return(half_halo)


def normalize_data_halo(data_shape, data_dtype, out=None, **parameters):
    """
        Compute halo for ``normalize_data`` given parameters.

        Notes:
            Shape and dtype refer to the data to be used as input. See
            ``wavelet_transform`` documentation for other parameters.

        Returns:
            tuple of ints:         Half halo shape to be tacked on to the data.
    """

    half_halo = tuple(itertools.repeat(0, len(data_shape)))

    return(half_halo)


def block_postprocess_data_parallel(client):
    """
        Links the client to a returned closure that runs ``postprocess_data``.

        This not an embarrassingly parallel process due to the merge step. So,
        this instead tries to perform the ROI extraction in parallel, which is
        embarrassingly parallel and then performs merging here in serial as
        extracted ROIs arrive.

        Args:
            client(Client):                client to send computations to.

        Returns:
            callable:                 parallelized callable.
    """

    def postprocess_data(new_dictionary, **parameters):
        # Get `concurrent.futures` compatible `Executor`.
        # Tries the `distributed` syntax and falls back for `ipyparallel`.
        try:
            executor = client.get_executor()
        except AttributeError:
            executor = client.executor()

        data_halo_blocks = []
        for i in irange(len(new_dictionary)):
            data_halo_blocks.append([i])
            for j in irange(1, len(new_dictionary.shape)):
                data_halo_blocks[-1].append(slice(None))
            data_halo_blocks[-1] = tuple(data_halo_blocks[-1])

        data_blocks = DataBlocks(new_dictionary, data_halo_blocks)

        result_blocks = executor.map(
            lambda d: (
                wavelet_denoising(d[...], **parameters["wavelet_denoising"])
            ),
            data_blocks
        )

        new_neurons_set = get_empty_neuron(
            shape=new_dictionary.shape[1:], dtype=new_dictionary.dtype
        )
        progress_bar = FloatProgress(min=0.0, max=1.0)
        display(progress_bar)
        for i, each_result_block in enumerate(result_blocks):
            progress_bar.value = i / float(len(data_blocks))
            new_neurons_set = merge_neuron_sets(
                new_neurons_set,
                each_result_block[...],
                **parameters["merge_neuron_sets"]
            )

        progress_bar.value = 1.0

        return(new_neurons_set)

    return(postprocess_data)
