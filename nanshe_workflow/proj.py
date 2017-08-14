__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Nov 10, 2015 16:41$"


import itertools
import math
import numbers

import numpy
import zarr

from builtins import (
    map as imap,
    range as irange,
    zip as izip,
)

from yail.core import sliding_window_filled

from nanshe_workflow.data import DataBlocks
from nanshe_workflow.ipy import display, FloatProgress
from nanshe_workflow.par import block_parallel


def stack_compute_traces_parallel(client, num_frames):
    """
        Links the client to a returned closure that computes traces.

        Args:
            client(Client):       client to send computations to.

            num_frames(int):      number of frames to compute at a time (
                                  similar to block size).

        Returns:
            callable:             parallelized callable.
    """

    def compute_traces(imagestack, rois, out=None):
        if out is None:
            traces_dtype = numpy.dtype(float)
            if issubclass(imagestack.dtype.type, numpy.floating):
                traces_dtype = imagestack.dtype

            traces = numpy.empty((len(rois), len(imagestack)), dtype=traces_dtype)
        else:
            traces = out

        trace_func = lambda d, r: zarr.array(d[...][:, r[...]].mean(axis=1))

        num_frame_groups = 0
        trace_halo_blocks = []
        roi_halo_blocks = []
        for num_frame_groups, (j_0, j_1) in enumerate(
                sliding_window_filled(
                    itertools.chain(
                        irange(0, len(imagestack), num_frames),
                        [len(imagestack)]),
                    2
                ),
                start=1
        ):
            for i in irange(len(rois)):
                each_trace_halo_block = [slice(j_0, j_1)]
                each_roi_halo_block = [i]
                for k in irange(1, len(imagestack.shape)):
                    each_trace_halo_block.append(slice(None))
                    each_roi_halo_block.append(slice(None))
                each_trace_halo_block = tuple(each_trace_halo_block)
                each_roi_halo_block = tuple(each_roi_halo_block)

                trace_halo_blocks.append(each_trace_halo_block)
                roi_halo_blocks.append(each_roi_halo_block)

        trace_data_blocks = DataBlocks(imagestack, trace_halo_blocks)
        roi_data_blocks = DataBlocks(rois, roi_halo_blocks)

        lview = client.load_balanced_view()
        trace_blocks = lview.map(
            trace_func,
            trace_data_blocks,
            roi_data_blocks
        )

        progress_bar = FloatProgress(
            min=0,
            max=(1 + math.floor((len(imagestack) - 1)/num_frames))
        )
        display(progress_bar)
        for k, ((j, i), each_trace_block) in enumerate(
                izip(
                    itertools.product(
                        irange(num_frame_groups),
                        irange(len(rois))
                    ),
                    trace_blocks
                )
        ):
            progress_bar.value = j
            traces[i, trace_halo_blocks[k][0]] = each_trace_block[...]

        progress_bar.value = progress_bar.max

        return(traces)

    return(compute_traces)


def stack_compute_quantile_projection_parallel(client, num_frames):
    """
        Links client to a returned closure that computes quantile projections.

        Args:
            client(Client):       client to send computations to.

            num_frames(int):      number of frames to compute at a time (
                                  similar to block size).

        Returns:
            callable:             parallelized callable.
    """

    def compute_quantile_projection(data, quantile):
        float_dtype = numpy.dtype(float)
        if issubclass(data.dtype.type, numpy.floating):
           float_dtype = data.dtype

        quantiles = numpy.zeros(data.shape[1:], float_dtype)

        quantile_func = lambda d, out=None: zarr.array(
            numpy.percentile(d[...], q=quantile*100, axis=0, out=out)
        )

        quantile_halo_blocks = []
        for i_0, i_1 in sliding_window_filled(
                itertools.chain(
                    irange(0, len(data), num_frames),
                    [len(data)]),
                2
        ):
            quantile_halo_blocks.append([slice(i_0, i_1)])
            for j in irange(1, len(data.shape)):
                quantile_halo_blocks[-1].append(slice(None))
            quantile_halo_blocks[-1] = tuple(quantile_halo_blocks[-1])

        quantile_data_blocks = DataBlocks(data, quantile_halo_blocks)

        lview = client.load_balanced_view()
        quantile_blocks = lview.map(quantile_func, quantile_data_blocks)

        progress_bar = FloatProgress(
            min=0,
            max=(1 + math.floor((len(data) - 1)/num_frames))
        )
        display(progress_bar)
        for i, each_data_block in enumerate(quantile_blocks):
            progress_bar.value = i
            quantile_func(
                numpy.concatenate(
                    [quantiles[None], each_data_block[...][None]]
                ),
                out=quantiles
            )

        progress_bar.value = progress_bar.max

        return(quantiles)

    return(compute_quantile_projection)


def stack_compute_min_projection_parallel(client, num_frames):
    """
        Links client to a returned closure that computes min projections.

        Args:
            client(Client):       client to send computations to.

            num_frames(int):      number of frames to compute at a time (
                                  similar to block size).

        Returns:
            callable:             parallelized callable.
    """

    def compute_min_projection(data):
        return(
            stack_compute_quantile_projection_parallel(
                client,
                num_frames=num_frames
            )(
                data,
                quantile=0.0
            )
        )
    return(compute_min_projection)


def stack_compute_max_projection_parallel(client, num_frames):
    """
        Links client to a returned closure that computes max projections.

        Args:
            client(Client):       client to send computations to.

            num_frames(int):      number of frames to compute at a time (
                                  similar to block size).

        Returns:
            callable:             parallelized callable.
    """

    def compute_max_projection(data):
        return(
            stack_compute_quantile_projection_parallel(
                client,
                num_frames=num_frames
            )(
                data,
                quantile=1.0
            )
        )
    return(compute_max_projection)


def stack_compute_moment_projections_parallel(client, num_frames):
    """
        Links client to a returned closure that computes moment projections.

        Args:
            client(Client):       client to send computations to.

            num_frames(int):      number of frames to compute at a time (
                                  similar to block size).

        Returns:
            callable:             parallelized callable.
    """

    def compute_moment_projections(data, num_moment):
        moments_dtype = numpy.dtype(float)
        if issubclass(data.dtype.type, numpy.floating):
            moments_dtype = data.dtype

        moments = numpy.zeros((num_moment,) + data.shape[1:], moments_dtype)

        moments_func = lambda d, m: zarr.array(
            (d[...].astype(moments_dtype)**m).sum(axis=0)
        )

        num_frame_groups = 0
        moment_halo_blocks = []
        for num_frame_groups, (i_0, i_1) in enumerate(
                sliding_window_filled(
                    itertools.chain(
                        irange(0, len(data), num_frames),
                        [len(data)]),
                    2
                ),
                start=1
        ):
            for j in irange(len(moments)):
                moment_halo_blocks.append([slice(i_0, i_1)])
                for k in irange(1, len(data.shape)):
                    moment_halo_blocks[-1].append(slice(None))
                moment_halo_blocks[-1] = tuple(moment_halo_blocks[-1])

        moment_data_blocks = DataBlocks(data, moment_halo_blocks)
        moment_ord = itertools.chain(*itertools.repeat(
            irange(num_moment),
            num_frame_groups
        ))

        lview = client.load_balanced_view()
        moment_blocks = lview.map(moments_func, moment_data_blocks, moment_ord)

        progress_bar = FloatProgress(
            min=0,
            max=(1 + math.floor((len(data) - 1)/num_frames))
        )
        display(progress_bar)
        for (i, j), each_moment_block in izip(
                itertools.product(
                    irange(num_frame_groups),
                    irange(len(moments))
                ),
                moment_blocks
        ):
            progress_bar.value = i
            moments[j] += each_moment_block[...]

        progress_bar.value = progress_bar.max

        moments /= len(data)

        return(moments)

    return(compute_moment_projections)


def stack_compute_adj_harmonic_mean_projection_parallel(client, num_frames):
    """
        Links client to a closure that computes the harmonic mean projection.

        Args:
            client(Client):       client to send computations to.

            num_frames(int):      number of frames to compute at a time (
                                  similar to block size).

        Returns:
            callable:             parallelized callable.
    """

    def compute_adj_harmonic_mean_projection(data):
        mean_dtype = numpy.dtype(float)
        if issubclass(data.dtype.type, numpy.floating):
            mean_dtype = data.dtype

        data_adj = stack_compute_min_projection_parallel(client, num_frames)
        data_adj = data_adj(data).min() - mean_dtype.type(1)

        mean = numpy.zeros(data.shape[1:], mean_dtype)

        mean_func = lambda d: zarr.array(
            (numpy.reciprocal(d[...].astype(mean_dtype) - data_adj)).sum(axis=0)
        )

        num_frame_groups = 0
        mean_halo_blocks = []
        for num_frame_groups, (i_0, i_1) in enumerate(
                sliding_window_filled(
                    itertools.chain(
                        irange(0, len(data), num_frames),
                        [len(data)]),
                    2
                ),
                start=1
        ):
            mean_halo_blocks.append([slice(i_0, i_1)])
            for k in irange(1, len(data.shape)):
                mean_halo_blocks[-1].append(slice(None))
            mean_halo_blocks[-1] = tuple(mean_halo_blocks[-1])

        mean_data_blocks = DataBlocks(data, mean_halo_blocks)

        lview = client.load_balanced_view()
        mean_blocks = lview.map(mean_func, mean_data_blocks)

        progress_bar = FloatProgress(
            min=0,
            max=(1 + math.floor((len(data) - 1)/num_frames))
        )
        display(progress_bar)
        for i, each_mean_block in enumerate(mean_blocks):
            progress_bar.value = i
            mean += each_mean_block[...]

        progress_bar.value = progress_bar.max

        mean /= mean_dtype.type(len(data))
        numpy.reciprocal(mean, out=mean)
        mean += data_adj

        return(mean)

    return(compute_adj_harmonic_mean_projection)


def stack_norm_layer_parallel(client, num_frames):
    """
        Links client to a returned closure that computes normalized layers.

        Args:
            client(Client):       client to send computations to.

            num_frames(int):      number of frames to compute at a time (
                                  similar to block size).

        Returns:
            callable:             parallelized callable.
    """

    def norm_layer(data, out=None):
        if out is None:
            out = numpy.empty(data.shape, data.dtype)

        assert len(data.shape) == len(out.shape)
        assert (numpy.array(data.shape) == numpy.array(out.shape)).all()

        data_min = stack_compute_min_projection_parallel(
            client, num_frames=num_frames
        )(data).min()
        data_max = stack_compute_max_projection_parallel(
            client, num_frames=num_frames
        )(data).max()

        calc_block = lambda data_shape,     \
                            data_dtype,     \
                            data_min,       \
                            data_max,       \
                            out_dtype=None, \
                            out=None: ((num_frames,) + data_shape[1:])
        calc_halo = lambda data_shape,      \
                           data_dtype,      \
                           data_min,        \
                           data_max,        \
                           out_dtype=None,  \
                           out=None: (len(data_shape) * (0,))

        def calc(some_data, data_min, data_max, out_dtype=None, out=None):
            if out is None and out_dtype is None:
                out_dtype = some_data.dtype

            if out is None:
                out = numpy.empty(some_data.shape, out_dtype)
            else:
                out_dtype = out.dtype

            if issubclass(out.dtype.type, numbers.Integral):
                out_dtype_info = numpy.iinfo(out.dtype.type)

                out_min = out_dtype_info.min
                out_max = out_dtype_info.max

                data_norm = some_data.astype(float)
            else:
                out_min = 0.0
                out_max = 1.0

                data_norm = some_data.astype(out.dtype)

            scale = (out_max - out_min)/(data_max - data_min)

            data_norm -= data_min
            data_norm *= scale
            data_norm += out_min

            numpy.clip(data_norm, out_min, out_max, out=data_norm)

            out[...] = data_norm

            # Handle out of bound cases.

            match = data_norm / out
            match[numpy.isnan(match)] = 1

            big_match_mask = numpy.isinf(match)
            out[big_match_mask] = out_max
            match[big_match_mask] = 1
            big_match_mask = None

            out[(match == -1) & (numpy.sign(data_norm) == 1)] = out_max
            out[(match == -1) & (numpy.sign(data_norm) == -1)] = out_min

            return(out)

        par_calc = block_parallel(client, calc_block, calc_halo)(calc)

        return(
            par_calc(data, data_min, data_max, out_dtype=out.dtype, out=out)
        )

    return(norm_layer)
