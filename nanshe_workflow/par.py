__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Nov 09, 2015 12:47$"


import copy
import gc
import math
import numbers
import os
from time import sleep

from psutil import cpu_count

import numpy

from builtins import zip as izip

from kenjutsu.measure import len_slices
from kenjutsu.blocks import split_blocks

from metawrap.metawrap import tied_call_args, unwrap

from nanshe_workflow.data import DataBlocks
from nanshe_workflow.ipy import Client, display, FloatProgress


def set_num_workers(num_workers=None):
    """
        Sets environment variable ``$CORES`` based on the number of workers.

        Note:
            If the number of workers is ``None`` or ``-1``, then the number of
            workers will be set to ``1`` less than ``$CORES`` (if set) or the
            number of logical cores as determined by ``psutil``.

        Args:
            num_workers(int):       The number of workers for the cluster.

        Returns:
            num_workers(int):       The number of workers that will be used.
    """

    if (num_workers is None) or (num_workers == -1):
        num_workers = int(os.environ.get("CORES", cpu_count()))
        num_workers -= 1
    else:
        assert isinstance(num_workers, numbers.Integral), \
               "Number of workers must be an integeral value."
        num_workers = int(num_workers)
        assert num_workers > 0, "Must have at least 1 worker."

    os.environ["CORES"] = str(num_workers + 1)

    return(num_workers)


def cleanup_cluster_files(profile):
    """
        Cleans up iPython cluster files before startup and after shutdown.

        Args:
            profile(str):           Which iPython profile to clean up for.
    """

    try:
        # iPython 4.x solution
        from IPython.paths import locate_profile
    except ImportError:
        # iPython 3.x solution
        from IPython.utils.path import locate_profile

    try:
        # iPython 3.x solution (use iPython 4.x name)
        from IPython.utils.path import get_security_file as find_connection_file
    except ImportError:
        # iPython 4.x solution
        from ipykernel.connect import find_connection_file

    for each_file in ["tasks.db", "tasks.db-journal"]:
        try:
            os.remove(os.path.join(locate_profile(profile), each_file))
        except OSError:
            pass

    for each_file in [profile + "_engines", profile + "_controller"]:
        try:
            os.remove(each_file)
        except OSError:
            pass

    for each_file in ["ipcontroller-client.json", "ipcontroller-engine.json"]:
        try:
            os.remove(find_connection_file(each_file, profile=profile))
        except IOError:
            pass
        except OSError:
            pass


def get_client(profile):
    """
        Sets up returns an active client with engines connected.

        Args:
            profile(str):           Which iPython profile to get client for.

        Returns:
            Client:                 A client to the specified iPython cluster.
    """

    client = None
    while client is None:
        try:
            client = Client(profile=profile)
        except IOError:
            sleep(1.0)

    while not client.ids:
        sleep(1.0)

    return(client)


def block_parallel(client, calculate_block_shape, calculate_halo):
    """
        Take a single core function and construct a form that can work on
        haloed blocks in parallel.

        Notes:

            To do this we need access to the client responsible for submitting
            jobs. Also, we need to estimate a block's shape and halo.

            We make some assumptions here.
            * There is only one piece of data that we will need to block.
            * All blocks are position invariant. (means their size is
              independent of their position)
            * All blocks will have a halo that is position invariant.

        Args:
            client(Client):                     client to send computations to.

            calculate_block_shape(callable):    computes blocks shape by having
                                                a similar signature to
                                                ``calculate`` except the data's
                                                shape and dtype are passed in
                                                first.

            calculate_halo(callable):           computes halo shape by having
                                                a similar signature to
                                                ``calculate`` except the data's
                                                shape and dtype are passed in
                                                first

        Returns:
            callable:                           parallel version of
                                                ``calculate`` computed on the
                                                iPython Cluster.
    """

    def build_block_parallel(calculate):
        def wrapper(data, *args, **kwargs):
            client[:].apply(gc.collect).get()
            gc.collect()

            ordered_bound_args, new_args, new_kwargs = tied_call_args(
                unwrap(calculate), data, *args, **kwargs
            )

            out = None
            if "out" in ordered_bound_args:
                out = ordered_bound_args.pop("out")
            elif "out" in new_kwargs:
                out = new_kwargs.pop("out")

            if out is None:
                out = numpy.empty(
                    data.shape,
                    data.dtype
                )

            new_args = tuple(ordered_bound_args.values())[1:len(args)+1] + new_args
            new_kwargs.update(dict(list(ordered_bound_args.items())[len(args)+1:]))
            ordered_bound_args = None

            block_shape = None
            if callable(calculate_block_shape):
                block_shape = calculate_block_shape(
                    data.shape, data.dtype, *new_args, **new_kwargs
                )
            else:
                block_shape = calculate_block_shape
                assert (
                    isinstance(block_shape, tuple) and
                    len(block_shape) == len(data.shape)
                )

            block_halo = None
            if callable(calculate_halo):
                block_halo = calculate_halo(
                    data.shape, data.dtype, *new_args, **new_kwargs)
            else:
                block_halo = calculate_halo
                assert (
                    isinstance(block_halo, tuple) and
                    len(block_halo) == len(data.shape)
                )

            data_blocks, data_halo_blocks, result_halos_trim = split_blocks(
                data.shape, block_shape, block_halo
            )

            lview = client.load_balanced_view()

            calculate_block = lambda dhb, rht: (
                calculate(dhb[...], *new_args, **new_kwargs)[rht]
            )
            result_blocks = lview.map(
                calculate_block,
                DataBlocks(data, data_halo_blocks),
                result_halos_trim
            )

            progress_bar = FloatProgress(min=0.0, max=1.0)
            display(progress_bar)
            for i, (each_data_block, each_result_block) in enumerate(
                    izip(data_blocks, result_blocks)
            ):
                progress_bar.value = i / float(len(result_blocks))
                out[each_data_block] = each_result_block

            progress_bar.value = 1.0

            client[:].apply(gc.collect).get()
            gc.collect()

            return(out)

        return(wrapper)

    return(build_block_parallel)


def shape_block_parallel(client):
    """
        Same as ``block_parallel``, but with restructured argument order.

        Args:
            client(Client):           client to send computations to.

        Returns:
            callable:                 parallelized callable.
    """

    def prebuild_shape_block_parallel(calculate):
        def build_shape_block_parallel(calculate_block_shape, calculate_halo):
            return(
                block_parallel(
                    client, calculate_block_shape, calculate_halo
                )(calculate)
            )

        return(build_shape_block_parallel)

    return(prebuild_shape_block_parallel)


def halo_block_parallel(client, calculate_halo):
    """
        Same as ``block_parallel``, but with restructured argument order.

        Args:
            client(Client):                client to send computations to.

            calculate_halo(callable):      computes halo shape by having a
                                           similar signature to ``calculate``
                                           except the data's shape and dtype
                                           are passed in first.

        Returns:
            callable:                      parallelized callable.
    """

    def prebuild_shape_block_parallel(calculate):
        def build_shape_block_parallel(calculate_block_shape):
            return(
                block_parallel(
                    client, calculate_block_shape, calculate_halo
                )(calculate)
            )

        return(build_shape_block_parallel)

    return(prebuild_shape_block_parallel)


def block_generate_dictionary_parallel(client, calculate_block_shape, calculate_halo=None):
    """
        Take a single core dictionary learning function and construct a form
        that can work on haloed blocks in parallel.

        Notes:

            To do this we need access to the client responsible for submitting
            jobs. Also, we need to estimate a block's shape and halo.

            We make some assumptions here.
            * There is only one piece of data that we will need to block.
            * All blocks are position invariant. (means their size is
              independent of their position)
            * All blocks will have a halo that is position invariant.

        Args:
            client(Client):                     client to send computations to.

            calculate_block_shape(callable):    computes blocks shape by having
                                                a similar signature to
                                                ``calculate`` except the data's
                                                shape and dtype are passed in
                                                first.

            calculate_halo(callable):           computes halo shape by having
                                                a similar signature to
                                                ``calculate`` except the data's
                                                shape and dtype are passed in
                                                first

        Returns:
            callable:                           parallel version of
                                                ``calculate`` computed on the
                                                iPython Cluster.
    """

    assert calculate_halo is None

    def build_block_parallel(calculate):
        def wrapper(data, *args, **kwargs):
            client[:].apply(gc.collect).get()
            gc.collect()

            ordered_bound_args, new_args, new_kwargs = tied_call_args(
                unwrap(calculate), data, *args, **kwargs
            )

            if "initial_dictionary" in ordered_bound_args:
                ordered_bound_args.pop("initial_dictionary")
            elif "initial_dictionary" in new_kwargs:
                new_kwargs.pop("initial_dictionary")

            n_components = None
            if "n_components" in ordered_bound_args:
                n_components = ordered_bound_args.pop("n_components")
            elif "n_components" in new_kwargs:
                n_components = new_kwargs.pop("n_components")

            if n_components is None:
                raise ValueError("Must define `n_components`.")

            out = None
            if "out" in ordered_bound_args:
                out = ordered_bound_args.pop("out")
            elif "out" in new_kwargs:
                out = new_kwargs.pop("out")

            if out is None:
                out = numpy.empty(
                    (n_components,) + data.shape[1:],
                    data.dtype
                )

            new_args = tuple(list(ordered_bound_args.values())[1:len(args)+1]) + new_args
            new_kwargs.update(dict(list(ordered_bound_args.items())[len(args)+1:]))
            ordered_bound_args = None

            block_shape = None
            if callable(calculate_block_shape):
                block_shape = calculate_block_shape(
                    data.shape, data.dtype, *new_args, **new_kwargs
                )
            else:
                block_shape = calculate_block_shape
                assert isinstance(block_shape, tuple) and len(block_shape) == len(data.shape)

            block_halo = calculate_halo

            # Compute how many basis images per block. Take into account some blocks may be smaller.
            data_shape_0_q, data_shape_0_r = divmod(
                data.shape[0], block_shape[0]
            )
            full_block_k, full_block_k_rem = divmod(
                n_components,
                data_shape_0_q + bool(data_shape_0_r)
            )
            data_shape_0_r_k_diff = (
                full_block_k -
                int(math.ceil(full_block_k * data_shape_0_r / float(block_shape[0])))
            ) % full_block_k
            full_block_k, full_block_k_rem = (
                numpy.array(divmod(data_shape_0_r_k_diff, data_shape_0_q)) +
                numpy.array([full_block_k, full_block_k_rem])
            )
            full_block_accum_1 = int(math.ceil((data_shape_0_q)/float(full_block_k_rem))) if full_block_k_rem else 0
            end_block_k = n_components
            end_block_k -= (
                data_shape_0_q - (1 - bool(data_shape_0_r))
            ) * full_block_k
            end_block_k -= data_shape_0_q / full_block_accum_1 if full_block_accum_1 else 0

            data_blocks, data_halo_blocks, result_halos_trim = split_blocks(
                data.shape, block_shape, block_halo
            )

            lview = client.load_balanced_view()

            result_blocks_loc = []
            data_blocks_kwargs = []
            frame_dict_sample = dict()
            data_blocks_dict_sample = []
            k_offset = 0
            for i, each_data_block in enumerate(data_blocks):
                each_kwargs = copy.deepcopy(new_kwargs)

                if each_data_block[0].start == 0:
                    j = 0
                    k_offset = 0

                if (each_data_block[0].stop + 1) != data.shape[0]:
                    each_n_components = full_block_k
                    if ((j + 1) % full_block_accum_1) == 0:
                        each_n_components += 1
                else:
                    # This block is shorter than normal.
                    each_n_components = end_block_k

                j += 1
                new_k_offset = min(
                    n_components,
                    k_offset + each_n_components
                )
                each_n_components = new_k_offset - k_offset
                each_kwargs["n_components"] = each_n_components
                each_result_block_loc = []
                each_result_block_loc.append(slice(
                    k_offset,
                    new_k_offset,
                    1
                ))
                each_result_block_loc += each_data_block[1:]
                each_result_block_loc = tuple(each_result_block_loc)

                k_offset = new_k_offset
                result_blocks_loc.append(each_result_block_loc)
                data_blocks_kwargs.append(each_kwargs)

                each_data_block_time_key = (
                    each_data_block[0].start,
                    each_data_block[0].stop,
                    each_data_block[0].step
                )
                if each_data_block_time_key not in frame_dict_sample:
                    frame_dict_sample[each_data_block_time_key] = numpy.random.choice(
                        numpy.arange(*each_data_block_time_key),
                        each_n_components,
                        replace=False
                    ).tolist()
                data_blocks_dict_sample.append(
                    (frame_dict_sample[each_data_block_time_key],) +
                    each_data_block[1:]
                )

            class DataBlocksDictSampleType(object):
                def __init__(self, data, data_blocks_dict_sample):
                    self.data = data
                    self.data_blocks_dict_sample = data_blocks_dict_sample

                def __iter__(self):
                    for each_data_block_dict_sample in self.data_blocks_dict_sample:
                        try:
                            yield self.data[each_data_block_dict_sample]
                        except TypeError:
                            each_data_block_dict = numpy.empty(
                                (len(each_data_block_dict_sample[0]),) +
                                len_slices(each_data_block_dict_sample[1:]),
                                dtype=self.data.dtype
                            )
                            for i, j in enumerate(each_data_block_dict_sample[0]):
                                each_data_block_dict[i] = self.data[
                                    (slice(j, j+1, 1),) +
                                    each_data_block_dict_sample[1:]
                                ]

                            yield each_data_block_dict

                def __len__(self):
                    return(len(self.data_blocks_dict_sample))

            calculate_block = lambda db, dbds, kw: calculate(
                db[...], dbds[...], *new_args, **kw
            )
            result_blocks = lview.map(
                calculate_block,
                DataBlocks(data, data_blocks),
                DataBlocksDictSampleType(data, data_blocks_dict_sample),
                data_blocks_kwargs
            )

            progress_bar = FloatProgress(min=0.0, max=1.0)
            display(progress_bar)
            for i, (each_data_block, each_result_blocks_loc, each_result_block) in enumerate(
                    izip(data_blocks, result_blocks_loc, result_blocks)
            ):
                progress_bar.value = i / float(len(result_blocks))

                out[each_result_blocks_loc] = each_result_block

            progress_bar.value = 1.0

            client[:].apply(gc.collect).get()
            gc.collect()

            return(out)

        return(wrapper)

    return(build_block_parallel)


def stack_compute_subtract_parallel(client, num_frames):
    """
        Subtract an image by a frame.

        Args:
            client(Client):       client to send computations to.

            num_frames(int):      number of frames to compute at a time (
                                  similar to block size).

        Returns:
            callable:             parallelized callable.
    """


    def wrapper(data1, data2, out=None):
        client[:].apply(gc.collect).get()
        gc.collect()

        data2 = data2[...]

        if out is None:
            out = numpy.empty(
                data1.shape,
                data1.dtype
            )

        block_shape = (num_frames,) + data1.shape[1:]

        data_blocks, data_halo_blocks, result_halos_trim = split_blocks(
            data1.shape, block_shape
        )

        lview = client.load_balanced_view()

        calculate_block = lambda dhb, rht: (
            (dhb[...] - data2[...])[rht]
        )
        result_blocks = lview.map(
            calculate_block,
            DataBlocks(data1, data_halo_blocks),
            result_halos_trim
        )

        progress_bar = FloatProgress(min=0.0, max=1.0)
        display(progress_bar)
        for i, (each_data_block, each_result_block) in enumerate(
                izip(data_blocks, result_blocks)
        ):
            progress_bar.value = i / float(len(result_blocks))
            out[each_data_block] = each_result_block

        progress_bar.value = 1.0

        client[:].apply(gc.collect).get()
        gc.collect()

        return(out)

    return(wrapper)


def shape_block_generate_dictionary_parallel(client):
    """
        Same as ``block_generate_dictionary_parallel``, but with restructured
        argument order.

        Args:
            client(Client):           client to send computations to.

        Returns:
            callable:                 parallelized callable.
    """

    def prebuild_shape_block_parallel(calculate):
        def build_shape_block_parallel(calculate_block_shape, calculate_halo):
            return(block_generate_dictionary_parallel(client, calculate_block_shape, calculate_halo)(calculate))

        return(build_shape_block_parallel)

    return(prebuild_shape_block_parallel)


def halo_block_generate_dictionary_parallel(client, calculate_halo):
    """
        Same as ``block_generate_dictionary_parallel``, but with restructured
        argument order.

        Args:
            client(Client):                client to send computations to.

            calculate_halo(callable):      computes halo shape by having a
                                           similar signature to ``calculate``
                                           except the data's shape and dtype
                                           are passed in first.

        Returns:
            callable:                 parallelized callable.
    """

    def prebuild_shape_block_parallel(calculate):
        def build_shape_block_parallel(calculate_block_shape):
            return(block_generate_dictionary_parallel(client, calculate_block_shape, calculate_halo)(calculate))

        return(build_shape_block_parallel)

    return(prebuild_shape_block_parallel)


def frame_stack_calculate_parallel(client, calculate):
    """
        Wraps a frame stack parallel function to delay specification of number
        of frames per block.

        Args:
            client(Client):                client to send computations to.

            calculate(callable):           a frame stack parallel function to
                                           wrap.

        Returns:
            callable:                      parallelized callable.
    """

    def prebuild_frame_stack_calculate_parallel(num_frames):
        return(calculate(client, num_frames))
    return(prebuild_frame_stack_calculate_parallel)
