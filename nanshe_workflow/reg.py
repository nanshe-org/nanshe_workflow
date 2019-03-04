from builtins import range as irange

import numpy
import scipy.ndimage

import dask
import dask.array

try:
    from dask.array import blockwise as da_blockwise
except ImportError:
    from dask.array import atop as da_blockwise


def fourier_shift_wrap(array, shift):
    result = numpy.empty_like(array)
    for i in irange(len(array)):
        result[i] = scipy.ndimage.fourier_shift(array[i], shift[0][i])
    return result


def find_best_match(matches):
    best_match = numpy.zeros(
        matches.shape[:1],
        dtype=matches.dtype
    )
    if matches.size:
        i = numpy.argmin((matches ** 2).sum(axis=0))
        best_match = matches[:, i]

    return best_match


def compute_offset(match_mask):
    while type(match_mask) is list:
        match_mask = match_mask[0]

    result = numpy.empty((len(match_mask), match_mask.ndim - 1), dtype=int)
    for i in irange(len(match_mask)):
        match_mask_i = match_mask[i]

        frame_shape = numpy.array(match_mask_i.shape)
        half_frame_shape = frame_shape // 2

        matches = numpy.array(match_mask_i.nonzero())
        above = (matches > half_frame_shape[:, None]).astype(matches.dtype)
        matches -= above * frame_shape[:, None]

        result[i] = find_best_match(matches)

    return result


def roll_frames_chunk(frames, shifts):
    # Needed as Dask shares objects and we plan to write to it.
    # Also if there is only one refcount the old object is freed.
    frames = numpy.copy(frames)

    for i in irange(len(frames)):
        frames[i] = numpy.roll(
            frames[i],
            tuple(shifts[i]),
            axis=tuple(irange(frames.ndim - 1))
        )

    return frames


def roll_frames(frames, shifts):
    frames = frames.rechunk({
        k: v for k, v in enumerate(frames.shape[1:], 1)
    })
    shifts = shifts.rechunk({1: shifts.shape[1]})

    rolled_frames = da_blockwise(
        roll_frames_chunk, tuple(irange(frames.ndim)),
        frames, tuple(irange(frames.ndim)),
        shifts, (0, frames.ndim),
        dtype=frames.dtype,
        concatenate=True
    )

    return rolled_frames
