#!/usr/bin/env python


import operator
import os
import sys

import numpy
import tifffile

from builtins import range as irange

import xnumpy
import xnumpy.core

import nanshe


if not os.environ.get("TEST_NOTEBOOKS"):
    sys.exit(0)

space = numpy.array([110, 120])
radii = numpy.array([6, 6, 6, 6, 7, 6])
magnitudes = numpy.array([15, 16, 15, 17, 16, 16])
points = numpy.array([[30, 24],
                      [59, 65],
                      [21, 65],
                      [80, 78],
                      [72, 16],
                      [45, 32]])

bases_indices = [[1, 3, 4], [0, 2], [5]]
linspace_length = 25

masks = nanshe.syn.data.generate_hypersphere_masks(space, points, radii)
images = nanshe.syn.data.generate_gaussian_images(space,
                                                  points,
                                                  radii/3.0,
                                                  magnitudes) * masks

bases_masks = numpy.zeros(
    (len(bases_indices),) + masks.shape[1:], dtype=masks.dtype
)
bases_images = numpy.zeros(
    (len(bases_indices),) + images.shape[1:], dtype=images.dtype
)

for i, each_basis_indices in enumerate(bases_indices):
    bases_masks[i] = masks[list(each_basis_indices)].max(axis=0)
    bases_images[i] = images[list(each_basis_indices)].max(axis=0)

image_stack = None
ramp = numpy.concatenate([
    numpy.linspace(0, 1, linspace_length),
    numpy.linspace(1, 0, linspace_length)
])

image_stack = numpy.zeros(
    (bases_images.shape[0] * len(ramp),) + bases_images.shape[1:],
    dtype=bases_images.dtype
)
for i in irange(len(bases_images)):
    image_stack_slice = slice(i * len(ramp), (i+1) * len(ramp), 1)

    image_stack[image_stack_slice] = xnumpy.core.permute_op(
        operator.mul,
        ramp,
        bases_images[i]
    )

image_stack *= numpy.iinfo(numpy.uint16).max / image_stack.max()
image_stack = image_stack.astype(numpy.uint16)

with tifffile.TiffWriter("reg.tif", bigtiff=True) as f:
    for i in irange(len(image_stack)):
        f.save(image_stack[i])
