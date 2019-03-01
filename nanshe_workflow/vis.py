__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Nov 10, 2015 19:44$"


import base64
import io
import os
import textwrap

import webcolors

import numpy

import scipy
import scipy.ndimage

from matplotlib.colors import ColorConverter
from matplotlib.cm import gist_rainbow

import bokeh
import bokeh.resources

from builtins import (
    map as imap,
    range as irange,
)

from imgroi.core import find_contours

from xnumpy.core import expand

from yail.core import disperse

from nanshe_workflow.util import hash_file, indent


def get_rgb_array(num_colors):
    """
        Gets an array of different colors.

        Args:
            num_colors(int):        How many colors to generate.

        Returns:
            numpy.ndarray:          A 2D array with the number of colors and
                                    unsigned chars for each of red, green, and
                                    blue.
    """

    old_num_colors = num_colors
    num_colors = num_colors if num_colors > 255 else 255

    colors = []

    rgb_color_values = list(disperse(irange(num_colors)))
    converter = ColorConverter()

    for _ in rgb_color_values:
        a_rgb_color = tuple()
        for __ in converter.to_rgb(gist_rainbow(_)):
             a_rgb_color += ( int(round(255*__)), )

        colors.append(a_rgb_color)

    colors = numpy.asarray(colors, dtype=numpy.uint8)

    if old_num_colors != num_colors:
        colors = colors[:old_num_colors]

    return(colors)


def get_rgba_array(num_colors):
    """
        Gets an array of different colors. The first one is transparent.

        Args:
            num_colors(int):        How many colors to generate.

        Returns:
            numpy.ndarray:          A 2D array with the number of colors and
                                    unsigned chars for each of red, green,
                                    blue, and alpha.
    """

    old_num_colors = num_colors
    num_colors = num_colors if num_colors > 255 else 255

    colors = []
    # Transparent for the zero label
    colors.append((0, 0, 0, 0))

    rgb_color_values = list(disperse(irange(num_colors)))
    converter = ColorConverter()

    for _ in rgb_color_values:
        a_rgb_color = tuple()
        for __ in converter.to_rgba(gist_rainbow(_)):
             a_rgb_color += ( int(round(255*__)), )

        colors.append(a_rgb_color)

    colors = numpy.asarray(colors, dtype=numpy.uint8)

    if old_num_colors != num_colors:
        colors = colors[:old_num_colors+1]

    return(colors)


def get_all_greys():
    """
        Gets an RGB array of 256 greys.

        Returns:
            numpy.ndarray:          A 2D array with the number of colors and
                                    unsigned chars for each of red, green,
                                    blue, and alpha.
    """

    grey_range = numpy.arange(256, dtype=numpy.uint8)
    grey_range = expand(grey_range, 3).copy()
    grey_range = grey_range.tolist()
    grey_range = list(imap(webcolors.rgb_to_hex, grey_range))
    return(grey_range)


def masks_to_contours_2d(mskimg):
    """
        Takes a mask stack and finds contour points for each mask.

        Returns:
            list of lists of ints:    Contour points for each dimension of
                                      each mask.
    """

    st = numpy.zeros((mskimg.ndim - 1) * (3,), dtype=int)
    st[2, 2] = 1
    st[1, 2] = 1
    st[2, 1] = 1
    st[1, 1] = 1

    mskctr_pts_x = []
    mskctr_pts_y = []
    for i in irange(len(mskimg)):
        mskimg_i = scipy.ndimage.binary_dilation(numpy.pad(
            mskimg[i], 2, "constant"), st
        )
        mskctr_i_pts = numpy.array(find_contours(mskimg_i).nonzero()) - 2

        # Sort 2D points clockwise.
        # Map the points to a polar coordinate system relative to the
        # contour's centroid. Then find how they would be sorted by
        # their angle.
        mskctr_i_ctd = mskctr_i_pts.mean(axis=1)
        mskctr_i_pts_radii = mskctr_i_pts - mskctr_i_ctd[:, None]
        mskctr_i_pts_angle = numpy.arctan2(
            mskctr_i_pts_radii[1], mskctr_i_pts_radii[0]
        )
        mskctr_i_pts_ord = numpy.argsort(mskctr_i_pts_angle)
        mskctr_i_pts = mskctr_i_pts[:, mskctr_i_pts_ord.tolist()]

        mskctr_i_y, mskctr_i_x = tuple(mskctr_i_pts)

        mskctr_pts_x.append(mskctr_i_x.tolist())
        mskctr_pts_y.append(mskctr_i_y.tolist())

    return(mskctr_pts_y, mskctr_pts_x)


def generate_cdn(sha_type):
    cdn_block = {}
    html_cdn_tmplt = {
        "css_files": """<link rel="stylesheet" type="text/css" href="{url}" integrity="{sha_type}-{sha_val}" crossorigin="anonymous" />""",
        "js_files": """<script type="text/javascript" src="{url}" integrity="{sha_type}-{sha_val}" crossorigin="anonymous"></script>""",
    }
    for res_type in html_cdn_tmplt.keys():
        cdn_block[res_type] = []
        for fn in getattr(bokeh.resources.Resources(mode="absolute"), res_type):
            url = "https://cdnjs.cloudflare.com/ajax/libs/bokeh/{ver}/{fn}".format(
                ver=bokeh.__version__, fn=os.path.basename(fn)
            )
            cdn_block[res_type].append(
                html_cdn_tmplt[res_type].format(
                    url=url, sha_type=sha_type, sha_val=base64.b64encode(hash_file(fn, sha_type)).decode()
                )
            )
        cdn_block[res_type] = "\n".join(cdn_block[res_type])

    cdn_block = "\n\n".join(cdn_block.values())

    return cdn_block


def write_html(filename, title, div, script, cdn):
    html_tmplt = textwrap.dedent(u"""\
        <html lang="en">
            <head>
                <meta charset="utf-8">
                <title>{title}</title>
                {cdn}
                <style>
                  html {{
                    width: 100%;
                    height: 100%;
                  }}
                  body {{
                    width: 90%;
                    height: 100%;
                    margin: auto;
                    background-color: black;
                  }}
                </style>
            </head>
            <body>
                {div}
                {script}
            </body>
        </html>
    """)

    html_cont = html_tmplt.format(
        title=title,
        div=indent(div, 8),
        script=indent(script, 8),
        cdn=indent(cdn, 8),
    )

    with io.open(filename, "w") as fh:
        fh.write(html_cont)
