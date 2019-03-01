import contextlib
import gzip
import hashlib
import io
import mmap

from builtins import (
    map as imap,
)


def gzip_compress(data, compresslevel=6):
    compressed = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed,
                       mode="wb",
                       compresslevel=compresslevel) as compressor:
        compressor.write(data)
    return compressed.getvalue()


def hash_file(fn, hn):
    h = hashlib.new(hn)
    with open(fn, "r") as fh:
        with contextlib.closing(mmap.mmap(fh.fileno(), 0, prot=mmap.PROT_READ)) as mm:
            h.update(mm)

    return h.digest()


def indent(text, spaces):
    spaces = " " * int(spaces)
    return "\n".join(imap(lambda l: spaces + l, text.splitlines()))
