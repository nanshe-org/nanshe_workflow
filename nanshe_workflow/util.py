import gzip
import io


def gzip_compress(data, compresslevel=6):
    compressed = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed,
                       mode="wb",
                       compresslevel=compresslevel) as compressor:
        compressor.write(data)
    return compressed.getvalue()
