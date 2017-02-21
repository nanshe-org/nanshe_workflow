__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Nov 05, 2015 13:54$"


from contextlib import contextmanager

import h5py
import numpy

from past.builtins import unicode


class DataBlocks(object):
    def __init__(self, data, data_blocks):
        self.data = data
        self.data_blocks = data_blocks

    def __iter__(self):
        for each_data_block in self.data_blocks:
            yield self.data[each_data_block]

    def __len__(self):
        return(len(self.data_blocks))


class LazyDataset(object):
    class LazyDatasetSelection(object):
        def __init__(self, filename, datasetname, key, shape, dtype, size):
            self.filename = filename
            self.datasetname = datasetname
            self.key = key
            self.shape = shape
            self.dtype = dtype
            self.size = size

        def __getitem__(self, key):
            pass

    def __init__(self, filename, datasetname):
        pass

    def __getitem__(self, key):
        pass

    def __len__(self):
        return(self.shape[0])

    @contextmanager
    def astype(self, dtype):
        yield None


class LazyHDF5Dataset(LazyDataset):
    class LazyHDF5DatasetSelection(LazyDataset.LazyDatasetSelection):
        def __getitem__(self, key):
            with h5py.File(self.filename, "r") as filehandle:
                dataset = filehandle[self.datasetname]
                with dataset.astype(self.dtype):
                    try:
                        return(dataset[self.key][key])
                    except TypeError:
                        key_sort = tuple()
                        key_rsort = tuple()
                        for each_key in self.key:
                            if (
                                isinstance(each_key, slice) or
                                isinstance(each_key, int) or
                                isinstance(each_key, str) or
                                isinstance(each_key, unicode)
                            ):
                                key_sort += (each_key,)
                                key_rsort += (slice(None),)
                                continue

                            each_key = numpy.array(each_key)
                            each_key_sort = numpy.argsort(each_key)
                            each_key_rsort = numpy.concatenate([
                                each_key_sort[None],
                                numpy.arange(len(each_key_sort))[None]
                            ]).T
                            each_key_rsort = numpy.array(
                                list(tuple(_) for _ in each_key_rsort),
                                dtype=[("sort", int), ("rsort", int)]
                            )
                            each_key_rsort.sort(order="sort")
                            each_key_rsort = each_key_rsort["rsort"].copy()

                            each_key_sort = each_key[each_key_sort]

                            key_sort += (each_key_sort,)
                            key_rsort += (each_key_rsort,)

                        return(dataset[key_sort][key_rsort][key])

    def __init__(self, filename, datasetname):
        self.filename = filename
        self.datasetname = datasetname

        with h5py.File(self.filename, "r") as filehandle:
            dataset = filehandle[self.datasetname]

            self.shape = dataset.shape
            self.dtype = dataset.dtype

        self.size = numpy.prod(self.shape)

    def __getitem__(self, key):
        return(
            LazyHDF5Dataset.LazyHDF5DatasetSelection(
                self.filename,
                self.datasetname,
                key,
                self.shape,
                self.dtype,
                self.size
            )
        )

    @contextmanager
    def astype(self, dtype):
        self_astype = LazyHDF5Dataset(self.filename, self.datasetname)
        self_astype.dtype = numpy.dtype(dtype)

        yield(self_astype)
