__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Nov 05, 2015 13:54$"


from contextlib import contextmanager
import os
import shutil

import h5py
import numpy
import zarr

from past.builtins import unicode


def io_remove(name):
    if not os.path.exists(name):
        return
    elif os.path.isfile(name):
        os.remove(name)
    elif os.path.isdir(name):
        shutil.rmtree(name)
    else:
        raise ValueError("Unable to remove path, '%s'." % name)


@contextmanager
def open_zarr(name, mode="r"):
    store = zarr.DirectoryStore(name)
    yield zarr.open_group(store, mode)


def hdf5_to_zarr(hdf5_file, zarr_file):
    def copy(name, h5py_obj):
        if isinstance(h5py_obj, h5py.Group):
            zarr_obj = zarr_file.create_group(name)
        elif isinstance(h5py_obj, h5py.Dataset):
            zarr_obj = zarr_file.create_dataset(
                name,
                data=h5py_obj,
                chunks=h5py_obj.chunks
            )
        else:
            raise NotImplementedError(
                "No Zarr type analogue for HDF5 type,"
                " '%s'." % str(type(h5py_obj))
            )

        zarr_obj.attrs.update(h5py_obj.attrs)

    hdf5_file.visititems(copy)


def zarr_to_hdf5(zarr_file, hdf5_file):
    def copy(name, zarr_obj):
        if isinstance(zarr_obj, zarr.Group):
            h5py_obj = hdf5_file.create_group(name)
        elif isinstance(zarr_obj, zarr.Array):
            h5py_obj = hdf5_file.create_dataset(
                name,
                data=zarr_obj,
                chunks=zarr_obj.chunks
            )
        else:
            raise NotImplementedError(
                "No HDF5 type analogue for Zarr type,"
                " '%s'." % str(type(zarr_obj))
            )

        h5py_obj.attrs.update(zarr_obj.attrs)

    zarr_file.visititems(copy)


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
