import struct
import shutil
import numpy as np
from functools import lru_cache

from llm.utils.general.log_helper import default_logger as logger


dtypes = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float32,
    7: np.double,
    8: np.uint16
}


def index_file_path(prefix_path):
    return prefix_path + '.idx'


def data_file_path(prefix_path):
    return prefix_path + '.bin'


def _warmup_mmap_file(path):
    with open(path, 'rb') as stream:
        while stream.read(100 * 1024 * 1024):
            pass


def best_fitting_dtype(vocab_size=None):
    if vocab_size is not None and vocab_size < 65500:
        return np.uint16
    else:
        return np.int32


def exscan_from_cumsum_(arr):
    # given an array holding the result of an inclusive scan (cumsum),
    # convert to an exclusive scan (shift to the right)
    # [10, 30, 35, 50] --> [0, 10, 30, 35]
    if arr.size > 1:
        arr[1:] = arr[:-1]
    if arr.size > 0:
        arr[0] = 0


def get_pointers_with_total(sizes, elemsize, dtype):
    """Return a numpy array of type np.dtype giving the byte offsets.

    Multiplies values in the sizes array by elemsize (bytes),
    and then computes an exclusive scan to get byte offsets.
    Returns the total number of bytes as second item in a tuple.
    """

    # scale values in sizes array by elemsize to get sizes in bytes
    pointers = np.array(sizes, dtype=dtype)
    pointers *= elemsize
    np.cumsum(pointers, axis=0, out=pointers)

    # get total number of bytes from all sizes (last element)
    bytes_last = pointers[-1] if len(sizes) > 0 else 0

    # convert to byte offsets
    exscan_from_cumsum_(pointers)

    return pointers, bytes_last


def code(dtype):
    for k in dtypes.keys():
        if dtypes[k] == dtype:
            return k
    raise ValueError(dtype)


def _num_tokens(documents, sizes):
    """Total number of tokens in the dataset."""
    return np.sum(sizes[documents])


def _num_epochs(tokens_per_epoch, seq_length, num_samples):
    """Based on number of samples and sequence lenght, calculate how many
    epochs will be needed."""
    num_epochs = 0
    total_tokens = 0
    while True:
        num_epochs += 1
        total_tokens += tokens_per_epoch
        # -1 is because we need to retrieve seq_length + 1 token each time
        # but the last token will overlap with the first token of the next
        # sample except for the last sample.
        if ((total_tokens - 1) // seq_length) >= num_samples:
            return num_epochs


class MMapIndex(object):
    _HDR_MAGIC = b'MMIDIDX\x00\x00'

    @staticmethod
    def write_header(fout, dtype, numsizes, numdocs):
        """Writes header for mmap indexed dataset to given file handle, return number of bytes written."""
        startpos = fout.tell()

        fout.write(MMapIndex._HDR_MAGIC)
        fout.write(struct.pack('<Q', 1))
        fout.write(struct.pack('<B', code(dtype)))
        fout.write(struct.pack('<Q', numsizes))
        fout.write(struct.pack('<Q', numdocs))

        endpos = fout.tell()
        return endpos - startpos

    @classmethod
    def writer(cls, path, dtype):
        class _Writer(object):
            def __enter__(self):
                self._file = open(path, 'wb')
                return self

            @staticmethod
            def _get_pointers(sizes, npdtype):
                """Return a numpy array of byte offsets given a list of sizes.

                Multiplies values in the sizes array by dtype size (bytes),
                and then computes an exclusive scan to get byte offsets.
                """

                # compute element sizes in bytes
                pointers, _ = get_pointers_with_total(sizes, dtype().itemsize, npdtype)
                return pointers

            def write(self, sizes, doc_idx):
                MMapIndex.write_header(self._file, dtype, len(sizes), len(doc_idx))

                sizes32 = np.array(sizes, dtype=np.int32)
                self._file.write(sizes32.tobytes(order='C'))
                del sizes32

                pointers = self._get_pointers(sizes, np.int64)
                self._file.write(pointers.tobytes(order='C'))
                del pointers

                doc_idx = np.array(doc_idx, dtype=np.int64)
                self._file.write(doc_idx.tobytes(order='C'))

            def __exit__(self, exc_type, exc_val, exc_tb):
                self._file.close()

        return _Writer()

    def __init__(self, path, skip_warmup=False):
        with open(path, 'rb') as stream:
            magic_test = stream.read(9)
            assert self._HDR_MAGIC == magic_test, (
                'Index file doesn\'t match expected format. '
                'Make sure that --dataset-impl is configured properly.'
            )
            version = struct.unpack('<Q', stream.read(8))
            assert (1,) == version

            dtype_code, = struct.unpack('<B', stream.read(1))
            self._dtype = dtypes[dtype_code]
            self._dtype_size = self._dtype().itemsize

            self._len = struct.unpack('<Q', stream.read(8))[0]
            self._doc_count = struct.unpack('<Q', stream.read(8))[0]
            offset = stream.tell()

        if not skip_warmup:
            logger.info("    warming up index mmap file...")
            _warmup_mmap_file(path)

        self._bin_buffer_mmap = np.memmap(path, mode='r', order='C')
        self._bin_buffer = memoryview(self._bin_buffer_mmap)
        logger.info("    reading sizes...")
        self._sizes = np.frombuffer(
            self._bin_buffer,
            dtype=np.int32,
            count=self._len,
            offset=offset)
        logger.info("    reading pointers...")
        self._pointers = np.frombuffer(self._bin_buffer, dtype=np.int64, count=self._len,
                                       offset=offset + self._sizes.nbytes)
        logger.info("    reading document index...")
        self._doc_idx = np.frombuffer(self._bin_buffer, dtype=np.int64, count=self._doc_count,
                                      offset=offset + self._sizes.nbytes + self._pointers.nbytes)

    def __del__(self):
        self._bin_buffer_mmap._mmap.close()
        del self._bin_buffer_mmap

    @property
    def dtype(self):
        return self._dtype

    @property
    def sizes(self):
        return self._sizes

    @property
    def doc_idx(self):
        return self._doc_idx

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        return self._pointers[i], self._sizes[i]

    def __len__(self):
        return self._len


class MMapIndexedDatasetBuilder(object):
    def __init__(self, out_file, dtype=np.int64):
        self._data_file = open(out_file, 'wb')
        self._dtype = dtype
        self._sizes = []
        self._doc_idx = [0]

    def add_item(self, tensor):
        np_array = np.array(tensor.numpy(), dtype=self._dtype)
        self._data_file.write(np_array.tobytes(order='C'))
        self._sizes.append(np_array.size)

    def end_document(self):
        self._doc_idx.append(len(self._sizes))

    def merge_file_(self, another_file):
        # Concatenate index
        index = MMapIndex(index_file_path(another_file))
        assert index.dtype == self._dtype

        total_len = len(index.sizes) + len(self._sizes)
        print(f"    concat {another_file} size={len(index.sizes)} for a total size of {total_len}")

        offset = len(self._sizes)
        self._sizes.extend(index.sizes)
        self._doc_idx.extend((offset + index.doc_idx)[1:])

        # Concatenate data
        with open(data_file_path(another_file), 'rb') as f:
            shutil.copyfileobj(f, self._data_file)

    def finalize(self, index_file):
        self._data_file.close()

        with MMapIndex.writer(index_file, self._dtype) as index:
            index.write(self._sizes, self._doc_idx)
