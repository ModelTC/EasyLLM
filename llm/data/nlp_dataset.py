import torch
from torch.utils.data import Dataset, ConcatDataset
import json
import os
import numpy as np
import random

from llm.utils.env import dist_env
from llm.utils.general.registry_factory import DATASET_REGISTRY
from llm.utils.tools.petrel_helper import PetrelHelper
from llm.utils.general.log_helper import default_logger as logger
import mmap
from multiprocessing.pool import ThreadPool as Pool

from .nlp_transforms import build_transformer
from .data_utils import index_file_path, data_file_path, _warmup_mmap_file
from .data_utils import MMapIndex, _num_tokens, _num_epochs, build_data_cache


IGNORE_INDEX = -100


@DATASET_REGISTRY.register('pretrain_bin')
class PretrainBinDataset(Dataset):
    def __init__(self,
                 txt_files='',
                 folders=None,
                 tokenizer=None,
                 transformer=None,
                 bin_size=256 * 1024,
                 seq_length=4096,
                 data_root=''):
        super(PretrainBinDataset, self).__init__()
        if not isinstance(txt_files, list):
            txt_files = [txt_files]
        if folders is not None:
            if not isinstance(folders, list):
                folders = [folders]
            txt_files = self.get_txt_files_from_folder(folders)
        self.bin_paths = self.load_bin_paths(txt_files)
        self.data_root = data_root
        self.index_buffer = self.get_index_buffer(bin_size, seq_length)
        self.bin_indexs = self.build_bin_index(self.bin_paths, bin_size, seq_length)
        if transformer is not None:
            # add datset handler in transform kwargs in need of mosaic/mixup etc.
            for trans in transformer:
                if 'kwargs' in trans and trans['kwargs'].get('with_tokenizer', False):
                    trans['kwargs']['tokenizer'] = tokenizer
                    trans['kwargs'].pop('with_tokenizer')
            self.transformer = build_transformer(transformer)
        else:
            self.transformer = None

    def get_txt_files_from_folder(self, folders):
        txt_files = []
        for folder in folders:
            for item in os.listdir(folder):
                if 'txt' in item:
                    txt_files.append(os.path.join(folder, item))
        return txt_files

    def load_bin_paths(self, txt_files):
        bin_paths = []
        for txt_file in txt_files:
            with open(txt_file, "r") as f:
                for line in f.readlines():
                    bin_paths.append(line.strip())
        return bin_paths

    def get_index_buffer(self, bin_size, seq_length):
        indexs = []
        temp = list(range(0, bin_size + seq_length, seq_length))
        for i in range(len(temp) - 1):
            indexs.append([0, temp[i], temp[i + 1]])
        indexs = np.array(indexs).astype(np.int32)
        return indexs

    def get_last_indexs(self, idx, size, seq_length):
        index = []
        num = size // seq_length + 1
        for i in range(num):
            index.append([idx, i * seq_length, (i + 1) * seq_length])
        index = np.array(index).astype(np.int32)
        return index

    def build_bin_index(self, bin_paths, bin_size, seq_length):
        bin_indexs = []
        for idx, bin_path in enumerate(bin_paths):
            base_name = os.path.basename(bin_path)
            size = int(base_name.split('_')[0])
            if size == bin_size:
                indexs = np.copy(self.index_buffer)
                for i in range(len(indexs)):
                    indexs[i][0] = idx
            else:
                indexs = self.get_last_indexs(idx, size, seq_length)
            bin_indexs.append(indexs)
            if idx % 10000 == 0:
                logger.info(f"building bin index {idx}")
        bin_indexs = np.vstack(bin_indexs)
        return bin_indexs

    def get_meta(self, idx):
        bin_index = self.bin_indexs[idx]
        path = os.path.join(self.data_root, self.bin_paths[bin_index[0]])
        meta = {
            'path': path,
            'bin_index': bin_index
        }
        if self.transformer is not None:
            meta = self.transformer(meta)
        return meta

    def __getitem__(self, idx):
        try:
            meta = self.get_meta(idx)
        except: # noqa
            meta = None
        while meta is None:
            new_idx = random.randint(0, len(self.metas) - 1)
            if idx == new_idx:
                continue
            meta = self.get_meta(new_idx)
        return meta

    def __len__(self):
        """
        Returns dataset length
        """
        return len(self.bin_indexs)


@DATASET_REGISTRY.register('base_nlp_json')
class BaseNLPJsonDataset(Dataset):
    def __init__(self,
                 json_file,
                 tokenizer=None,
                 transformer=None,
                 json_type='all'):
        super(BaseNLPJsonDataset, self).__init__()
        if not isinstance(json_file, list):
            json_file = [json_file]
        if not isinstance(json_type, list):
            if isinstance(json_file, list):
                json_type = [json_type] * len(json_file)
            else:
                json_type = [json_type]
        self.metas = self.load_metas(json_file, json_type=json_type)
        if transformer is not None:
            # add datset handler in transform kwargs in need of mosaic/mixup etc.
            for trans in transformer:
                if 'kwargs' in trans and trans['kwargs'].get('with_tokenizer', False):
                    trans['kwargs']['tokenizer'] = tokenizer
                    trans['kwargs'].pop('with_tokenizer')
            self.transformer = build_transformer(transformer)
        else:
            self.transformer = None

    def load_metas(self, json_files, json_type=['all']):
        metas = []
        for idx, json_file in enumerate(json_files):
            if json_type[idx] == 'all':
                temp = PetrelHelper.load_json(json_file)
                metas.extend(temp)
            elif json_type[idx] == 'line':
                with PetrelHelper.open(json_file) as f:
                    for i, line in enumerate(f):
                        meta = json.loads(line)
                        metas.append(meta)
                        if ((i + 1) % 1000 == 0):
                            logger.info('{} items of data have been loaded'.format(i + 1))
        return metas

    def get_meta(self, idx):
        meta = self.metas[idx]
        if self.transformer is not None:
            meta = self.transformer(meta)
        return meta

    def __getitem__(self, idx):
        meta = self.get_meta(idx)
        while meta is None:
            new_idx = random.randint(0, len(self.metas) - 1)
            if idx == new_idx:
                continue
            meta = self.get_meta(new_idx)
        return meta

    def __len__(self):
        """
        Returns dataset length
        """
        return len(self.metas)


@DATASET_REGISTRY.register('mmap_json')
class MMapJsonDataset(Dataset):
    def __init__(self,
                 json_file,
                 folder=None,
                 tokenizer=None,
                 transformer=None,
                 json_type='line',
                 cache_dir='./cache',
                 cache_worker=4):
        super(MMapJsonDataset, self).__init__()
        if folder is not None:
            json_file = self.build_json_files(folder)
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir
        if not isinstance(json_file, list):
            json_file = [json_file]
        if not isinstance(json_type, list):
            if isinstance(json_file, list):
                json_type = [json_type] * len(json_file)
            else:
                json_type = [json_type]
        self.cache_worker = cache_worker
        self.load_position_map(json_file)
        self.build_mmap(json_file)
        if transformer is not None:
            # add datset handler in transform kwargs in need of mosaic/mixup etc.
            for trans in transformer:
                if 'kwargs' in trans and trans['kwargs'].get('with_tokenizer', False):
                    trans['kwargs']['tokenizer'] = tokenizer
                    trans['kwargs'].pop('with_tokenizer')
            self.transformer = build_transformer(transformer)
        else:
            self.transformer = None

    def build_json_files(self, folder):
        json_file = []
        for root, dirs, files in os.walk(folder, topdown=False):
            for name in files:
                json_file.append(os.path.join(root, name))
        return json_file

    def _build_mmap_func(self, json_meta):
        json_file, idx = json_meta[0], json_meta[1]
        with open(json_file, 'r') as f:
            self.mmap_data[idx] = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    def build_mmap(self, json_files):
        self.mmap_data = []
        for _ in range(len(json_files)):
            self.mmap_data.append(None)
        with Pool(self.cache_worker) as p:
            _ = p.map(self._build_mmap_func, zip(json_files, range(len(json_files))))

    def _load_position_map_func(self, json_meta):
        dataset_size = 0
        json_file, file_idx = json_meta[0], json_meta[1]
        data_pos_path = json_file.replace('/', '_').split('.')[0] + '_pos.txt'
        if os.path.exists(os.path.join(self.cache_dir, data_pos_path)):
            with open(os.path.join(self.cache_dir, data_pos_path), "r") as f:
                for line in f.readlines():
                    temp = line.strip().split()
                    self.position_map[file_idx].append([int(temp[0]), int(temp[1]), int(temp[2])])
                    dataset_size += 1
            logger.info(f"load mmap dataset position from {self.cache_dir}, dataset_size: {dataset_size}")
        else:
            if (dist_env.get_pipeline_model_parallel_rank() == 0) and (dist_env.get_tensor_model_parallel_rank() == 0):
                with open(json_file, 'r', encoding='utf-8') as fr:
                    with open(os.path.join(self.cache_dir, data_pos_path), "w") as fw:
                        last_size = 0
                        cur_size = 0
                        for line in fr.readlines():
                            line_size = len(line.encode())
                            cur_size += line_size
                            begin_end_idx = [last_size, cur_size, file_idx]
                            self.position_map[file_idx].append(begin_end_idx)
                            print(f"{begin_end_idx[0]} {begin_end_idx[1]} {begin_end_idx[2]}", file=fw)
                            last_size = cur_size
                            dataset_size += 1
            if torch.distributed.is_initialized():
                torch.distributed.barrier(group=dist_env.get_pipeline_model_parallel_group())
            logger.info(f"cache mmap dataset position in {self.cache_dir}, dataset_size: {dataset_size}")

    def load_position_map(self, json_files):
        self.position_map = []
        for _ in range(len(json_files)):
            self.position_map.append([])
        with Pool(self.cache_worker) as p:
            _ = p.map(self._load_position_map_func, zip(json_files, range(len(json_files))))
        new_position_map = []
        for item in self.position_map:
            new_position_map.extend(item)
        self.position_map = new_position_map

    def __getitem__(self, idx):
        mmap_begin_idx, mmap_end_idx, file_idx = self.position_map[idx]
        data_str = self.mmap_data[file_idx][mmap_begin_idx:mmap_end_idx]
        meta = json.loads(data_str)
        if self.transformer is not None:
            meta = self.transformer(meta)
        return meta

    def __len__(self):
        """
        Returns dataset length
        """
        return len(self.position_map)


@DATASET_REGISTRY.register('mmap_index_json')
class MMapIndexJsonDataset(Dataset):
    def __init__(self,
                 data_root,
                 seed,
                 total_samples,
                 max_seq_length,
                 json_file=None,
                 json_type='line',
                 tokenizer=None,
                 transformer=None,
                 cache_dir='train_data_cache',
                 cache_prefix='mmap_sentebnse',
                 cache_worker=4,
                 cache_skip_warmup=True,
                 cache_log_freq=100,
                 cache_splits=1,
                 cache_location_builder='py',
                 ignore_index=-100,
                 cutoff_last_epoch=0.95):
        super(MMapIndexJsonDataset, self).__init__()
        self.tokenizer = tokenizer
        # build transformer firstly.
        if transformer is not None:
            # add datset handler in transform kwargs in need of mosaic/mixup etc.
            for trans in transformer:
                if 'kwargs' in trans and trans['kwargs'].get('with_tokenizer', False):
                    trans['kwargs']['tokenizer'] = tokenizer
                    trans['kwargs'].pop('with_tokenizer')
            self.transformer = build_transformer(transformer)
        else:
            self.transformer = None
        self.ignore_index = ignore_index

        # build dependency
        if cache_location_builder == 'cpp':
            if (dist_env.get_pipeline_model_parallel_rank() == 0) and (dist_env.get_tensor_model_parallel_rank() == 0):
                from llm.data.cores.compile import compile_helper
                compile_helper()
            if torch.distributed.is_initialized():
                torch.distributed.barrier(group=dist_env.get_pipeline_model_parallel_group())

        # load/build cache data.
        self.cache_prefix = os.path.join(data_root, cache_dir, cache_prefix)
        self.cache_worker = cache_worker
        self.cache_log_freq = cache_log_freq
        if (dist_env.get_pipeline_model_parallel_rank() == 0) and (dist_env.get_tensor_model_parallel_rank() == 0):
            if not self.exists(self.cache_prefix):
                os.makedirs(os.path.join(data_root, cache_dir), exist_ok=True)
                assert json_file is not None, 'you have not a cache .bin&.idx file of cache_prefix: {}, you must provide as least one json file to generate them'.format(self.cache_prefix)      # noqa
                assert json_type == 'line', 'Only support line type json to build the cache .bin&.idx file'
                if not isinstance(json_file, list):
                    json_file = [json_file]
                build_data_cache(self.data_encode, tokenizer, json_file, self.cache_prefix, self.cache_worker, cache_splits=cache_splits)
        if torch.distributed.is_initialized():
            torch.distributed.barrier(group=dist_env.get_pipeline_model_parallel_group())

        self._index = MMapIndex(index_file_path(self.cache_prefix), cache_skip_warmup)

        if not cache_skip_warmup:
            logger.info("    warming up data mmap file...")
            _warmup_mmap_file(data_file_path(self.cache_prefix))
        logger.info("    creating numpy buffer of mmap...")
        self._bin_buffer_mmap = np.memmap(data_file_path(self.cache_prefix), mode='r', order='C')
        logger.info("    creating memory view of numpy buffer...")
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

        total_num_of_sentenses = self._index.sizes.shape[0]
        sentenses_id = np.arange(total_num_of_sentenses, step=1, dtype=np.int32)
        self.cache_location_builder = cache_location_builder
        self.sentenses_idx, self.location_idx, self.shuffle_idx = self.load_index_mappings(
            sentenses_id, self._index.sizes, total_samples, max_seq_length, seed, cutoff_last_epoch)

    def cache_log(self, log_str):
        if (self.cache_log_freq != -1):
            logger.info(log_str)

    def data_encode(self, json_line):
        meta = json.loads(json_line)
        assert self.transformer is not None
        meta = self.transformer(meta)
        return meta

    def exists(self, path):
        return (os.path.exists(index_file_path(path)) and os.path.exists(data_file_path(path)))

    def _build_sentense_idx(self, sentense_id, num_epochs, np_rng, separate_last_epoch):
        """Build an array with length = number-of-epochs * number-of-dcuments.
        Each index is mapped to a corresponding document."""
        if not separate_last_epoch or num_epochs == 1:
            sentense_idx = np.mgrid[0:num_epochs, 0:len(sentense_id)][1]
            sentense_idx[:] = sentense_id
            sentense_idx = sentense_idx.reshape(-1)
            sentense_idx = sentense_idx.astype(np.int32)
            np_rng.shuffle(sentense_idx)
            return sentense_idx

        sentense_idx_first = self._build_sentense_idx(sentense_id, num_epochs - 1, np_rng, False)
        sentense_idx_last = self._build_sentense_idx(sentense_id, 1, np_rng, False)
        return np.concatenate((sentense_idx_first, sentense_idx_last))

    def _build_location_idx(self, sentenses_sizes, sentense_idx, seq_length, num_epochs, tokens_per_epoch):
        # Total number of samples. For -1 see comments in `_num_epochs`.
        num_samples = (num_epochs * tokens_per_epoch - 1) // seq_length
        location_idx = np.zeros([num_samples + 1, 2], dtype=np.int32)

        # Index into sample_idx.
        sample_index = 0
        # Index into doc_idx.
        doc_idx_index = 0
        # Begining offset for each document.
        doc_offset = 0
        # Start with first document and no offset.
        location_idx[sample_index][0] = doc_idx_index
        location_idx[sample_index][1] = doc_offset
        sample_index += 1
        while sample_index <= num_samples:
            # Start with a fresh sequence.
            remaining_seq_length = seq_length + 1
            while remaining_seq_length != 0:
                # Get the document length.
                doc_id = sentense_idx[doc_idx_index]
                doc_length = sentenses_sizes[doc_id] - doc_offset
                # And add it to the current sequence.
                remaining_seq_length -= doc_length
                # If we have more than a full sequence, adjust offset and set
                # remaining length to zero so we return from the while loop.
                # Note that -1 here is for the same reason we have -1 in
                # `_num_epochs` calculations.
                if remaining_seq_length <= 0:
                    doc_offset += (remaining_seq_length + doc_length - 1)
                    remaining_seq_length = 0
                else:
                    # Otherwise, start from the begining of the next document.
                    doc_idx_index += 1
                    doc_offset = 0
            # Record the sequence.
            location_idx[sample_index][0] = doc_idx_index
            location_idx[sample_index][1] = doc_offset
            sample_index += 1
        return location_idx

    def _build_shuffle_idx(self, num_samples, total_size, np_rng):
        """Build the range [0, size) and shuffle."""
        dtype_ = np.uint32
        if total_size >= (np.iinfo(np.uint32).max - 1):
            dtype_ = np.int64

        shuffle_idx_first = np.arange(start=0, stop=num_samples, step=1, dtype=dtype_)
        np_rng.shuffle(shuffle_idx_first)
        if num_samples == total_size:
            return shuffle_idx_first

        shuffle_idx_last = np.arange(start=num_samples, stop=total_size, step=1, dtype=dtype_)
        np_rng.shuffle(shuffle_idx_last)
        return np.concatenate((shuffle_idx_first, shuffle_idx_last))

    def load_index_mappings(self, sentenses_id, sentenses_sizes, total_samples,
                            seq_length, seed, cutoff_last_epoch):
        """Build doc-idx, sample-idx, and shuffle-idx.
        doc-idx: is an array (ordered) of documents to be used in training.
        sample-idx: is the start document index and document offset for each
        training sample.
        shuffle-idx: maps the sample index into a random index into sample-idx.
        """
        # Number of tokens in each epoch and number of required epochs.
        tokens_per_epoch = _num_tokens(sentenses_id, sentenses_sizes)
        num_epochs = _num_epochs(tokens_per_epoch, seq_length, total_samples)
        # rng state
        np_rng = np.random.RandomState(seed=seed)

        # Filename of the index mappings.
        _filename = self.cache_prefix + '_indexmap'
        _filename += '_{}ns'.format(total_samples)
        _filename += '_{}sl'.format(seq_length)
        _filename += '_{}s'.format(seed)
        sen_idx_filename = _filename + '_sentense_idx.npy'
        loc_idx_filename = _filename + '_location_idx.npy'
        shuffle_idx_filename = _filename + '_shuffle_idx.npy'

        # Build the indexed mapping if not exist.
        if (dist_env.get_pipeline_model_parallel_rank() == 0) and (dist_env.get_tensor_model_parallel_rank() == 0):
            if (not os.path.isfile(sen_idx_filename)) or (not os.path.isfile(loc_idx_filename)) or (not os.path.isfile(shuffle_idx_filename)):      # noqa
                logger.info(' > WARNING: could not find index map files, building the indices on rank 0 ...')
                # For the last epoch, decide whether include the entire epoch
                # in the global shuffle or not.

                # If we need only one epoch, then separating last epoch  does
                # not mean anything.
                if num_epochs == 1:
                    separate_last_epoch = False
                    self.cache_log(' > only one epoch required, setting separate_last_epoch to False')
                else:
                    # Get the number of samples for the last epoch
                    num_samples_from_epochs_minus_one = ((num_epochs - 1) * tokens_per_epoch - 1) // seq_length
                    last_epoch_num_samples = total_samples - num_samples_from_epochs_minus_one
                    assert last_epoch_num_samples >= 0, f'last epoch number of samples {last_epoch_num_samples} should be non-negative.'        # noqa
                    num_samples_per_epoch = (tokens_per_epoch - 1) // seq_length
                    assert last_epoch_num_samples <= num_samples_per_epoch, f'last epoch number of samples {last_epoch_num_samples} exceeded max value {num_samples_per_epoch}.'        # noqa
                    # If we have less than cutoff_last_epoch * samples_per_epoch of the samples for the last epoch,
                    # seperate out the epoch and treat it differently.
                    separate_last_epoch = (last_epoch_num_samples < int(cutoff_last_epoch * num_samples_per_epoch))
                    if separate_last_epoch:
                        string = ' > last epoch number of samples ({}) is smaller '\
                            'than {}% of number of samples per epoch ({}), setting separate_last_epoch to True'
                    else:
                        string = ' > last epoch number of samples ({}) is larger '\
                            'than {}% of number of samples per epoch ({}), setting separate_last_epoch to False'
                    self.cache_log(string.format(last_epoch_num_samples, cutoff_last_epoch * 100,
                                                 num_samples_per_epoch))

                # sentense-idx.
                sentense_idx = self._build_sentense_idx(sentenses_id, num_epochs, np_rng, separate_last_epoch)
                np.save(sen_idx_filename, sentense_idx, allow_pickle=True)
                self.cache_log('Build sentense idx file done!')

                # location-idx.
                assert sentense_idx.dtype == np.int32 and sentenses_sizes.dtype == np.int32
                if self.cache_location_builder == 'cpp':
                    # Use C++ implementation for speed.
                    # First compile and then import.
                    from llm.data.cores import helpers
                    location_idx = helpers.build_location_idx(sentenses_sizes, sentense_idx, seq_length,
                                                              num_epochs, tokens_per_epoch)
                elif self.cache_location_builder == 'py':
                    location_idx = self._build_location_idx(sentenses_sizes, sentense_idx, seq_length,
                                                            num_epochs, tokens_per_epoch)
                else:
                    raise NotImplementedError
                np.save(loc_idx_filename, location_idx, allow_pickle=True)
                self.cache_log('Build location idx file done!')

                # shuffle-idx.
                # -1 is due to data structure used to retieve the index:
                #    sample i --> [sample_idx[i], sample_idx[i+1])
                if separate_last_epoch:
                    num_samples_ = num_samples_from_epochs_minus_one
                else:
                    num_samples_ = location_idx.shape[0] - 1
                shuffle_idx = self._build_shuffle_idx(num_samples_, location_idx.shape[0] - 1, np_rng)
                np.save(shuffle_idx_filename, shuffle_idx, allow_pickle=True)
                self.cache_log('Build shuffle idx file done!')

        if torch.distributed.is_initialized():
            torch.distributed.barrier(group=dist_env.get_pipeline_model_parallel_group())

        # Load mappings.
        sentense_idx = np.load(sen_idx_filename, allow_pickle=True, mmap_mode='r')
        locationidx = np.load(loc_idx_filename, allow_pickle=True, mmap_mode='r')
        shuffle_idx = np.load(shuffle_idx_filename, allow_pickle=True, mmap_mode='r')

        return sentense_idx, locationidx, shuffle_idx

    def mmap_get_items(self, idx, offset=0, length=None):
        """ Retrieves a single item from the dataset with the option to only
        return a portion of the item.

        get(idx) is the same as [idx] but get() does not support slicing.
        """
        ptr, size = self._index[idx]
        if length is None:
            length = size - offset
        ptr += offset * np.dtype(self._index.dtype).itemsize
        np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype,
                                 count=length, offset=ptr)
        return np_array

    def __len__(self):
        # -1 is due to data structure used to retieve the index:
        #    sample i --> [sample_idx[i], sample_idx[i+1])
        return self.location_idx.shape[0] - 1

    def __getitem__(self, idx):
        # Get the shuffled index.
        idx = self.shuffle_idx[idx]
        # Start and end documents and offsets.
        doc_index_f = self.location_idx[idx][0]
        doc_index_l = self.location_idx[idx + 1][0]
        offset_f = self.location_idx[idx][1]
        offset_l = self.location_idx[idx + 1][1]
        # If we are within the same document, just extract the chunk.
        if doc_index_f == doc_index_l:
            sample = self.mmap_get_items(self.sentenses_idx[doc_index_f],
                                         offset=offset_f, length=offset_l - offset_f + 1)
        else:
            # Otherwise, get the rest of the initial document.
            sample_list = [self.mmap_get_items(self.sentenses_idx[doc_index_f], offset=offset_f)]
            # Loop over all in between documents and add the entire document.
            for i in range(doc_index_f + 1, doc_index_l):
                sample_list.append(self.mmap_get_items(self.sentenses_idx[i]))
            # And finally add the relevant portion of last document.
            sample_list.append(self.mmap_get_items(
                self.sentenses_idx[doc_index_l],
                length=offset_l + 1))
            sample = np.concatenate(sample_list)

        input_ids = torch.LongTensor(sample.tolist())
        labels = input_ids.clone()
        results = {'input_ids': input_ids, 'labels': labels}
        return results


def build_dataset(cfg_dataset, tokenizer):
    if isinstance(cfg_dataset, list):
        # ConcatDataset
        datasets = [build_dataset(cfg, tokenizer) for cfg in cfg_dataset]
        return ConcatDataset(datasets)
    if 'kwargs' not in cfg_dataset:
        cfg_dataset['kwargs'] = {}
    cfg_dataset['kwargs']['tokenizer'] = tokenizer
    return DATASET_REGISTRY.build(cfg_dataset)
