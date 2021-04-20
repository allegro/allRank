import os
from typing import Tuple

import numpy as np
import torch
from sklearn.datasets import load_svmlight_file
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import Compose

from allrank.utils.file_utils import open_local_or_gs
from allrank.utils.ltr_logging import get_logger

logger = get_logger()
PADDED_Y_VALUE = -1
PADDED_INDEX_VALUE = -1


class ToTensor(object):
    """
    Wrapper for ndarray->Tensor conversion.
    """
    def __call__(self, sample):
        """
        :param sample: tuple of three ndarrays
        :return: ndarrays converted to tensors
        """
        x, y, indices = sample
        return torch.from_numpy(x).type(torch.float32), torch.from_numpy(y).type(torch.float32), torch.from_numpy(indices).type(torch.long)


class FixLength(object):
    """
    Wrapper for slate transformation to fix its length, either by zero padding or sampling.

    For a given slate, if its length is less than self.dim_given, x's and y's are padded with zeros to match that length.
    If its length is greater than self.dim_given, a random sample of items from that slate is taken to match the self.dim_given.
    """
    def __init__(self, dim_given):
        """
        :param dim_given: dimensionality of x after length fixing operation
        """
        assert isinstance(dim_given, int)
        self.dim_given = dim_given

    def __call__(self, sample):
        """
        :param sample: ndarrays tuple containing features, labels and original ranks of shapes
        [sample_length, features_dim], [sample_length] and [sample_length], respectively
        :return: ndarrays tuple containing features, labels and original ranks of shapes
            [self.dim_given, features_dim], [self.dim_given] and [self.dim_given], respectively
        """
        sample_size = len(sample[1])
        if sample_size < self.dim_given:  # when expected dimension is larger than number of observation in instance do the padding
            fixed_len_x, fixed_len_y, indices = self._pad(sample, sample_size)
        else:  # otherwise do the sampling
            fixed_len_x, fixed_len_y, indices = self._sample(sample, sample_size)

        return fixed_len_x, fixed_len_y, indices

    def _sample(self, sample, sample_size):
        """
        Sampling from a slate longer than self.dim_given.
        :param sample: ndarrays tuple containing features, labels and original ranks of shapes
            [sample_length, features_dim], [sample_length] and [sample_length], respectively
        :param sample_size: target slate length
        :return: ndarrays tuple containing features, labels and original ranks of shapes
            [sample_size, features_dim], [sample_size] and [sample_size]
        """
        indices = np.random.choice(sample_size, self.dim_given, replace=False)
        fixed_len_y = sample[1][indices]
        if fixed_len_y.sum() == 0:
            if sample[1].sum() == 1:
                indices = np.concatenate([np.random.choice(indices, self.dim_given - 1, replace=False), [np.argmax(sample[1])]])
                fixed_len_y = sample[1][indices]
            elif sample[1].sum() > 0:
                return self._sample(sample, sample_size)
        fixed_len_x = sample[0][indices]
        return fixed_len_x, fixed_len_y, indices

    def _pad(self, sample, sample_size):
        """
        Zero padding a slate shorter than self.dim_given
        :param sample: ndarrays tuple containing features, labels and original ranks of shapes
            [sample_length, features_dim], [sample_length] and [sample_length]
        :param sample_size: target slate length
        :return: ndarrays tuple containing features, labels and original ranks of shapes
            [sample_size, features_dim], [sample_size] and [sample_size]
        """
        fixed_len_x = np.pad(sample[0], ((0, self.dim_given - sample_size), (0, 0)), "constant")
        fixed_len_y = np.pad(sample[1], (0, self.dim_given - sample_size), "constant", constant_values=PADDED_Y_VALUE)
        indices = np.pad(np.arange(0, sample_size), (0, self.dim_given - sample_size), "constant", constant_values=PADDED_INDEX_VALUE)
        return fixed_len_x, fixed_len_y, indices


class LibSVMDataset(Dataset):
    """
    LibSVM Learning to Rank dataset.
    """
    def __init__(self, X, y, query_ids, transform=None):
        """
        :param X: scipy sparse matrix containing features of the dataset of shape [dataset_size, features_dim]
        :param y: ndarray containing target labels of shape [dataset_size]
        :param query_ids: ndarray containing group (slate) membership of dataset items of shape [dataset_size, features_dim]
        :param transform: a callable defining an optional transformation called on the dataset
        """
        X = X.toarray()

        groups = np.cumsum(np.unique(query_ids, return_counts=True)[1])

        self.X_by_qid = np.split(X, groups)[:-1]
        self.y_by_qid = np.split(y, groups)[:-1]

        self.longest_query_length = max([len(a) for a in self.X_by_qid])

        logger.info("loaded dataset with {} queries".format(len(self.X_by_qid)))
        logger.info("longest query had {} documents".format(self.longest_query_length))

        self.transform = transform

    @classmethod
    def from_svm_file(cls, svm_file_path, transform=None):
        """
        Instantiate a LibSVMDataset from a LibSVM file path.
        :param svm_file_path: LibSVM file path
        :param transform: a callable defining an optional transformation called on the dataset
        :return: LibSVMDataset instantiated from a given file and with an optional transformation defined
        """
        x, y, query_ids = load_svmlight_file(svm_file_path, query_id=True)
        logger.info("loaded dataset from {} and got x shape {}, y shape {} and query_ids shape {}".format(
            svm_file_path, x.shape, y.shape, query_ids.shape))
        return cls(x, y, query_ids, transform)

    def __len__(self):
        """
        :return: number of groups (slates) in the dataset
        """
        return len(self.X_by_qid)

    def __getitem__(self, idx):
        """
        :param idx: index of a group
        :return: ndarrays tuple containing features and labels of shapes [slate_length, features_dim] and [slate_length], respectively
        """
        X = self.X_by_qid[idx]
        y = self.y_by_qid[idx]

        sample = X, y

        if self.transform:
            sample = self.transform(sample)

        return sample

    @property
    def shape(self):
        """
        :return: shape of the dataset [batch_dim, document_dim, features_dim] where batch_dim is the number of groups
            (slates) and document_dim is the length of the longest group
        """
        batch_dim = len(self)
        document_dim = self.longest_query_length
        features_dim = self[0][0].shape[-1]
        return [batch_dim, document_dim, features_dim]


def load_libsvm_role(input_path: str, role: str) -> LibSVMDataset:
    """
    Helper function loading a LibSVMDataset of a specific role.

    The file can be located either in the local filesystem or in GCS.
    :param input_path: LibSVM file directory
    :param role: dataset role (file name without an extension)
    :return: LibSVMDataset from file {input_path}/{role}.txt
    """
    path = os.path.join(input_path, "{}.txt".format(role))
    logger.info("will load {} data from {}".format(role, path))
    with open_local_or_gs(path, "rb") as input_stream:
        ds = LibSVMDataset.from_svm_file(input_stream)
    logger.info("{} DS shape: {}".format(role, ds.shape))
    return ds


def fix_length_to_longest_slate(ds: LibSVMDataset) -> Compose:
    """
    Helper function returning a transforms.Compose object performing length fixing and tensor conversion.

    Length fixing operation will fix every slate's length to maximum length present in the LibSVMDataset.
    :param ds: LibSVMDataset to transform
    :return: transforms.Compose object
    """
    logger.info("Will pad to the longest slate: {}".format(ds.longest_query_length))
    return transforms.Compose([FixLength(int(ds.longest_query_length)), ToTensor()])


def load_libsvm_dataset(input_path: str, slate_length: int, validation_ds_role: str) \
        -> Tuple[LibSVMDataset, LibSVMDataset]:
    """
    Helper function loading a train LibSVMDataset and a specified validation LibSVMDataset.
    :param input_path: directory containing the LibSVM files
    :param slate_length: target slate length of the training dataset
    :param validation_ds_role: dataset role used for valdation (file name without an extension)
    :return: tuple of LibSVMDatasets containing train and validation datasets,
        where train slates are padded to slate_length and validation slates to val_ds.longest_query_length
    """
    train_ds = load_libsvm_dataset_role("train", input_path, slate_length)

    val_ds = load_libsvm_dataset_role(validation_ds_role, input_path, slate_length)

    return train_ds, val_ds


def load_libsvm_dataset_role(role: str, input_path: str, slate_length: int) -> LibSVMDataset:
    """
    Helper function loading a single role LibSVMDataset
    :param role: the role of the dataset - specifies file name and padding behaviour
    :param input_path: directory containing the LibSVM files
    :param slate_length: target slate length of the training dataset
    :return: loaded LibSVMDataset
    """
    ds = load_libsvm_role(input_path, role)
    if role == "train":
        ds.transform = transforms.Compose([FixLength(slate_length), ToTensor()])
    else:
        ds.transform = fix_length_to_longest_slate(ds)
    return ds


def create_data_loaders(train_ds: LibSVMDataset, val_ds: LibSVMDataset, num_workers: int, batch_size: int):
    """
    Helper function creating train and validation data loaders with specified number of workers and batch sizes.
    :param train_ds: LibSVMDataset train dataset
    :param val_ds: LibSVMDataset validation dataset
    :param num_workers: number of data loader workers
    :param batch_size: size of the batches returned by the data loaders
    :return: tuple containing train and validation DataLoader objects
    """
    # We are multiplying the batch size by the processing units count
    gpu_count = torch.cuda.device_count()
    total_batch_size = max(1, gpu_count) * batch_size
    logger.info("total batch size is {}".format(total_batch_size))

    # Please note that the batch size for validation dataloader is twice the total_batch_size
    train_dl = DataLoader(train_ds, batch_size=total_batch_size, num_workers=num_workers, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=total_batch_size, num_workers=num_workers, shuffle=False)
    return train_dl, val_dl
