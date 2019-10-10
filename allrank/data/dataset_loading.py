import os

import numpy as np
import torch
from sklearn.datasets import load_svmlight_file
from tensorflow import gfile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import Compose

from allrank.utils.ltr_logging import get_logger

logger = get_logger()
PADDED_Y_VALUE = -1
PADDED_INDEX_VALUE = -1


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        x, y, indices = sample
        return torch.from_numpy(x).type(torch.float32), torch.from_numpy(y).type(torch.float32), torch.from_numpy(indices).type(torch.long)


class FixLength(object):
    """Fix all listings to have equal length by either zero padding or sampling.

    For a given listing, if its length is less than dim_given, it is zero padded to match that length (x's are padded with vectors of zeros,
    y's are padded with zeroes.)

    If its length is greater than dim_given, a random sample of items from that listing is taken to match the dim_given.

    Args:
        dim_given (int): Dimension of x after length fixing operation.
    """

    def __init__(self, dim_given):
        assert isinstance(dim_given, int)
        self.dim_given = dim_given

    def __call__(self, sample):
        sample_size = len(sample[1])
        if sample_size < self.dim_given:  # when expected dimension is larger than number of observation in instance do the padding
            fixed_len_x, fixed_len_y, indices = self._pad(sample, sample_size)
        else:  # otherwise do the sampling
            fixed_len_x, fixed_len_y, indices = self._sample(sample, sample_size)

        return fixed_len_x, fixed_len_y, indices

    def _sample(self, sample, sample_size):
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
        fixed_len_x = np.pad(sample[0], ((0, self.dim_given - sample_size), (0, 0)), "constant")
        fixed_len_y = np.pad(sample[1], (0, self.dim_given - sample_size), "constant", constant_values=PADDED_Y_VALUE)
        indices = np.pad(np.arange(0, sample_size), (0, self.dim_given - sample_size), "constant", constant_values=PADDED_INDEX_VALUE)
        return fixed_len_x, fixed_len_y, indices


class LibSVMDataset(Dataset):
    """LibSVM Learning to Rank dataset."""

    def __init__(self, X, y, query_ids, transform=None):
        """
        Args:
            x (scipy sparse matrix): Features of dataset.
            y (numpy array): Target of dataset.
            query_ids (numpy array): Ids determining group membership.
            transform (callable, optional): Optional transform to be applied
                on a sample.
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
        Args:
            svm_file_path (string): Path to the svm file with data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        x, y, query_ids = load_svmlight_file(svm_file_path, query_id=True)
        logger.info("loaded dataset from {} and got x shape {}, y shape {} and query_ids shape {}".format(
            svm_file_path, x.shape, y.shape, query_ids.shape))
        return cls(x, y, query_ids, transform)

    def __len__(self):
        return len(self.X_by_qid)

    def __getitem__(self, idx):
        X = self.X_by_qid[idx]
        y = self.y_by_qid[idx]

        sample = X, y

        if self.transform:
            sample = self.transform(sample)

        return sample

    @property
    def shape(self):
        batch_dim = len(self)
        document_dim = self.longest_query_length
        features_dim = self[0][0].shape[-1]
        return [batch_dim, document_dim, features_dim]


def load_libsvm_role(input_path: str, role: str) -> LibSVMDataset:
    path = os.path.join(input_path, "{}.txt".format(role))
    logger.info("will load {} data from {}".format(role, path))
    with gfile.Open(path, "rb") as input_stream:
        ds = LibSVMDataset.from_svm_file(input_stream)
    logger.info("{} DS shape: {}".format(role, ds.shape))
    return ds


def fix_length_to_longest_listing(ds: LibSVMDataset) -> Compose:
    logger.info("Will pad to the longest listing: {}".format(ds.longest_query_length))
    return transforms.Compose([FixLength(int(ds.longest_query_length)), ToTensor()])


def load_libsvm_dataset(input_path: str, listing_length: int, validation_ds_role: str):
    train_ds = load_libsvm_role(input_path, "train")
    train_ds.transform = transforms.Compose([FixLength(listing_length), ToTensor()])

    val_ds = load_libsvm_role(input_path, validation_ds_role)
    val_ds.transform = fix_length_to_longest_listing(val_ds)

    return train_ds, val_ds


def create_data_loaders(train_ds, val_ds, num_workers, batch_size):
    gpu_count = torch.cuda.device_count()
    total_batch_size = max(1, gpu_count) * batch_size
    logger.info("total batch size is {}".format(total_batch_size))

    train_dl = DataLoader(train_ds, batch_size=total_batch_size, num_workers=num_workers, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=total_batch_size * 2, num_workers=num_workers, shuffle=False)
    return train_dl, val_dl
