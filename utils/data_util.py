import copy
from abc import ABC, abstractmethod
from collections import deque, defaultdict
from pathlib import Path
from typing import Callable, List, TypeVar, Type, Dict
import numpy as np
import scipy

import FADS

from mpclab_common.pytypes import VehicleState
from scipy.spatial import ConvexHull, Delaunay, QhullError
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm

from utils import pytorch_util as ptu
import os
from loguru import logger
import matplotlib.pyplot as plt
import time
import torch

class StackedLoader:
    """For the balanced training for data from the target domain and the source domain"""
    def __init__(self, loader1, loader2):
        self.loader1 = loader1
        self.loader2 = loader2

    def __iter__(self):
        return self._stacked_iterator()

    def __len__(self):
        return min(len(self.loader1), len(self.loader2))

    def _stacked_iterator(self):
        for batch1, batch2 in zip(self.loader1, self.loader2):
            stacked = self._stack_batches(batch1, batch2)
            yield stacked

    def _stack_batches(self, batch1, batch2):
        # Note: there might be bugs in this method, check it tomorrow
        # Recursively stack batches that are tuples, lists, or tensors
        if isinstance(batch1, torch.Tensor):
            return torch.cat([batch1, batch2], dim=0)
        elif isinstance(batch1, (list, tuple)):
            return type(batch1)(self._stack_batches(b1, b2) for b1, b2 in zip(batch1, batch2))
        elif isinstance(batch1, dict):
            return {k: self._stack_batches(batch1[k], batch2[k]) for k in batch1}
        else:
            raise TypeError("Unsupported batch type for stacking.")


class EfficientReplayBuffer(Dataset, ABC):
    _replay_buffer_name = None

    @property
    def replay_buffer_name(self):
        return self._replay_buffer_name or self.__class__.__name__

    def __init__(self,
                 maxsize=1_000_000, transform=None, random_eviction: bool = True, constants: dict = None,
                 lazy_init=True, name=None):
        """A more memory efficient implementation of the replay buffer with numpy arrays.
        Overwrite _fetch to determine how the dataset interacts with dataloaders.
        """
        super().__init__()
        self.maxsize = maxsize
        self.fields = {}
        self.constants = constants if constants is not None else {}
        self.transform = transform if transform is not None else {}
        self.random_eviction = random_eviction
        self.lazy_init = lazy_init
        self._replay_buffer = name

        self.left = 0
        self.right = 0
        self.size = 0

        self.initialized = False

    def __len__(self):
        return self.size

    def __getitem__(self, index_ext):
        assert index_ext in range(self.size), "Index out of range"
        return self._fetch((index_ext + self.left) % self.maxsize)

    def _fetch(self, index):
        """Note: index is the absolute index in the arrays. """
        # data = {}
        # for k, v in self.fields.items():
        #     if k in self.transform:
        #         data[k] = self.transform[k](v[index])
        #     else:
        #         data[k] = v[index]

        # return {**data, **self.constants}

        data = {}
        extracted = {}
        for k in self.fields.keys():
            extracted[k] = self.fields[k][index]

        for k in self.fields.keys():
            if k in self.transform:
                data[k] = self.transform[k](extracted)
            else:
                data[k] = self.fields[k][index]

        return {**data, **self.constants}


    def add_frame(self, obs, rews, terminated, truncated, info, **kwargs):
        self.append(batched=False,
                    rewards=rews,
                    **obs, **kwargs)

    def initialize(self, batched: bool, size: int = None, **kwargs):
        if self.initialized:
            return
        if self.lazy_init:
            init_func = lambda shape, dtype: np.empty(shape, dtype)
        else:
            init_func = lambda shape, dtype: np.full(shape, 1, dtype)
        if batched:
            for attr, data in kwargs.items():
                self.fields[attr] = init_func((self.maxsize, *data.shape[1:]), dtype=data.dtype)
                # self.fields[attr] = np.empty((self.maxsize, *data.shape[1:]), dtype=type(data))
        else:
            for attr, data in kwargs.items():
                if np.isscalar(data):
                    # self.fields[attr] = np.empty((self.maxsize,), dtype=type(data))
                    self.fields[attr] = init_func((self.maxsize,), dtype=type(data))
                else:
                    # self.fields[attr] = np.empty((self.maxsize, *data.shape), dtype=data.dtype)
                    self.fields[attr] = init_func((self.maxsize, *data.shape), dtype=data.dtype)
        self.initialized = True

    def clear(self):
        self.left = self.right = self.size = 0

    def preprocess(self):
        pass

    def absorb(self, other: 'EfficientReplayBuffer'):
        if not other.initialized:
            return
        self.append(batched=True,
                    size=other.size,
                    **other.consolidate())
        other.clear()

    def get_traj_len(self, batched, size=None, **kwargs):
        if not batched:
            return 1
        traj_len = size
        for data in kwargs.values():
            if traj_len is None:
                traj_len = data.shape[0]
            else:
                assert traj_len == data.shape[0]
        return traj_len

    def get_idxs_for_new_data(self, traj_len) -> np.ndarray:
        """
        Determine the index for the new data. Evict existing data if the insertion leads to overflowing.

        @param traj_len: Length of the data to be inserted.
        @return: An numpy array indicating where the new data should be inserted.
        """
        index = (np.arange(traj_len) + self.right) % self.maxsize  # Determine the index to put the new data first.
        self.right = (self.right + traj_len) % self.maxsize  # Move the right cursor further right.

        n_overflow = self.size + traj_len - self.maxsize
        if n_overflow > 0:
            # If the insertion causes overflow, the elements on the leftmost side of the replay buffer gets overwritten.
            if self.random_eviction:
                # Randomly select n elements and swap them with the ones at the front, so they are evicted instead.
                self.move_random_n_elements_to_front(n=n_overflow)
            # Move the left cursor to indicate the new starting point of the buffer.
            self.left = (self.left + n_overflow) % self.maxsize
            self.size = self.maxsize  # The replay buffer will surely be full after this.
        else:
            self.size += traj_len
        return index

    def append(self, batched: bool, size=None, **kwargs):
        self.initialize(batched, **kwargs)
        traj_len = self.get_traj_len(batched, **kwargs)
        index = self.get_idxs_for_new_data(traj_len)
        for attr, data in kwargs.items():
            self.fields[attr][index] = data
        return traj_len

    def pop(self, idx=None) -> Dict[str, np.ndarray]:
        """
        idx: the external index. Must be in [0, size).
        Note: This implementation doesn't retain the original order!
        """
        if isinstance(idx, int):
            idx = [idx]
        idx = list(set(idx))
        assert all(0 <= x < self.size for x in idx)
        idx = (np.asarray(idx) + self.left) % self.maxsize
        last_k_idx = (self.right - np.arange(len(idx)) - 1) % self.maxsize
        for attr, data in self.fields.items():
            self.fields[attr][[idx, last_k_idx]] = data[[last_k_idx, idx]]
        self.right = (self.right - len(idx)) % self.maxsize
        self.size -= len(idx)
        return {attr: data[last_k_idx] for attr, data in self.fields.items()}

    def popback(self, n) -> None:
        self.right = (self.right - min(n, len(self))) % self.maxsize
        self.size = max(self.size - n, 0)

    def move_random_n_elements_to_front(self, n: int):
        """
        Move <n_evictions> random entries to the front (for eviction).

        @param n: Number of random elements to move to front.
        @return:
        """
        if n <= 0:
            return
        index = (np.random.randint(0, self.size, (n, )) + self.left) % self.maxsize
        other_index = (np.arange(n) + self.left) % self.maxsize

        for attr in self.fields:
            self.fields[attr][[index, other_index]] = self.fields[attr][[other_index, index]]

    def retrieve_entire_field(self, field_name : str):
        """retrieve all the examples of a specific data field in the buffer"""
        if field_name not in self.fields:
            raise ValueError(f"the field name {field_name} is not a attainable field of this data buffer")
        return self.fields[field_name][self.left : self.right, :]

    def consolidate(self) -> Dict[str, np.ndarray]:
        if self.left + self.size <= self.maxsize:
            data = {k: v[self.left: self.right] for k, v in self.fields.items()}
        else:
            data = {k: np.concatenate((v[self.left:], v[:self.right]), axis=0) for k, v in self.fields.items()}
        # self.left, self.right = 0, self.size
        # for field in self.fields:
        #     self.fields[field][:self.size] = data[field]
        return data

    def export(self, path=None, name=None):
        path = path or Path(__file__).resolve().parent.parent / 'data'
        path = Path(path)
        os.makedirs(path, exist_ok=True)
        name = f"{self.replay_buffer_name}{f'_{name}' if name else ''}"
        logger.debug("Saving Replay Buffer...")

        np.savez_compressed(path / f"{name}.npz",
                            size=self.size,
                            **self.consolidate())

        logger.debug("Replay Buffer Saved!")

    def load(self, path=None, name=None):
        path = path or Path(__file__).resolve().parent.parent / 'data'
        path = Path(path)
        name = f"{self.replay_buffer_name}{f'_{name}' if name else ''}"
        if not os.path.exists(path / f"{name}.npz"):
            logger.warning("Replay buffer save file not found!")
            return
        logger.debug("Found the saved replay Buffer. Loading replay buffer...")

        data = np.load(path / f"{name}.npz", mmap_mode='r')
        data = {k: v.copy() for k, v in data.items()}

        self.size = self.left = self.right = 0
        # self.size = self.right = data['size']
        # self.left = 0

        self.initialize(batched=True, **data)
        self.append(batched=True, **data)
        logger.debug(f"The replay buffer is successfully loaded. Current size = {self.__len__()}")

    def dataloader(self, batch_size: int = 64, shuffle: bool = True, num_workers: int = 0, manifest=None) -> DataLoader:
        if manifest is None:
            raise ValueError("Must specify fields to fetch.")

        def collate_fn(batch):
            """
            Custom collate function to extract only the requested features and labels.
            """
            ret = []
            for blocks in manifest:
                ret.append([ptu.from_numpy(np.stack([item[field] for item in batch], axis=0)) for field in blocks])
            return ret

        return DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                          drop_last=True,
                          collate_fn=collate_fn)

    def sample_latest_data(self, traj_len) -> Dict[str, np.ndarray]:
        if traj_len > self.size:
            raise IndexError(f'Current replay buffer size is {self.size}, but {traj_len} examples are requested.')
        idx = (self.right - np.arange(traj_len, 0, -1)) % self.maxsize
        return {k: v[idx] for k, v in self.fields.items()}

    def sample_random_data(self, traj_len) -> Dict[str, np.ndarray]:
        if traj_len > self.size:
            raise IndexError(f'Current replay buffer size is {self.size}, but {traj_len} examples are requested.')
        idx = (np.random.randint(0, self.size, traj_len) + self.left) % self.maxsize
        return {k: v[idx] for k, v in self.fields.items()}

    def is_in_knn_convex_hull(self, query, fields, k: int, threshold=0.25): # original value of threshold = 0.5
        """
        The query must be batched, and have the same dimension as the corresponding field.
        """
        nbr = NearestNeighbors(n_neighbors=k, algorithm='auto')
        # Prepare the data
        if self.left + self.size == self.right:
            data = np.concatenate([self.fields[field][self.left:self.right] for field in fields], axis=-1)
        else:
            data = np.concatenate(
                [np.concatenate((self.fields[field][self.left:], self.fields[field][:self.right]), axis=0) for field in
                 fields], axis=-1)
        # Efficiently subsample data using random choice without full permutation
        n_samples = min(len(data), 32768)
        if n_samples < len(data):
            # indices = np.random.choice(len(data), size=n_samples, replace=False)
            fads = FADS.FADS(data)
            indices = fads.DS(n_samples)
            data = data[indices]
        else:
            data = data.copy()

        nbr.fit(data)
        dists, indices = nbr.radius_neighbors(query, radius=threshold, return_distance=True, sort_results=True)
        # indices = [idx[:k] for idx in indices]
        ret = []
        for q, idx in tqdm(zip(query, indices), total=len(indices), desc='Self-labeling'):
            if len(idx) < q.shape[0] + 1:
                ret.append(False)
                continue
            k_nearest_points = data[idx]
            try:
                hull = ConvexHull(k_nearest_points, qhull_options='QJ Pp')
            except QhullError:
                ret.append(False)
                continue
            if hull.vertices.shape[0] < 8:
                ret.append(False)
                continue
            # Check if q is inside the convex hull using equations
            # Compute dot product of q with each equation's normal and add offset
            vals = np.dot(hull.equations[:, :-1], q) + hull.equations[:, -1]
            # Allow a small tolerance for numerical precision
            is_inside = np.all(vals <= 1e-8)
            ret.append(is_inside)
        return np.asarray(ret)

    # def __is_in_knn_convex_hull(self, query, fields, k: int, threshold=1.):
    #     """
    #     The query must be batched, and have the same dimension as the corresponding field.
    #     """
    #     nbr = NearestNeighbors(n_neighbors=k, algorithm='auto')
    #     if self.left + self.size == self.right:
    #         data = np.concatenate([self.fields[field][self.left:self.right] for field in fields], axis=-1)
    #     else:
    #         data = np.concatenate(
    #             [np.concatenate((self.fields[field][self.left:], self.fields[field][:self.right]), axis=0) for field in
    #              fields], axis=-1)
    #     data = np.random.permutation(data)[:1024]  # Limit the size for comparison. Use randomization to keep the distribution approximately the same.
    #     nbr.fit(data)
    #     # dists, indices = nbr.kneighbors(query)  # dists, indices: (Q, k)
    #     _, indices = nbr.radius_neighbors(query, radius=threshold, return_distance=True, sort_results=True)[:k]
    #     # indices = indices[:k]  # Limiting the size of the neighbors.
    #     ret = []
    #     for q, idx in zip(query, indices):
    #         # k_nearest_points = data[idx][dist <= threshold]
    #         k_nearest_points = data[idx]
    #         if k_nearest_points.shape[0] < q.shape[0] + 1:
    #             ret.append(False)
    #             continue
    #         hull = ConvexHull(k_nearest_points, qhull_options='QJ Pp')
    #         if hull.vertices.shape[0] < 8:
    #             ret.append(False)
    #             continue
    #         delaunay = Delaunay(k_nearest_points[hull.vertices], qhull_options='QJ Pp')
    #         ret.append(delaunay.find_simplex(q) >= 0)
    #     return np.asarray(ret)


class EfficientReplayBufferPN(EfficientReplayBuffer):
    def __init__(self, maxsize: int = 1_000_000, transform=None, random_eviction: bool = True,
                 lazy_init=True):
        self.D_pos = EfficientReplayBuffer(maxsize=maxsize, transform=transform, random_eviction=random_eviction,
                                           constants={'safe': np.array([1.], dtype=np.float32)},
                                           lazy_init=lazy_init, name='D_pos')
        self.D_neg = EfficientReplayBuffer(maxsize=maxsize, transform=transform, random_eviction=random_eviction,
                                           constants={'safe': np.array([0.1], dtype=np.float32)},
                                           lazy_init=lazy_init, name='D_neg')
        self.buffer = EfficientReplayBuffer(maxsize=2048, random_eviction=False, lazy_init=lazy_init)
        # self.buffer2 = EfficientReplayBuffer(maxsize=16384, random_eviction=False)

    def add_frame(self, obs, rews, terminated, truncated, info, **kwargs):
        if terminated:
            self.D_pos.absorb(self.buffer)
            # self.buffer2.absorb(self.buffer)
        elif truncated:
            self.D_neg.absorb(self.buffer)
            # self.D_neg.absorb(self.buffer2)
        self.buffer.append(batched=False,
                           rewards=rews,
                           **obs, **kwargs)

    def clear_buffer(self):
        self.buffer.clear()
        # self.buffer2.clear()

    def __len__(self):
        return len(self.D_pos) + len(self.D_neg)

    def __getitem__(self, idx):
        if idx < len(self.D_pos):
            return self.D_pos[idx]
        return self.D_neg[idx - len(self.D_pos)]

    def popback(self, n) -> None:
        return self.buffer.popback(n)

    def pop(self, idx=None) -> None:
        raise NotImplementedError

    def export(self, path=None, name=None):
        self.D_pos.export(path=path, name=f"{name}_pos")
        self.D_neg.export(path=path, name=f"{name}_neg")

    def load(self, path=None, name=None):
        self.D_pos.load(path=path, name=f"{name}_pos")
        self.D_neg.load(path=path, name=f"{name}_neg")

    def preprocess(self, rho=1.0): # original value of rho = 1.0
        self.buffer.clear()
        if not self.D_neg.initialized or len(self.D_neg) < 16:
            return
        mask = self.D_pos.is_in_knn_convex_hull(self.D_neg.consolidate()['state'], ['state'], k=100, threshold=rho)
        projected_safe = self.D_neg.pop(np.where(mask)[0])
        # self.D_pos.append(batched=True, size=np.sum(mask), **projected_safe))
        logger.debug(f"Ditched {np.sum(mask)} examples from D_neg. {len(self.D_neg)} left uncertain. {len(self.D_pos)} known safe.")
        return projected_safe

class EfficientContrastReplayBuffer(EfficientReplayBuffer):

    """Replay buffer specifically used for contrastive learning"""

    def __init__(self,
                 maxsize=1_000_000, transform=None, random_eviction: bool = True, constants: dict = None,
                 lazy_init=True, name=None, debug = False, negative_num = 3):
        """A more memory efficient implementation of the replay buffer with numpy arrays.
        Overwrite _fetch to determine how the dataset interacts with dataloaders.
        """
        super().__init__()
        self.maxsize = maxsize
        self.fields = {}
        self.constants = constants if constants is not None else {}
        self.transform = transform if transform is not None else {}
        self.random_eviction = random_eviction
        self.lazy_init = lazy_init
        self._replay_buffer = name

        self.left = 0
        self.right = 0
        self.size = 0

        self.initialized = False
        self.debug = debug
        self.negative_num = negative_num

        if self.debug:
            self.fig, self.axes = plt.subplots(1, 3, figsize=(12, 4))
            self.axes[0].set_title("Camera")
            self.axes[1].set_title("Positive")
            self.axes[2].set_title("Negatives")

            # Initialize with blank images
            self.display_camera = self.axes[0].imshow(np.zeros((224, 224, 3), dtype=np.uint8))
            self.display_positive = self.axes[1].imshow(np.zeros((224, 224, 3), dtype=np.uint8))
            self.display_negatives = self.axes[2].imshow(np.zeros((224, 224 * self.negative_num, 3), dtype=np.uint8))

            plt.ion()
            plt.show()
    
    def update_display(self, data):
        """Update the debug visualization with camera, positive, and negatives."""
        if not self.debug:
            return

        self.display_camera.set_data(data["camera"])
        self.display_positive.set_data(data["positive"])

        negatives = data["negatives"]  # shape [B, 224, 224, 3]
        if negatives.shape[0] == 0:
            tiled = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            tiled = np.concatenate(negatives, axis=1)  # stack along width → [224, 224 * B, 3]

        self.display_negatives.set_data(tiled)
        time.sleep(5.0)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _fetch(self, index):
        """Note: index is the absolute index in the arrays. """

        data = {}
        for k in self.fields.keys():
            data[k] = self.fields[k][index]
        
        # Add a positive key that is going to be augmented
        data["positive"] = np.copy(data['camera'])

        # sample negative examples sequences
        sample_negative = self.action_sample
        sample_negative(data, index)

        for k in data.keys():
            if k in self.transform:
                data[k] = self.transform[k](data)
        if self.debug:
            self.update_display(data)
        return {**data, **self.constants}
    
    """generate negative examples: the images taken at different positions along the track
        Such examples will be stored as a list of images under the key value of 'negatives'
        The negative examples will be sampled as far from the gps position of the positive example as possible"""
    
    def naive_sample(self, data, index):
        """sample negative examples only based on the gps positions corresponding to the query image and the negative images"""
        res = []
        min_distance = 5.0 # the euclidean distance between the gps position of the positve example, and the negative examples
        gps_list = self.fields["gps"]
        cur_gps = self.fields["gps"][index]

        for _ in range(self.negative_num):
            find = False
            while not find:

                random_idx = np.random.randint(low = self.left, high = self.right)
                random_gps = gps_list[random_idx]

                # to be optimized: the distance mechanism should be adjusted to better reflect the contrast between positive and negative
                if np.linalg.norm(cur_gps - random_gps) >= min_distance:
                    find = True
                    res.append(np.copy(self.fields["camera"][random_idx]))

        data["negatives"] = np.stack(res, axis = 0)
    
    def action_sample(self, data, index):

        """sample logic as follow:
        Look for the examples that have velocities within v_distance, and actions out of the range of action_distance
        i.e. velocities as similar as possible but actions as different as possible
        """
        res = []
        v_distance = 0.5
        action_distance = 2.0
        v_list = self.fields["velocity"]
        cur_v = self.fields["velocity"][index]
        action_list = self.fields["action"]
        cur_action = self.fields["action"]

        for _ in range(self.negative_num):
            find = False
            while not find:

                random_idx = np.random.randint(low = self.left, high = self.right)
                random_v = v_list[random_idx]
                random_action = action_list[random_idx]

                # to be optimized: the distance mechanism should be adjusted to better reflect the contrast between positive and negative
                if np.linalg.norm(cur_v - random_v) <= v_distance and np.linalg.norm(cur_action - random_action) >= action_distance:
                    find = True
                    res.append(np.copy(self.fields["camera"][random_idx]))

        data["negatives"] = np.stack(res, axis = 0)

class EfficientContrastReplayBufferPN(EfficientReplayBufferPN):
    def __init__(self, maxsize: int = 1_000_000, transform=None, random_eviction: bool = True,
                 lazy_init=True):
        self.D_pos = EfficientContrastReplayBuffer(maxsize=maxsize, transform=transform, random_eviction=random_eviction,
                                           constants={'safe': np.array([1.], dtype=np.float32)},
                                           lazy_init=lazy_init, name='D_pos')
        self.D_neg = EfficientContrastReplayBuffer(maxsize=maxsize, transform=transform, random_eviction=random_eviction,
                                           constants={'safe': np.array([0.1], dtype=np.float32)},
                                           lazy_init=lazy_init, name='D_neg')
        self.buffer = EfficientReplayBuffer(maxsize=2048, random_eviction=False, lazy_init=lazy_init)

class sourceTargetBalanceBuffer():
    
    def __init__(self,
                 maxsize=1_000_000, transform=None, random_eviction: bool = True, constants: dict = None,
                 lazy_init=True, name=None, source_buffer = None):
        self.source_buffer : EfficientReplayBuffer = None
        self.target_buffer :EfficientReplayBuffer = None

        self.target_buffer = EfficientReplayBuffer(maxsize=maxsize // 2,
                                        lazy_init=lazy_init,
                                        transform = transform,
                                        random_eviction = random_eviction,
                                        constants = constants,
                                        name = name)
        if source_buffer is None:
            self.source_buffer = EfficientReplayBuffer(maxsize=maxsize,
                                        lazy_init=lazy_init,
                                        transform = transform,
                                        random_eviction = random_eviction,
                                        constants = constants,
                                        name = name)
        else:
            self.source_buffer = source_buffer # if the source_buffer is given, directly set it as a class attribute

    def balanced_dataloader(self, batch_size: int = 64, shuffle: bool = True, num_workers: int = 0, manifest=None) -> DataLoader:

        """The resulted dataloader contains the same amount of data from
        the source domain and the target domain"""

        source_loader = self.source_buffer.dataloader(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                                manifest=manifest)
        target_loader = self.target_buffer.dataloader(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                                manifest=manifest)
        
        return StackedLoader(source_loader, target_loader)
    
    def series_dataloader(self, batch_size: int = 64, shuffle: bool = True, num_workers: int = 0, manifest=None) -> DataLoader:

        """The resulted dataloader is not necesarily balanced.
        It contains all the data from the source domain loader and the target domain loader"""

        if manifest is None:
            raise ValueError("Must specify fields to fetch.")

        def collate_fn(batch):
            """
            Custom collate function to extract only the requested features and labels.
            """
            ret = []
            for blocks in manifest:
                ret.append([ptu.from_numpy(np.stack([item[field] for item in batch], axis=0)) for field in blocks])
            return ret
        
        series_dataset = ConcatDataset([self.source_buffer, self.target_buffer])
        series_loader = DataLoader(
            series_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,
            num_workers = num_workers,
            collate_fn=collate_fn  # assumes both loaders use the same collate_fn
        )

        return series_loader
    
    def dataloader(self, batch_size: int = 64, shuffle: bool = True, num_workers: int = 0, manifest=None):
        """the default data loaderis the series loader"""
        return self.series_dataloader(batch_size = batch_size, shuffle = shuffle, num_workers = num_workers, manifest = manifest)
    
class EfficientReplayBufferPN_nopreprocess(EfficientReplayBufferPN):
    def preprocess(self):
        return


class EfficientReplayBufferSA(EfficientReplayBuffer):
    def dataloader(self, batch_size: int = 64, shuffle: bool = True, num_workers: int = 0, manifest=None) -> DataLoader:
        return super().dataloader(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                  manifest=[['states'], ['actions']])


class EfficientReplayBufferSCA(EfficientReplayBuffer):
    def dataloader(self, batch_size: int = 64, shuffle: bool = True, num_workers: int = 0, manifest=None) -> DataLoader:
        return super().dataloader(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                  manifest=[['states', 'conds'], ['actions']])
