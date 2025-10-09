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
from utils.data_util import EfficientReplayBuffer
import os
from loguru import logger
import matplotlib.pyplot as plt
import time
import torch
from mpclab_common.track import get_track
import random

def test_barc_data_load():
    "test the barc_data_loading function"
    data_path = "~/Documents/data"
    file_name = "ParaDriveLocalComparison_Sep7_2"
    target_buffer = EfficientReplayBuffer(maxsize= 5000, lazy_init=True)
    target_buffer.load_barc_data(data_path = data_path, data_name = file_name)

    # verify the visualization of the target_buffer's gps information
    # fig, ax = plt.subplots(figsize=(8, 6))
    # track_obj = get_track(track_file = "L_track_barc")
    # track_obj.plot_map(ax)
    # gps = np.asarray([target_buffer[i]['gps'] for i in range(target_buffer.size)])
    # plot_global_points(coord = gps[:, : 2], ax = ax)

    # visualize the RGB images of the target buffer
    random_visualize_rgbs(target_buffer)

def compare_RGBs(num_pairs: int = 8, pairs_per_row: int = 4, seed: int = 20):
    """
    Randomly sample items from barc_buffer. For each, find the CARLA sample with
    the smallest global (x,y) distance. Display RGB pairs side-by-side.

    Parameters
    ----------
    num_pairs : int
        Number of BARC–CARLA pairs to visualize.
    pairs_per_row : int
        How many pairs per row in the figure (each pair uses 2 columns).
    seed : int | None
        Optional RNG seed for reproducibility.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # ----------------------------
    # Load buffers (given in prompt)
    # ----------------------------
    data_dir = Path(__file__).parent.parent / 'data'
    carla_buffer = EfficientReplayBuffer(maxsize=25000, lazy_init=True)
    carla_buffer.load(path=data_dir, name="L_track_barc_Hardware_params_model")

    data_path = os.path.expanduser("~/Documents/data")  # expand ~ to avoid FileNotFoundError
    file_name = "ParaDriveLocalComparison_Sep7_2"
    barc_buffer = EfficientReplayBuffer(maxsize=5000, lazy_init=True)
    barc_buffer.load_barc_data(data_path=data_path, data_name=file_name)

    if len(carla_buffer) == 0 or len(barc_buffer) == 0:
        raise ValueError("One of the buffers is empty. Cannot compare RGBs.")

    # ----------------------------
    # Pre-extract CARLA XY for fast NN lookup
    # ----------------------------
    def _safe_xy(buf, i):
        gps = buf[i].get('gps', None)
        if gps is None or len(gps) < 2:
            return None
        xy = np.asarray(gps[:2], dtype=float)
        if not np.all(np.isfinite(xy)):
            return None
        return xy

    def _safe_img(buf, i):
        rgb = buf[i].get('camera', None)
        if rgb is None:
            return None
        rgb = np.asarray(rgb)
        # Expect channel-last RGB; normalize floats
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 1)
        return rgb

    carla_xy = []
    carla_valid_idx = []
    for j in range(len(carla_buffer)):
        xy = _safe_xy(carla_buffer, j)
        if xy is not None:
            carla_xy.append(xy)
            carla_valid_idx.append(j)

    if len(carla_xy) == 0:
        raise ValueError("No valid (x,y) coordinates found in CARLA buffer.")

    carla_xy = np.vstack(carla_xy)  # shape (M, 2)

    # ----------------------------
    # Choose random BARC indices to compare
    # ----------------------------
    barc_indices_all = [i for i in range(len(barc_buffer)) if _safe_xy(barc_buffer, i) is not None]
    if len(barc_indices_all) == 0:
        raise ValueError("No valid (x,y) coordinates found in BARC buffer.")

    num_pairs = min(num_pairs, len(barc_indices_all))
    barc_indices = random.sample(barc_indices_all, num_pairs)

    # ----------------------------
    # For each BARC index, find nearest CARLA index by Euclidean distance
    # ----------------------------
    pairs = []
    for i in barc_indices:
        b_xy = _safe_xy(barc_buffer, i)           # shape (2,)
        diffs = carla_xy - b_xy                   # broadcast to (M, 2)
        d2 = np.einsum('ij,ij->i', diffs, diffs)  # squared distances (M,)
        nn_pos = int(np.argmin(d2))
        j = carla_valid_idx[nn_pos]               # original CARLA buffer index
        dist = float(np.sqrt(d2[nn_pos]))

        b_img = _safe_img(barc_buffer, i)
        c_img = _safe_img(carla_buffer, j)
        # Skip if either image missing
        if b_img is None or c_img is None:
            continue

        pairs.append((i, j, dist, b_img, c_img, b_xy, carla_xy[nn_pos]))

    if len(pairs) == 0:
        raise RuntimeError("Could not assemble any valid image pairs to show.")

    # ----------------------------
    # Make figure: 2 columns per pair (BARC | CARLA)
    # ----------------------------
    n = len(pairs)
    nrows = int(np.ceil(n / pairs_per_row))
    ncols = pairs_per_row * 2  # two images per pair

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.2 * nrows))
    axes = np.atleast_2d(axes)
    axes = axes.reshape(nrows, ncols)

    # Fill figure
    for k, (i_barc, i_carla, dist, img_b, img_c, xy_b, xy_c) in enumerate(pairs):
        r = k // pairs_per_row
        c = (k % pairs_per_row) * 2

        # BARC image
        ax_b = axes[r, c]
        ax_b.imshow(img_b)
        ax_b.set_title(f"BARC idx {i_barc}\n(xy={xy_b[0]:.2f},{xy_b[1]:.2f})", fontsize=10)
        ax_b.axis('off')

        # CARLA image
        ax_c = axes[r, c + 1]
        ax_c.imshow(img_c)
        ax_c.set_title(f"CARLA idx {i_carla}\nmatch dist={dist:.3f} m\n(xy={xy_c[0]:.2f},{xy_c[1]:.2f})", fontsize=10)
        ax_c.axis('off')

    # Hide any unused axes if grid bigger than pairs
    for k in range(n, nrows * pairs_per_row):
        r = k // pairs_per_row
        c = (k % pairs_per_row) * 2
        axes[r, c].axis('off')
        axes[r, c + 1].axis('off')

    plt.tight_layout()
    plt.show()
    return fig

def random_visualize_rgbs(target_buffer, num_samples=8, ncols=4, figsize=(12, 8)):
    """
    Randomly sample RGB images from target_buffer and visualize them in a grid.

    Parameters
    ----------
    target_buffer : EfficientReplayBuffer
        The buffer containing image entries. Each entry should allow access via
        target_buffer[i]['camera'] → np.ndarray(H, W, 3), dtype uint8 or float.
    num_samples : int, optional
        Total number of images to visualize (default 8).
    ncols : int, optional
        Number of columns in the subplot grid (default 4).
    figsize : tuple, optional
        Figure size in inches.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure handle for further customization or saving.
    """
    buffer_len = len(target_buffer)
    if buffer_len == 0:
        raise ValueError("The target_buffer is empty — nothing to visualize.")

    # Choose random valid indices
    num_samples = min(num_samples, buffer_len)
    sample_idxs = random.sample(range(buffer_len), num_samples)

    # Compute grid size
    nrows = int(np.ceil(num_samples / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1)  # flatten for uniform indexing

    for ax, idx in zip(axes, sample_idxs):
        rgb = target_buffer[idx]['camera']
        if rgb.dtype != np.uint8:
            # Normalize if necessary for display
            rgb = np.clip(rgb, 0, 1)

        ax.imshow(rgb)
        ax.set_title(f"Sample #{idx}", fontsize=10)
        ax.axis('off')

    # Turn off any unused axes
    for ax in axes[num_samples:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    return fig

def plot_global_points(ax, coord, *, color='red', s=20, alpha=0.8, label=None):
    """
    Scatter-plot a list of global (x, y) coordinates on a given Matplotlib Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes where the track or map is already plotted.
    coord : np.ndarray, shape (N, 2)
        Array of global coordinates (x, y) to visualize.
    color : str, optional
        Point color (default 'red').
    s : float, optional
        Marker size.
    alpha : float, optional
        Marker transparency (0–1).
    label : str, optional
        Legend label for the points.

    Returns
    -------
    scatter : matplotlib.collections.PathCollection
        The scatter artist created by Matplotlib.
    """
    coord = np.asarray(coord, dtype=float)
    if coord.ndim != 2 or coord.shape[1] != 2:
        raise ValueError(f"Expected shape (N, 2), got {coord.shape}")

    sc = ax.scatter(coord[:, 0], coord[:, 1],
                    color=color, s=s, alpha=alpha, label=label,
                    edgecolors='none')

    if label is not None:
        ax.legend(loc='best', frameon=False)

    ax.set_aspect('equal')
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    plt.show()
    return sc

def visualize_barc_states():
    """Verify which states are the global x, y coordinates"""
    data_path = "~/Documents/data"
    file_name = "ParaDriveLocalComparison_Sep7_9"
    # Expand ~ and normalize inputs
    data_dir = Path(data_path).expanduser()
    data_stem = Path(file_name).stem  # handles "foo" or "foo.npz"

    # If user accidentally passed a full file path in data_path, handle that too
    if data_dir.is_file() and data_dir.suffix == ".npz":
        file_path = data_dir
    else:
        file_path = (data_dir / f"{data_stem}.npz")
        
    if not file_path.exists():
        raise FileNotFoundError(f"Could not find npz at {file_path}")
    else:
        logger.info(f"Found barc_data in {file_path}")

    # Load lazily to avoid copying large arrays into RAM twice
    # (we’ll copy only what we need below).
    data = np.load(file_path, mmap_mode="r", allow_pickle=False)

    guess = data["states"][:, 0 : 2] # modify this line to find the most sensible entries for the x, y coordinates

    """so far the best one for global coordinates: data["states"][:, 3: 5].
    The most sensible data pairs for v_long, and v_tran: data['states'][0 : 2]"""

    fig, ax = plt.subplots(figsize=(8, 6))
    # track_obj = get_track(track_file = "L_track_barc")
    # track_obj.plot_map(ax) 
    plot_global_points(coord = guess, ax = ax)

if __name__ == "__main__":
    visualize_barc_states()