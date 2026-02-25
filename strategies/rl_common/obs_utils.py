"""
Observation encoding utilities for RL strategies.

Converts raw BinState / Box data into normalised tensors suitable for
neural network input.  All functions return numpy arrays.

Encoding schemes:
  1. Heightmap: 2D grid normalised by bin height → CNN input
  2. Box features: (l, w, h, vol, weight) normalised → MLP input
  3. Buffer features: stack of box features for all visible boxes
  4. Multi-channel heightmap: height + support + item footprint

These utilities are shared across all RL strategies to ensure
consistent observation encoding.

References:
  - Zhao et al. (AAAI 2021): 4-channel (height, item_w, item_d, item_h)
  - Tsang et al. (2025): 2-channel (bin_height, action_map)
  - Xiong et al. (RA-L 2024): heightmap + EMS features
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from config import Box, BinConfig
    from simulator.bin_state import BinState


def encode_heightmap(
    bin_state,       # BinState
    bin_config,      # BinConfig
    normalise: bool = True,
) -> np.ndarray:
    """
    Encode a bin's heightmap as a normalised 2D array.

    Args:
        bin_state:  Current bin state.
        bin_config: Bin dimensions (for normalisation).
        normalise:  If True, divide by bin_config.height.

    Returns:
        np.ndarray of shape (grid_l, grid_w), dtype float32.
    """
    hm = bin_state.heightmap.astype(np.float32)
    if normalise and bin_config.height > 0:
        hm = hm / bin_config.height
    return hm


def encode_multi_channel_heightmap(
    bin_state,       # BinState
    bin_config,      # BinConfig
    box=None,        # Optional[Box] — current box for item channels
) -> np.ndarray:
    """
    Encode a multi-channel heightmap (Zhao et al. AAAI 2021 style).

    Channels:
      0: normalised height map
      1: item length / bin length (broadcast)
      2: item width / bin width (broadcast)
      3: item height / bin height (broadcast)

    If no box is provided, item channels are zero.

    Args:
        bin_state:  Current bin state.
        bin_config: Bin dimensions.
        box:        Optional current box.

    Returns:
        np.ndarray of shape (4, grid_l, grid_w), dtype float32.
    """
    gl, gw = bin_config.grid_l, bin_config.grid_w
    channels = np.zeros((4, gl, gw), dtype=np.float32)

    # Channel 0: heightmap
    channels[0] = bin_state.heightmap.astype(np.float32) / max(bin_config.height, 1.0)

    # Channels 1-3: item dimensions (broadcast)
    if box is not None:
        channels[1] = box.length / max(bin_config.length, 1.0)
        channels[2] = box.width / max(bin_config.width, 1.0)
        channels[3] = box.height / max(bin_config.height, 1.0)

    return channels


def encode_support_map(
    bin_state,       # BinState
    bin_config,      # BinConfig
    box=None,        # Optional[Box]
    orient_idx: int = 0,
) -> np.ndarray:
    """
    Encode a support ratio map for the current box at each grid position.

    For each (x, y) grid cell, compute the support ratio if the box
    were placed there.  Expensive — use sparingly or cache.

    Args:
        bin_state:  Current bin state.
        bin_config: Bin dimensions.
        box:        Box to compute support for.
        orient_idx: Orientation index.

    Returns:
        np.ndarray of shape (grid_l, grid_w), dtype float32.
        Values in [0, 1] where 1 = fully supported.
    """
    from config import Orientation

    gl, gw = bin_config.grid_l, bin_config.grid_w
    support_map = np.zeros((gl, gw), dtype=np.float32)

    if box is None:
        return support_map

    orients = Orientation.get_flat(box.length, box.width, box.height)
    if orient_idx >= len(orients):
        orient_idx = 0
    ol, ow, oh = orients[orient_idx]

    res = bin_config.resolution
    ol_g = int(round(ol / res))
    ow_g = int(round(ow / res))

    for gx in range(gl - ol_g + 1):
        for gy in range(gw - ow_g + 1):
            x = gx * res
            y = gy * res
            z = bin_state.get_height_at(x, y, ol, ow)
            if z + oh <= bin_config.height:
                support = bin_state.get_support_ratio(x, y, ol, ow, z)
                support_map[gx, gy] = support

    return support_map


def encode_box_features(
    box,             # Box
    bin_config,      # BinConfig
) -> np.ndarray:
    """
    Encode a single box as a normalised feature vector.

    Features: [length, width, height, volume, weight] — all normalised.

    Args:
        box:        Box to encode.
        bin_config: Bin dimensions (for normalisation).

    Returns:
        np.ndarray of shape (5,), dtype float32.
    """
    max_dim = max(bin_config.length, bin_config.width, bin_config.height)
    return np.array([
        box.length / max_dim,
        box.width / max_dim,
        box.height / max_dim,
        box.volume / max(bin_config.volume, 1.0),
        box.weight / 50.0,  # normalise weight (typical max ~50kg)
    ], dtype=np.float32)


def encode_buffer_features(
    boxes: list,     # List[Box]
    bin_config,      # BinConfig
    max_boxes: int = 8,
) -> np.ndarray:
    """
    Encode a list of boxes (buffer/grippable) as a feature matrix.

    Pads with zeros if fewer boxes than max_boxes.

    Args:
        boxes:      List of Box objects.
        bin_config: Bin dimensions (for normalisation).
        max_boxes:  Maximum number of boxes to encode.

    Returns:
        np.ndarray of shape (max_boxes, 5), dtype float32.
    """
    features = np.zeros((max_boxes, 5), dtype=np.float32)
    for i, box in enumerate(boxes[:max_boxes]):
        features[i] = encode_box_features(box, bin_config)
    return features


def encode_flat_observation(
    bin_states: list,     # List[BinState]
    grippable: list,      # List[Box]
    buffer_view: list,    # List[Box]
    bin_config,           # BinConfig
    pick_window: int = 4,
    buffer_size: int = 8,
) -> np.ndarray:
    """
    Encode the full observation as a single flat vector.

    Useful for simple MLP-based policies.  Concatenates:
      - All heightmaps (flattened)
      - Grippable box features
      - Buffer box features
      - Bin statistics

    Args:
        bin_states:  List of BinState objects.
        grippable:   Grippable boxes.
        buffer_view: All visible boxes.
        bin_config:  Bin dimensions.
        pick_window: Number of grippable boxes.
        buffer_size: Total buffer size.

    Returns:
        1D np.ndarray, dtype float32.
    """
    parts = []

    # Heightmaps (flattened, normalised)
    for bs in bin_states:
        hm = encode_heightmap(bs, bin_config, normalise=True)
        parts.append(hm.flatten())

    # Grippable box features
    grip_feats = encode_buffer_features(grippable, bin_config, max_boxes=pick_window)
    parts.append(grip_feats.flatten())

    # Buffer features
    buf_feats = encode_buffer_features(buffer_view, bin_config, max_boxes=buffer_size)
    parts.append(buf_feats.flatten())

    # Bin stats
    for bs in bin_states:
        stats = np.array([
            bs.get_fill_rate(),
            bs.get_max_height() / max(bin_config.height, 1.0),
            min(bs.get_surface_roughness() / 100.0, 1.0),
            len(bs.placed_boxes) / 50.0,
        ], dtype=np.float32)
        parts.append(stats)

    return np.concatenate(parts)
