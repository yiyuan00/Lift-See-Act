#!/usr/bin/env python3
"""Prepared DexYCB dataset and memory-mapped LSA teacher features.

This module intentionally does not import or instantiate VGGT/DepthPro.  It
reproduces the sample discovery and supervision used by dataset.py while only
loading the current RGB/depth/hand-prior frame (future image-like modalities
were loaded but discarded by the original dataset).
"""

from __future__ import annotations

import bisect
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms


DEFAULT_ROOTS = {
    "image": "data/dexycb/image",
    "keypoint": "data/dexycb/hand_keyPoint",
    "keypoint2d": "data/dexycb/hand_keyPoint2D",
    "depth": "data/dexycb/dp_image",
    "kpmap": "data/dexycb/kpmap",
}


def roots_from_data_root(data_root: str) -> Dict[str, str]:
    """Map a prepared DexYCB root to the five modalities used by LSA."""
    root = Path(data_root)
    return {
        "image": str(root / "image"),
        "keypoint": str(root / "hand_keyPoint"),
        "keypoint2d": str(root / "hand_keyPoint2D"),
        "depth": str(root / "dp_image"),
        "kpmap": str(root / "kpmap"),
    }


def _natural_key(value: str) -> List[object]:
    import re

    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", value)]


def _natural_sorted(values: Sequence[str]) -> List[str]:
    return sorted(values, key=_natural_key)


def stable_sample_key(char_name: str, seq_name: str, frame_name: str) -> str:
    return f"{char_name}/{seq_name}/{frame_name}"


class LSASequenceIndex:
    """Discover exactly the valid sequences and sliding windows used by training."""

    def __init__(
        self,
        image_root: str = DEFAULT_ROOTS["image"],
        keypoint_root: str = DEFAULT_ROOTS["keypoint"],
        keypoint2d_root: str = DEFAULT_ROOTS["keypoint2d"],
        depth_root: str = DEFAULT_ROOTS["depth"],
        kpmap_root: str = DEFAULT_ROOTS["kpmap"],
        context_length: int = 1,
        pred_length: int = 3,
    ) -> None:
        self.image_root = image_root
        self.keypoint_root = keypoint_root
        self.keypoint2d_root = keypoint2d_root
        self.depth_root = depth_root
        self.kpmap_root = kpmap_root
        self.context_length = context_length
        self.pred_length = pred_length
        self.min_seq_length = context_length + pred_length
        self.sequences: List[Dict[str, object]] = []
        self.rejected: Dict[str, int] = {}
        self._discover()
        self.offsets: List[int] = [0]
        for seq in self.sequences:
            count = int(seq["num_frames"]) - self.min_seq_length + 1
            self.offsets.append(self.offsets[-1] + count)

    def _reject(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1

    def _discover(self) -> None:
        # Keep os.listdir order to match the original dataset's global indices.
        for char_name in os.listdir(self.image_root):
            char_path = os.path.join(self.image_root, char_name)
            if not os.path.isdir(char_path):
                continue
            for seq_name in os.listdir(char_path):
                image_dir = os.path.join(char_path, seq_name)
                if not os.path.isdir(image_dir):
                    continue
                frame_names = [
                    name for name in os.listdir(image_dir)
                    if name.startswith("color_")
                    and name.lower().endswith((".png", ".jpg"))
                ]
                frame_names = _natural_sorted(frame_names)
                kp_path = os.path.join(
                    self.keypoint_root, char_name, seq_name, "3DP.npz"
                )
                if not os.path.exists(kp_path):
                    self._reject("missing_3DP")
                    continue
                try:
                    with np.load(kp_path) as payload:
                        num_frames = len(payload["points"])
                except Exception:
                    self._reject("bad_3DP")
                    continue
                if len(frame_names) != num_frames:
                    self._reject("rgb_3DP_count_mismatch")
                    continue
                if num_frames < self.min_seq_length:
                    self._reject("too_short")
                    continue

                keypoint2d_path: Optional[str] = None
                keypoint2d_dir = os.path.join(
                    self.keypoint2d_root, char_name, seq_name
                )
                if os.path.isdir(keypoint2d_dir):
                    for root, _, files in os.walk(keypoint2d_dir):
                        if "2DP.npz" in files:
                            keypoint2d_path = os.path.join(root, "2DP.npz")
                            break
                if keypoint2d_path is None:
                    self._reject("missing_2DP")
                    continue

                depth_dir = os.path.join(self.depth_root, char_name, seq_name)
                if not os.path.isdir(depth_dir):
                    self._reject("missing_depth_dir")
                    continue
                depth_names = _natural_sorted([
                    name for name in os.listdir(depth_dir)
                    if name.startswith("depth_")
                    and name.lower().endswith((".png", ".jpg"))
                ])
                if len(depth_names) != len(frame_names):
                    self._reject("depth_count_mismatch")
                    continue

                kpmap_dir = os.path.join(self.kpmap_root, char_name, seq_name)
                if not os.path.isdir(kpmap_dir):
                    self._reject("missing_kpmap_dir")
                    continue
                kpmap_names = _natural_sorted([
                    name for name in os.listdir(kpmap_dir)
                    if name.startswith("kpmap_")
                    and name.lower().endswith((".png", ".jpg"))
                ])
                if len(kpmap_names) != len(frame_names):
                    self._reject("kpmap_count_mismatch")
                    continue

                self.sequences.append({
                    "char_name": char_name,
                    "seq_name": seq_name,
                    "image_dir": image_dir,
                    "keypoint_path": kp_path,
                    "keypoint2d_path": keypoint2d_path,
                    "depth_dir": depth_dir,
                    "kpmap_dir": kpmap_dir,
                    "num_frames": num_frames,
                    "frame_names": frame_names,
                    "depth_names": depth_names,
                    "kpmap_names": kpmap_names,
                })

    def __len__(self) -> int:
        return self.offsets[-1]

    def locate(self, index: int) -> Tuple[Dict[str, object], int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        seq_index = bisect.bisect_right(self.offsets, index) - 1
        start_index = index - self.offsets[seq_index]
        return self.sequences[seq_index], start_index

    def sample_key(self, index: int) -> str:
        seq, start_index = self.locate(index)
        frame_name = seq["frame_names"][start_index]
        return stable_sample_key(
            str(seq["char_name"]), str(seq["seq_name"]), str(frame_name)
        )

    def sample_keys(self) -> List[str]:
        return [self.sample_key(index) for index in range(len(self))]

    def index_sha256(self) -> str:
        digest = hashlib.sha256()
        for key in self.sample_keys():
            digest.update(key.encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()

    def compute_keypoint_stats(
        self, seed: int = 42, max_sequences: int = 20
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        rng = random.Random(seed)
        selected = rng.sample(
            range(len(self.sequences)),
            min(max_sequences, len(self.sequences)),
        )
        total = None
        squared = None
        count = 0
        selected_keys: List[str] = []
        for seq_index in selected:
            seq = self.sequences[seq_index]
            with np.load(str(seq["keypoint_path"])) as payload:
                points = np.asarray(payload["points"], dtype=np.float64)
            sample_sum = points.sum(axis=0)
            sample_squared = np.square(points).sum(axis=0)
            total = sample_sum if total is None else total + sample_sum
            squared = (
                sample_squared if squared is None else squared + sample_squared
            )
            count += points.shape[0]
            selected_keys.append(
                f"{seq['char_name']}/{seq['seq_name']}"
            )
        if total is None or squared is None or count == 0:
            raise RuntimeError("No keypoints available for normalization")
        mean = total / count
        std = np.sqrt(squared / count - np.square(mean) + 1e-8)
        return mean.astype(np.float32), std.astype(np.float32), selected_keys


class CachedLSADataset(Dataset):
    """Training dataset backed by memory-mapped teacher representations."""

    def __init__(
        self,
        cache_root: str,
        roots: Optional[Dict[str, str]] = None,
        img_size: Tuple[int, int] = (224, 224),
    ) -> None:
        roots = {**DEFAULT_ROOTS, **(roots or {})}
        self.index = LSASequenceIndex(
            image_root=roots["image"],
            keypoint_root=roots["keypoint"],
            keypoint2d_root=roots["keypoint2d"],
            depth_root=roots["depth"],
            kpmap_root=roots["kpmap"],
            context_length=1,
            pred_length=3,
        )
        self.cache_root = Path(cache_root)
        manifest_path = self.cache_root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing cache manifest: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest["sample_count"] != len(self.index):
            raise RuntimeError(
                f"Cache sample count {self.manifest['sample_count']} != "
                f"dataset sample count {len(self.index)}"
            )
        actual_hash = self.index.index_sha256()
        if self.manifest["index_sha256"] != actual_hash:
            raise RuntimeError(
                "Dataset/cache index mismatch. Refusing to use misaligned features."
            )

        self.vggt = np.load(
            self.cache_root / self.manifest["arrays"]["vggt"]["file"],
            mmap_mode="r",
        )
        self.dpro = np.load(
            self.cache_root / self.manifest["arrays"]["dpro"]["file"],
            mmap_mode="r",
        )
        expected_vggt = (len(self.index), 196, 2048)
        expected_dpro = (len(self.index), 196, 1024)
        if self.vggt.shape != expected_vggt:
            raise RuntimeError(f"Unexpected VGGT cache shape: {self.vggt.shape}")
        if self.dpro.shape != expected_dpro:
            raise RuntimeError(f"Unexpected DepthPro cache shape: {self.dpro.shape}")

        stats = np.load(self.cache_root / "keypoint_stats.npz")
        self.kp_mean = np.asarray(stats["mean"], dtype=np.float32)
        self.kp_std = np.asarray(stats["std"], dtype=np.float32)
        self.img_size = img_size
        self.rgb_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        self.map_transform = transforms.Compose([
            transforms.Resize(
                img_size, interpolation=Image.Resampling.BICUBIC
            ),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> Dict[str, object]:
        seq, start = self.index.locate(item)
        frame_path = os.path.join(
            str(seq["image_dir"]), str(seq["frame_names"][start])
        )
        depth_path = os.path.join(
            str(seq["depth_dir"]), str(seq["depth_names"][start])
        )
        kpmap_path = os.path.join(
            str(seq["kpmap_dir"]), str(seq["kpmap_names"][start])
        )
        with Image.open(frame_path) as image:
            rgb = self.rgb_transform(image.convert("RGB"))
        with Image.open(depth_path) as image:
            depth = self.map_transform(image.convert("L"))
        with Image.open(kpmap_path) as image:
            kpmap = self.map_transform(image.convert("L"))

        with np.load(str(seq["keypoint_path"])) as payload:
            keypoints = np.asarray(
                payload["points"][start:start + 4], dtype=np.float32
            ).copy()
        with np.load(str(seq["keypoint2d_path"])) as payload:
            keypoints2d = np.asarray(
                payload["points"][start:start + 4], dtype=np.float32
            ).copy()
        keypoints = (keypoints - self.kp_mean) / self.kp_std
        keypoints2d[..., 0] /= 640.0
        keypoints2d[..., 1] /= 480.0

        return {
            "global_index": item,
            "sample_key": self.index.sample_key(item),
            "images": rgb.unsqueeze(0),
            "depths": depth.unsqueeze(0),
            "kpmaps": kpmap.unsqueeze(0),
            "keypoints": torch.from_numpy(keypoints[:1]),
            "keypoints_2d": torch.from_numpy(keypoints2d[:1]),
            "targets": torch.from_numpy(keypoints[1:]),
            "targets_2d": torch.from_numpy(keypoints2d[1:]),
            "vggt_target": np.asarray(self.vggt[item]),
            "dpro_target": np.asarray(self.dpro[item]),
        }


def cached_collate(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    return {
        "global_index": torch.tensor(
            [int(item["global_index"]) for item in batch], dtype=torch.long
        ),
        "sample_key": [str(item["sample_key"]) for item in batch],
        "images": torch.stack([item["images"] for item in batch]),
        "depths": torch.stack([item["depths"] for item in batch]),
        "kpmaps": torch.stack([item["kpmaps"] for item in batch]),
        "keypoints": torch.stack([item["keypoints"] for item in batch]),
        "keypoints_2d": torch.stack(
            [item["keypoints_2d"] for item in batch]
        ),
        "targets": torch.stack([item["targets"] for item in batch]),
        "targets_2d": torch.stack([item["targets_2d"] for item in batch]),
        "vggt_target": torch.from_numpy(np.stack([
            item["vggt_target"] for item in batch
        ])),
        "dpro_target": torch.from_numpy(np.stack([
            item["dpro_target"] for item in batch
        ])),
    }


class FixedGlobalBatchSampler(Sampler[int]):
    """Give each rank its slice of the original global batches.

    For global batch 64 and world size 4, rank r receives positions
    [16*r:16*(r+1)] from every original batch. This preserves the original
    batch grouping and exact optimizer-step count.
    """

    def __init__(
        self,
        ordered_indices: Sequence[int],
        rank: int,
        world_size: int,
        global_batch_size: int = 16,
        local_batch_size: int = 4,
    ) -> None:
        if global_batch_size != world_size * local_batch_size:
            raise ValueError("global batch must equal world_size * local batch")
        usable = len(ordered_indices) // global_batch_size * global_batch_size
        batches = np.asarray(
            ordered_indices[:usable], dtype=np.int64
        ).reshape(-1, global_batch_size)
        begin = rank * local_batch_size
        end = begin + local_batch_size
        self.indices = batches[:, begin:end].reshape(-1).tolist()
        self.global_steps = batches.shape[0]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def original_split_indices(
    length: int, seed: int = 42, train_fraction: float = 0.8
) -> Tuple[List[int], List[int]]:
    train_size = int(train_fraction * length)
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(length, generator=generator).tolist()
    return permutation[:train_size], permutation[train_size:]
