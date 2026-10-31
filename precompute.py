#!/usr/bin/env python3
"""Offline VGGT/Depth Pro feature extraction with resumable memmaps."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
import socket
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.transforms import functional as TF

from dataset import LSASequenceIndex, roots_from_data_root
from vggt.models.vggt import VGGT
import depth_pro
from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT


VGGT_DIM = 2048
DPRO_DIM = 1024
OUT_HW = 14


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--flush-every", type=int, default=25)
    parser.add_argument("--verify-samples", type=int, default=32)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--vggt-model", default="facebook/VGGT-1B"
    )
    parser.add_argument(
        "--depth-checkpoint",
        default="checkpoints/depth_pro.pt",
    )
    return parser.parse_args()


def atomic_json(path: Path, payload: Dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_vggt(source: str, device: torch.device) -> VGGT:
    """Load VGGT from a Hugging Face model ID, directory, or model.pt file."""
    checkpoint = Path(source)
    if checkpoint.is_file():
        model = VGGT()
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state)
        return model.to(device).eval()
    return VGGT.from_pretrained(source).to(device).eval()


def extract_vggt_features(model: VGGT, images: torch.Tensor) -> torch.Tensor:
    """Return the final spatial tokens without running VGGT task heads."""
    features, _ = model.aggregator(images)
    return features[-1]


def extract_dpro_features(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Return the final global ViT tokens from the unmodified Depth Pro model."""
    if images.shape[-2:] != (model.img_size, model.img_size):
        images = F.interpolate(
            images,
            size=(model.img_size, model.img_size),
            mode="bilinear",
            align_corners=False,
        )
    _, _, low_resolution = model.encoder._create_pyramid(images)
    return model.encoder.image_encoder(low_resolution)


def build_2d_sincos_pos_embed(
    height: int, width: int, dim: int, device: torch.device
) -> torch.Tensor:
    def one_dim(length: int, channels: int) -> torch.Tensor:
        position = torch.arange(
            length, device=device, dtype=torch.float32
        ).unsqueeze(1)
        index = torch.arange(
            0, channels, 2, device=device, dtype=torch.float32
        )
        divisor = torch.pow(10000, index / channels)
        return torch.cat([
            torch.sin(position / divisor),
            torch.cos(position / divisor),
        ], dim=1)

    half = dim // 2
    y = one_dim(height, half)[:, None, :]
    x = one_dim(width, half)[None, :, :]
    return torch.cat([
        y.expand(height, width, half),
        x.expand(height, width, half),
    ], dim=-1)


def vggt_transform(image: Image.Image) -> torch.Tensor:
    target_size = 518
    width, height = image.size
    new_width = target_size
    new_height = round(height * (new_width / width) / 14) * 14
    image = image.resize(
        (new_width, new_height), Image.Resampling.BICUBIC
    )
    tensor = TF.to_tensor(image)
    if new_height > target_size:
        start = (new_height - target_size) // 2
        tensor = tensor[:, start:start + target_size, :]
    return tensor


class TeacherFrameDataset(Dataset):
    def __init__(self, index: LSASequenceIndex) -> None:
        self.index = index
        self.dpro_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ])

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int):
        sequence, start = self.index.locate(item)
        path = os.path.join(
            str(sequence["image_dir"]),
            str(sequence["frame_names"][start]),
        )
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            return (
                item,
                self.dpro_transform(rgb),
                vggt_transform(rgb),
            )


def collate_frames(batch: Sequence[object]):
    indices, dpro_images, vggt_images = zip(*batch)
    shapes = {tuple(image.shape) for image in vggt_images}
    if len(shapes) != 1:
        raise RuntimeError(f"Variable VGGT image shapes in batch: {shapes}")
    return (
        torch.tensor(indices, dtype=torch.long),
        torch.stack(dpro_images),
        torch.stack(vggt_images),
    )


def align_vggt(
    output: torch.Tensor,
    position: torch.Tensor,
    image_hw: Sequence[int],
) -> torch.Tensor:
    batch = output.shape[0]
    tokens = output[:, :, 5:].squeeze(1)
    grid_h = int(image_hw[0]) // 14
    grid_w = int(image_hw[1]) // 14
    expected_tokens = grid_h * grid_w
    if tokens.shape[1] != expected_tokens:
        raise RuntimeError(
            "VGGT token/grid mismatch: "
            f"received {tokens.shape[1]} tokens for a "
            f"{grid_h}x{grid_w} grid"
        )
    pixels = (
        tokens.view(batch, grid_h, grid_w, VGGT_DIM)
        .permute(0, 3, 1, 2)
        .contiguous()
    )
    aligned = F.interpolate(
        pixels, size=(OUT_HW, OUT_HW),
        mode="bilinear", align_corners=False,
    )
    aligned = aligned.permute(0, 2, 3, 1).reshape(
        batch, OUT_HW * OUT_HW, VGGT_DIM
    )
    # The float32 positional embedding promotes the normalized target to FP32.
    return F.normalize(aligned + position.view(1, 196, VGGT_DIM), dim=-1)


def align_dpro(
    output: torch.Tensor, position: torch.Tensor
) -> torch.Tensor:
    batch = output.shape[0]
    tokens = output[:, 1:, :]
    pixels = (
        tokens.view(batch, 24, 24, DPRO_DIM)
        .permute(0, 3, 1, 2)
        .contiguous()
    )
    aligned = F.interpolate(
        pixels, size=(OUT_HW, OUT_HW),
        mode="bilinear", align_corners=False,
    )
    aligned = aligned.permute(0, 2, 3, 1).reshape(
        batch, OUT_HW * OUT_HW, DPRO_DIM
    )
    return F.normalize(
        aligned + position.view(1, 196, DPRO_DIM).to(aligned.dtype),
        dim=-1,
    )


def contiguous_rank_indices(length: int, rank: int, world: int) -> List[int]:
    begin = math.floor(length * rank / world)
    end = math.floor(length * (rank + 1) / world)
    return list(range(begin, end))


def tensor_similarity(reference: torch.Tensor, cached: torch.Tensor) -> Dict[str, float]:
    ref = reference.float().reshape(reference.shape[0], -1)
    got = cached.float().reshape(cached.shape[0], -1)
    cosine = F.cosine_similarity(ref, got, dim=-1)
    absolute = (ref - got).abs()
    return {
        "mean_cosine": cosine.mean().item(),
        "min_cosine": cosine.min().item(),
        "mean_absolute_error": absolute.mean().item(),
        "max_absolute_error": absolute.max().item(),
    }


def main() -> None:
    args = parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if world < 1:
        raise RuntimeError(f"Invalid world size: {world}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    cache_root = Path(args.cache_root)
    if rank == 0:
        cache_root.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    roots = roots_from_data_root(args.data_root)
    index = LSASequenceIndex(
        image_root=roots["image"],
        keypoint_root=roots["keypoint"],
        keypoint2d_root=roots["keypoint2d"],
        depth_root=roots["depth"],
        kpmap_root=roots["kpmap"],
    )
    if len(index) == 0:
        raise RuntimeError(f"No valid samples found under {args.data_root}")
    complete = False
    if rank == 0:
        manifest_path = cache_root / "manifest.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            vggt_path = cache_root / existing["arrays"]["vggt"]["file"]
            dpro_path = cache_root / existing["arrays"]["dpro"]["file"]
            complete = (
                existing.get("sample_count") == len(index)
                and existing.get("index_sha256") == index.index_sha256()
                and vggt_path.exists()
                and dpro_path.exists()
                and vggt_path.stat().st_size
                == existing["arrays"]["vggt"]["bytes"]
                and dpro_path.stat().st_size
                == existing["arrays"]["dpro"]["bytes"]
            )
            if not complete:
                raise RuntimeError(
                    "A manifest exists but the finalized cache is inconsistent"
                )
    complete_tensor = torch.tensor(
        1 if complete else 0, device=device, dtype=torch.uint8
    )
    dist.broadcast(complete_tensor, src=0)
    if complete_tensor.item() == 1:
        if rank == 0:
            print(
                f"Validated existing complete cache at {cache_root}; "
                "nothing to do.",
                flush=True,
            )
        dist.barrier()
        dist.destroy_process_group()
        return
    dataset = TeacherFrameDataset(index)
    local_indices = contiguous_rank_indices(len(index), rank, world)
    loader = DataLoader(
        Subset(dataset, local_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
        collate_fn=collate_frames,
    )

    model_load_start = time.perf_counter()
    vggt = load_vggt(args.vggt_model, device)
    dpro_config = replace(
        DEFAULT_MONODEPTH_CONFIG_DICT,
        checkpoint_uri=args.depth_checkpoint,
    )
    dpro, _ = depth_pro.create_model_and_transforms(
        config=dpro_config, device=device, precision=torch.float16
    )
    dpro.eval()
    model_load_seconds = time.perf_counter() - model_load_start
    vggt_pos = build_2d_sincos_pos_embed(
        OUT_HW, OUT_HW, VGGT_DIM, device
    )
    dpro_pos = build_2d_sincos_pos_embed(
        OUT_HW, OUT_HW, DPRO_DIM, device
    )

    # FP16 preflight on deterministic local samples.
    verify_count = min(args.verify_samples, len(local_indices))
    verify_indices = local_indices[:verify_count]
    verify_loader = DataLoader(
        Subset(dataset, verify_indices),
        batch_size=min(args.batch_size, max(1, verify_count)),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_frames,
    )
    local_verification = {
        "vggt": [], "dpro": [],
    }
    with torch.inference_mode():
        for _, dpro_images, vggt_images in verify_loader:
            dpro_images = dpro_images.to(device, non_blocking=True)
            vggt_images = vggt_images.to(device, non_blocking=True).unsqueeze(1)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                vggt_target = align_vggt(
                    extract_vggt_features(vggt, vggt_images),
                    vggt_pos,
                    vggt_images.shape[-2:],
                )
                dpro_target = align_dpro(
                    extract_dpro_features(dpro, dpro_images), dpro_pos
                )
            local_verification["vggt"].append(
                tensor_similarity(vggt_target, vggt_target.half())
            )
            local_verification["dpro"].append(
                tensor_similarity(dpro_target, dpro_target.half())
            )
    local_min_cos = min(
        metric["min_cosine"]
        for values in local_verification.values()
        for metric in values
    )
    local_max_error = max(
        metric["max_absolute_error"]
        for values in local_verification.values()
        for metric in values
    )
    preflight = torch.tensor(
        [local_min_cos, local_max_error], device=device, dtype=torch.float64
    )
    dist.all_reduce(preflight[0], op=dist.ReduceOp.MIN)
    dist.all_reduce(preflight[1], op=dist.ReduceOp.MAX)
    cache_dtype = np.float16 if preflight[0].item() >= 0.9999 else np.float32
    if args.preflight_only:
        result = {
            "rank": rank,
            "verified_samples": verify_count,
            "global_min_cosine": preflight[0].item(),
            "global_max_absolute_error": preflight[1].item(),
            "selected_dtype": np.dtype(cache_dtype).name,
            "gpu_name": torch.cuda.get_device_name(device),
            "model_load_seconds": model_load_seconds,
        }
        print(json.dumps(result, indent=2))
        dist.barrier()
        dist.destroy_process_group()
        return

    dtype_name = np.dtype(cache_dtype).name
    vggt_partial = cache_root / "vggt_targets.partial.npy"
    dpro_partial = cache_root / "dpro_targets.partial.npy"
    done_path = cache_root / "completed.partial.npy"
    if rank == 0:
        expected = {
            "vggt": ((len(index), 196, VGGT_DIM), cache_dtype),
            "dpro": ((len(index), 196, DPRO_DIM), cache_dtype),
        }
        if not vggt_partial.exists():
            np.lib.format.open_memmap(
                vggt_partial, mode="w+", dtype=cache_dtype,
                shape=expected["vggt"][0],
            ).flush()
        if not dpro_partial.exists():
            np.lib.format.open_memmap(
                dpro_partial, mode="w+", dtype=cache_dtype,
                shape=expected["dpro"][0],
            ).flush()
        if not done_path.exists():
            np.lib.format.open_memmap(
                done_path, mode="w+", dtype=np.uint8,
                shape=(len(index),),
            ).flush()
        mean, std, selected = index.compute_keypoint_stats(seed=args.seed)
        np.savez(
            cache_root / "keypoint_stats.npz",
            mean=mean, std=std,
            selected_sequences=np.asarray(selected),
            seed=np.asarray(args.seed),
        )
    dist.barrier()

    vggt_cache = np.load(vggt_partial, mmap_mode="r+")
    dpro_cache = np.load(dpro_partial, mmap_mode="r+")
    completed = np.load(done_path, mmap_mode="r+")
    if vggt_cache.dtype != cache_dtype or dpro_cache.dtype != cache_dtype:
        raise RuntimeError(
            "Existing partial cache dtype differs from preflight choice"
        )

    torch.cuda.reset_peak_memory_stats(device)
    start_wall = time.perf_counter()
    vggt_ms = 0.0
    dpro_ms = 0.0
    write_seconds = 0.0
    processed = 0
    skipped = 0
    failures: List[Dict[str, object]] = []

    for batch_number, (indices, dpro_images, vggt_images) in enumerate(loader):
        raw_indices = indices.numpy()
        missing_mask = np.asarray(completed[raw_indices] == 0)
        if not missing_mask.any():
            skipped += len(raw_indices)
            continue
        selected_positions = np.nonzero(missing_mask)[0]
        selected_indices = raw_indices[selected_positions]
        dpro_images = dpro_images[selected_positions].to(
            device, non_blocking=True
        )
        vggt_images = vggt_images[selected_positions].to(
            device, non_blocking=True
        ).unsqueeze(1)
        try:
            with torch.inference_mode():
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    vggt_target = align_vggt(
                        extract_vggt_features(vggt, vggt_images),
                        vggt_pos,
                        vggt_images.shape[-2:],
                    )
                end.record()
                end.synchronize()
                vggt_ms += begin.elapsed_time(end)

                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    dpro_target = align_dpro(
                        extract_dpro_features(dpro, dpro_images), dpro_pos
                    )
                end.record()
                end.synchronize()
                dpro_ms += begin.elapsed_time(end)

            write_start = time.perf_counter()
            vggt_cache[selected_indices] = (
                vggt_target.detach().cpu().numpy().astype(cache_dtype)
            )
            dpro_cache[selected_indices] = (
                dpro_target.detach().cpu().numpy().astype(cache_dtype)
            )
            completed[selected_indices] = 1
            write_seconds += time.perf_counter() - write_start
            processed += len(selected_indices)
            if (batch_number + 1) % args.flush_every == 0:
                vggt_cache.flush()
                dpro_cache.flush()
                completed.flush()
        except Exception as error:
            failures.append({
                "indices": selected_indices.tolist(),
                "error": repr(error),
            })
    vggt_cache.flush()
    dpro_cache.flush()
    completed.flush()
    torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - start_wall

    metrics = {
        "rank": rank,
        "local_rank": local_rank,
        "hostname": socket.gethostname(),
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_total_mib": torch.cuda.get_device_properties(device).total_memory
        / 1024 ** 2,
        "batch_size_per_gpu": args.batch_size,
        "model_load_seconds": model_load_seconds,
        "wall_seconds": wall_seconds,
        "processed": processed,
        "skipped_completed": skipped,
        "vggt_cuda_seconds": vggt_ms / 1000.0,
        "dpro_cuda_seconds": dpro_ms / 1000.0,
        "serialization_seconds": write_seconds,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device)
        / 1024 ** 3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device)
        / 1024 ** 3,
        "failures": failures,
        "preflight": {
            "global_min_cosine": preflight[0].item(),
            "global_max_absolute_error": preflight[1].item(),
            "selected_dtype": dtype_name,
        },
    }
    atomic_json(cache_root / f"precompute_rank{rank}.json", metrics)
    dist.barrier()

    if rank == 0:
        done = np.load(done_path, mmap_mode="r")
        completed_count = int(done.sum())
        if completed_count != len(index):
            raise RuntimeError(
                f"Incomplete cache: {completed_count}/{len(index)}"
            )
        final_vggt = cache_root / "vggt_targets.npy"
        final_dpro = cache_root / "dpro_targets.npy"
        os.replace(vggt_partial, final_vggt)
        os.replace(dpro_partial, final_dpro)
        os.replace(done_path, cache_root / "completed.npy")
        saved_vggt = np.load(final_vggt, mmap_mode="r")
        saved_dpro = np.load(final_dpro, mmap_mode="r")
        post_metrics: Dict[str, List[Dict[str, float]]] = {
            "vggt": [], "dpro": [],
        }
        loss_relative_errors: Dict[str, List[float]] = {
            "vggt": [], "dpro": [],
        }
        verification_generator = torch.Generator(device=device)
        verification_generator.manual_seed(args.seed)
        with torch.inference_mode():
            for indices, dpro_images, vggt_images in verify_loader:
                dpro_images = dpro_images.to(device, non_blocking=True)
                vggt_images = vggt_images.to(
                    device, non_blocking=True
                ).unsqueeze(1)
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    reference_vggt = align_vggt(
                        extract_vggt_features(vggt, vggt_images),
                        vggt_pos,
                        vggt_images.shape[-2:],
                    )
                    reference_dpro = align_dpro(
                        extract_dpro_features(dpro, dpro_images), dpro_pos
                    )
                cached_vggt = torch.from_numpy(
                    np.asarray(saved_vggt[indices.numpy()]).copy()
                ).to(device)
                cached_dpro = torch.from_numpy(
                    np.asarray(saved_dpro[indices.numpy()]).copy()
                ).to(device)
                for name, reference, cached in (
                    ("vggt", reference_vggt, cached_vggt),
                    ("dpro", reference_dpro, cached_dpro),
                ):
                    post_metrics[name].append(
                        tensor_similarity(reference, cached)
                    )
                    student = torch.randn(
                        reference.shape,
                        generator=verification_generator,
                        device=device,
                        dtype=torch.float32,
                    )
                    student = F.normalize(student, dim=-1)
                    online_loss = (
                        1.0
                        - (
                            student
                            * reference.to(student.dtype)
                        ).sum(dim=-1)
                    ).mean()
                    cached_loss = (
                        1.0
                        - (
                            student
                            * cached.to(student.dtype)
                        ).sum(dim=-1)
                    ).mean()
                    relative = (
                        (online_loss - cached_loss).abs()
                        / online_loss.abs().clamp_min(1e-8)
                    )
                    loss_relative_errors[name].append(relative.item())
        post_verification = {}
        for name in ("vggt", "dpro"):
            post_verification[name] = {
                "verified_samples": verify_count,
                "mean_cosine": float(np.mean([
                    item["mean_cosine"] for item in post_metrics[name]
                ])),
                "min_cosine": min(
                    item["min_cosine"] for item in post_metrics[name]
                ),
                "mean_absolute_error": float(np.mean([
                    item["mean_absolute_error"]
                    for item in post_metrics[name]
                ])),
                "max_absolute_error": max(
                    item["max_absolute_error"]
                    for item in post_metrics[name]
                ),
                "max_loss_relative_error": max(
                    loss_relative_errors[name]
                ),
            }
        if min(
            post_verification[name]["min_cosine"]
            for name in ("vggt", "dpro")
        ) < 0.9999:
            raise RuntimeError(
                f"Post-write cosine verification failed: {post_verification}"
            )
        if max(
            post_verification[name]["max_loss_relative_error"]
            for name in ("vggt", "dpro")
        ) > 0.001:
            raise RuntimeError(
                f"Post-write loss verification failed: {post_verification}"
            )
        rank_metrics = [
            json.loads(
                (cache_root / f"precompute_rank{i}.json").read_text()
            )
            for i in range(world)
        ]
        manifest = {
            "schema_version": 1,
            "created_unix_time": time.time(),
            "sample_count": len(index),
            "valid_sequences": len(index.sequences),
            "index_sha256": index.index_sha256(),
            "context_length": 1,
            "prediction_length": 3,
            "roots": roots,
            "arrays": {
                "vggt": {
                    "file": final_vggt.name,
                    "shape": [len(index), 196, VGGT_DIM],
                    "dtype": dtype_name,
                    "bytes": final_vggt.stat().st_size,
                },
                "dpro": {
                    "file": final_dpro.name,
                    "shape": [len(index), 196, DPRO_DIM],
                    "dtype": dtype_name,
                    "bytes": final_dpro.stat().st_size,
                },
            },
            "models": {
                "vggt": args.vggt_model,
                "depth_pro_checkpoint": args.depth_checkpoint,
                "depth_pro_checkpoint_sha256": sha256_file(
                    args.depth_checkpoint
                ),
            },
            "hardware": {
                "world_size": world,
                "gpus": [
                    {
                        "rank": item["rank"],
                        "name": item["gpu_name"],
                        "total_mib": item["gpu_total_mib"],
                    }
                    for item in rank_metrics
                ],
            },
            "preflight": rank_metrics[0]["preflight"],
            "post_write_verification": post_verification,
            "rank_metrics": rank_metrics,
        }
        atomic_json(cache_root / "manifest.json", manifest)
        summary = {
            "sample_count": len(index),
            "wall_seconds": max(item["wall_seconds"] for item in rank_metrics),
            "aggregate_samples_per_second": len(index) / max(
                item["wall_seconds"] for item in rank_metrics
            ),
            "gpu_hours": sum(
                item["wall_seconds"] for item in rank_metrics
            ) / 3600.0,
            "vggt_cuda_seconds_sum": sum(
                item["vggt_cuda_seconds"] for item in rank_metrics
            ),
            "dpro_cuda_seconds_sum": sum(
                item["dpro_cuda_seconds"] for item in rank_metrics
            ),
            "max_peak_allocated_gib": max(
                item["peak_allocated_gib"] for item in rank_metrics
            ),
            "max_peak_reserved_gib": max(
                item["peak_reserved_gib"] for item in rank_metrics
            ),
            "cache_bytes": final_vggt.stat().st_size
            + final_dpro.stat().st_size,
            "cache_gib": (
                final_vggt.stat().st_size + final_dpro.stat().st_size
            ) / 1024 ** 3,
            "dtype": dtype_name,
        }
        atomic_json(cache_root / "precompute_summary.json", summary)
        print(json.dumps(summary, indent=2))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
