#!/usr/bin/env python3
"""Cached LSA pretraining for one or four GPUs.

VGGT and Depth Pro are absent during training; their spatially aligned,
normalized targets are read from disk.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import statistics
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from model import RobotTransformerAR
from dataset import (
    CachedLSADataset,
    FixedGlobalBatchSampler,
    cached_collate,
    original_split_indices,
    roots_from_data_root,
)


SF_ALPHA = 0.3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--backbone", default="google/vit-base-patch16-224-in21k"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--local-batch-size", type=int, default=16)
    parser.add_argument("--global-batch-size", type=int, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--no-cache-warm", action="store_true")
    return parser.parse_args()


def normalize_map(tensor: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    minimum = tensor.min(dim=-1, keepdim=True)[0]
    maximum = tensor.max(dim=-1, keepdim=True)[0]
    return (tensor - minimum) / (maximum - minimum + epsilon)


def build_2d_sincos_pos_embed(
    height: int, width: int, dim: int, device=None
) -> torch.Tensor:
    device = device or torch.device("cpu")

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


class MAEDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        decoder_dim: int = 512,
        decoder_depth: int = 8,
        patch_size: int = 16,
        img_size: Tuple[int, int] = (224, 224),
        out_ch: int = 1,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(embed_dim, decoder_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=8,
            dim_feedforward=decoder_dim * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=decoder_depth)
        self.head = nn.Linear(
            decoder_dim, out_ch * patch_size * patch_size
        )
        patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        position = build_2d_sincos_pos_embed(
            img_size[0] // patch_size,
            img_size[1] // patch_size,
            decoder_dim,
        )
        self.register_buffer(
            "pos_embed",
            position.view(1, patches, decoder_dim),
            persistent=False,
        )
        self.fold = nn.Fold(
            output_size=img_size,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.proj(tensor)
        tensor = tensor + self.pos_embed.to(
            dtype=tensor.dtype, device=tensor.device
        )
        tensor = self.decoder(tensor)
        tensor = self.head(tensor).permute(0, 2, 1)
        return self.fold(tensor)


class TwoLayerProjection(nn.Module):
    def __init__(self, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(768)
        self.fc1 = nn.Linear(768, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.GELU()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = tensor.shape
        tensor = tensor.reshape(batch * tokens, channels)
        tensor = self.bn(tensor)
        tensor = self.fc2(self.activation(self.fc1(tensor)))
        tensor = F.normalize(tensor, dim=-1)
        return tensor.view(batch, tokens, -1)


class CachedLSAModel(nn.Module):
    def __init__(
        self,
        prediction_length: int = 3,
        backbone: str = "google/vit-base-patch16-224-in21k",
    ) -> None:
        super().__init__()
        self.policy = RobotTransformerAR(
            pre_len=prediction_length, backbone=backbone
        )
        self.vggt_head = TwoLayerProjection(2048, 1024)
        self.dpro_head = TwoLayerProjection(1024, 768)
        self.depth_decoder = MAEDecoder()
        self.prediction_length = prediction_length

    def forward(
        self,
        images: torch.Tensor,
        depth: torch.Tensor,
        kpmap: torch.Tensor,
        targets: torch.Tensor,
        vggt_target: torch.Tensor,
        dpro_target: torch.Tensor,
        validation: bool = False,
        motion_only: bool = False,
    ) -> Dict[str, torch.Tensor]:
        batch, sequence, channels, height, width = images.shape
        attentions, encoded, _, hidden = self.policy.vit(
            images.view(batch * sequence, channels, height, width)
        )
        pooled = encoded.mean(dim=1).view(batch, sequence, -1)
        prediction, _ = self.policy(object_pc=pooled)
        action = prediction["action"][:, -1, :].reshape(
            -1, self.prediction_length, 3
        )
        loss_prediction = F.l1_loss(action, targets.squeeze(1))
        if validation or motion_only:
            return {"loss_prediction": loss_prediction}

        attention_stack = torch.stack(attentions[:6], dim=1)
        _, layers, heads, tokens, _ = attention_stack.shape
        hand = F.interpolate(
            kpmap.squeeze(2), size=(14, 14),
            mode="bilinear", align_corners=False,
        )
        hand = hand / (hand.amax(dim=(2, 3), keepdim=True) + 1e-6)
        cls_attention = (
            attention_stack[:, :, :, 0, 1:]
            .reshape(batch, layers * heads, 14, 14)
            .clamp_(0, 1)
            .detach()
        )
        score = (cls_attention * hand).mean(dim=(2, 3))
        selected_index = torch.topk(score, k=1, dim=1).indices
        gather_index = selected_index.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, 14, 14
        )
        final_map = torch.gather(
            cls_attention, dim=1, index=gather_index
        ).mean(dim=1).flatten(1)
        student_maps = attention_stack.mean(dim=2)[:, :, 0, 1:]
        loss_attention = sum(
            F.mse_loss(
                normalize_map(final_map).detach(),
                normalize_map(student_maps[:, layer]),
            )
            for layer in range(6)
        )

        layer3 = hidden[3][:, 1:, :]
        layer6 = hidden[6][:, 1:, :]
        layer3_cls = hidden[3][:, 0, :]
        layer6_cls = hidden[6][:, 0, :]
        vggt_student = self.vggt_head(layer3)
        dpro_student = self.dpro_head(layer3)
        vggt_target = vggt_target.to(vggt_student.dtype)
        dpro_target = dpro_target.to(dpro_student.dtype)
        vggt_loss_per_sample = (
            1.0 - (vggt_student * vggt_target).sum(dim=-1)
        ).mean(dim=1)
        dpro_loss_per_sample = (
            1.0 - (dpro_student * dpro_target).sum(dim=-1)
        ).mean(dim=1)

        fused = torch.cat([layer3_cls, layer6_cls], dim=-1)
        query = fused.view(batch, 3, 512).mean(dim=1).unsqueeze(1)
        vggt_512 = vggt_target.view(
            batch, 196, 4, 512
        ).mean(dim=2)
        dpro_512 = dpro_target.view(
            batch, 196, 2, 512
        ).mean(dim=2)
        vggt_score = (
            (query * vggt_512).sum(dim=-1) / math.sqrt(512)
        ).mean(dim=1)
        dpro_score = (
            (query * dpro_512).sum(dim=-1) / math.sqrt(512)
        ).mean(dim=1)
        gate = torch.softmax(
            torch.stack([vggt_score, dpro_score], dim=-1), dim=-1
        )

        predicted_depth = self.depth_decoder(layer6)
        ground_truth_depth = depth[:, 0, 0].unsqueeze(1)
        loss_depth = F.mse_loss(predicted_depth, ground_truth_depth)
        return {
            "loss_prediction": loss_prediction,
            "loss_attention": loss_attention,
            "loss_depth": loss_depth,
            "vggt_loss_per_sample": vggt_loss_per_sample,
            "dpro_loss_per_sample": dpro_loss_per_sample,
            "gate": gate,
        }


class AllGatherWithGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor) -> torch.Tensor:
        ctx.rank = dist.get_rank()
        ctx.world = dist.get_world_size()
        gathered = [torch.empty_like(tensor) for _ in range(ctx.world)]
        dist.all_gather(gathered, tensor)
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        gradient = gradient.contiguous()
        dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
        return gradient.chunk(ctx.world, dim=0)[ctx.rank].contiguous()


def all_gather_gradient(tensor: torch.Tensor) -> torch.Tensor:
    return AllGatherWithGradient.apply(tensor)


def reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    result = tensor.detach().clone()
    dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return result / dist.get_world_size()


def atomic_json(path: Path, payload: Dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def warm_file(path: Path, block_size: int = 128 * 1024 * 1024) -> int:
    total = 0
    with path.open("rb", buffering=0) as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            total += len(block)
    return total


def percentile(values: List[float], percent: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values), percent))


def main() -> None:
    args = parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if world not in (1, 4):
        raise RuntimeError(f"Expected 1 or 4 ranks, got {world}")
    if args.global_batch_size is None:
        args.global_batch_size = world * args.local_batch_size
    if args.global_batch_size != world * args.local_batch_size:
        raise RuntimeError("Global batch must equal world * local batch")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    output_root = Path(args.output_root)
    if rank == 0:
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "checkpoints").mkdir(exist_ok=True)
    dist.barrier()
    run_start = time.perf_counter()

    dataset = CachedLSADataset(
        args.cache_root,
        roots=roots_from_data_root(args.data_root),
    )
    train_indices, validation_indices = original_split_indices(
        len(dataset), seed=args.seed
    )
    train_sampler = FixedGlobalBatchSampler(
        train_indices, rank, world,
        global_batch_size=args.global_batch_size,
        local_batch_size=args.local_batch_size,
    )
    validation_sampler = FixedGlobalBatchSampler(
        validation_indices, rank, world,
        global_batch_size=args.global_batch_size,
        local_batch_size=args.local_batch_size,
    )
    loader_options = {
        "batch_size": args.local_batch_size,
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
        "prefetch_factor": 2 if args.workers > 0 else None,
        "collate_fn": cached_collate,
        "drop_last": True,
    }
    train_loader = DataLoader(
        dataset, sampler=train_sampler, **loader_options
    )
    validation_loader = DataLoader(
        dataset, sampler=validation_sampler, **loader_options
    )
    expected_train_steps = len(train_indices) // args.global_batch_size
    expected_validation_steps = (
        len(validation_indices) // args.global_batch_size
    )
    if (
        len(train_loader) != expected_train_steps
        or len(validation_loader) != expected_validation_steps
    ):
        raise RuntimeError(
            f"Unexpected loader sizes: {len(train_loader)}, "
            f"{len(validation_loader)}"
        )

    cache_warm_seconds = 0.0
    cache_warm_bytes = 0
    if rank == 0 and not args.no_cache_warm:
        warm_start = time.perf_counter()
        manifest = dataset.manifest
        for name in ("vggt", "dpro"):
            cache_warm_bytes += warm_file(
                Path(args.cache_root)
                / manifest["arrays"][name]["file"]
            )
        cache_warm_seconds = time.perf_counter() - warm_start
    warm_tensor = torch.tensor(
        [cache_warm_seconds, cache_warm_bytes],
        dtype=torch.float64, device=device,
    )
    dist.broadcast(warm_tensor, src=0)
    cache_warm_seconds = warm_tensor[0].item()
    cache_warm_bytes = int(warm_tensor[1].item())
    dist.barrier()

    model = CachedLSAModel(
        prediction_length=3, backbone=args.backbone
    ).to(device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=True,
        gradient_as_bucket_view=True,
        # Auxiliary heads are intentionally unused during motion-only warm-up.
        find_unused_parameters=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    dist.barrier()
    torch.cuda.reset_peak_memory_stats(device)
    if args.validation_only:
        model.eval()
        validation_start = time.perf_counter()
        validation_sum = torch.zeros(
            2, device=device, dtype=torch.float64
        )
        with torch.inference_mode():
            for batch in validation_loader:
                components = model(
                    batch["images"].to(device, non_blocking=True),
                    batch["depths"].to(device, non_blocking=True),
                    batch["kpmaps"].to(device, non_blocking=True),
                    batch["targets"].to(device, non_blocking=True),
                    batch["vggt_target"].to(device, non_blocking=True),
                    batch["dpro_target"].to(device, non_blocking=True),
                    validation=True,
                )
                validation_sum[0] += (
                    components["loss_prediction"].double()
                    * args.local_batch_size
                )
                validation_sum[1] += args.local_batch_size
        torch.cuda.synchronize(device)
        local_seconds = time.perf_counter() - validation_start
        dist.all_reduce(validation_sum, op=dist.ReduceOp.SUM)
        duration = torch.tensor(
            local_seconds, device=device, dtype=torch.float64
        )
        dist.all_reduce(duration, op=dist.ReduceOp.MAX)
        peak = torch.tensor([
            torch.cuda.max_memory_allocated(device) / 1024 ** 3,
            torch.cuda.max_memory_reserved(device) / 1024 ** 3,
        ], device=device, dtype=torch.float64)
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        if rank == 0:
            result = {
                "status": "validation_benchmark",
                "steps": len(validation_loader),
                "samples": int(validation_sum[1].item()),
                "wall_seconds": duration.item(),
                "samples_per_second": (
                    validation_sum[1].item() / duration.item()
                ),
                "loss": (
                    validation_sum[0] / validation_sum[1]
                ).item(),
                "max_peak_allocated_gib": peak[0].item(),
                "max_peak_reserved_gib": peak[1].item(),
            }
            atomic_json(output_root / "validation_benchmark.json", result)
            print(json.dumps(result, indent=2))
        dist.barrier()
        dist.destroy_process_group()
        return
    training_start = time.perf_counter()
    epoch_records: List[Dict[str, object]] = []
    all_step_gpu_ms: List[float] = []
    completed_steps = 0
    stopped_for_smoke = False
    sample_order_verified = False

    for epoch in range(args.epochs):
        model.train()
        motion_only = epoch < args.warmup_epochs
        epoch_start = time.perf_counter()
        step_events: List[Tuple[torch.cuda.Event, torch.cuda.Event]] = []
        summed_losses = {
            "total": 0.0,
            "prediction": 0.0,
            "teacher": 0.0,
            "depth": 0.0,
            "attention": 0.0,
        }
        epoch_steps = 0
        for batch in train_loader:
            if not sample_order_verified:
                local_indices = batch["global_index"].to(device)
                gathered_indices = [
                    torch.empty_like(local_indices) for _ in range(world)
                ]
                dist.all_gather(gathered_indices, local_indices)
                global_indices = torch.cat(gathered_indices).cpu().tolist()
                expected = train_indices[:args.global_batch_size]
                if global_indices != expected:
                    raise RuntimeError(
                        f"Global batch order mismatch: {global_indices} != "
                        f"{expected}"
                    )
                sample_order_verified = True
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            images = batch["images"].to(device, non_blocking=True)
            depths = batch["depths"].to(device, non_blocking=True)
            kpmaps = batch["kpmaps"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            vggt_target = batch["vggt_target"].to(
                device, non_blocking=True
            )
            dpro_target = batch["dpro_target"].to(
                device, non_blocking=True
            )
            optimizer.zero_grad(set_to_none=True)
            components = model(
                images, depths, kpmaps, targets,
                vggt_target, dpro_target,
                validation=False,
                motion_only=motion_only,
            )
            if motion_only:
                zero = components["loss_prediction"].new_zeros(())
                teacher_loss = zero
                loss_depth = zero
                loss_attention = zero
                loss = components["loss_prediction"]
            else:
                global_vggt = all_gather_gradient(
                    components["vggt_loss_per_sample"]
                )
                global_dpro = all_gather_gradient(
                    components["dpro_loss_per_sample"]
                )
                global_gate = all_gather_gradient(components["gate"])
                weights = global_gate.mean(dim=0)
                teacher_loss = (
                    weights[0] * global_vggt.mean()
                    + weights[1] * global_dpro.mean()
                )
                loss_depth = components["loss_depth"]
                loss_attention = components["loss_attention"]
                spatial_loss = (
                    teacher_loss
                    + loss_depth
                    + 0.5 * loss_attention
                )
                loss = (
                    components["loss_prediction"]
                    + SF_ALPHA * spatial_loss
                )
            loss.backward()
            optimizer.step()
            end.record()
            step_events.append((begin, end))

            reported = torch.stack([
                loss.detach(),
                components["loss_prediction"].detach(),
                teacher_loss.detach(),
                loss_depth.detach(),
                loss_attention.detach(),
            ])
            reported = reduce_mean(reported)
            for key, value in zip(summed_losses, reported.tolist()):
                summed_losses[key] += value
            epoch_steps += 1
            completed_steps += 1
            if args.smoke_steps and completed_steps >= args.smoke_steps:
                stopped_for_smoke = True
                break

        torch.cuda.synchronize(device)
        epoch_seconds_local = time.perf_counter() - epoch_start
        duration_tensor = torch.tensor(
            epoch_seconds_local, device=device, dtype=torch.float64
        )
        dist.all_reduce(duration_tensor, op=dist.ReduceOp.MAX)
        epoch_seconds = duration_tensor.item()
        local_step_ms = [
            begin.elapsed_time(end) for begin, end in step_events
        ]
        gathered_step_ms: List[List[float]] = [
            [] for _ in range(world)
        ]
        dist.all_gather_object(gathered_step_ms, local_step_ms)
        if rank == 0:
            # A global step completes when its slowest rank completes.
            for position in range(len(local_step_ms)):
                all_step_gpu_ms.append(max(
                    values[position] for values in gathered_step_ms
                ))
            epoch_records.append({
                "epoch": epoch + 1,
                "steps": epoch_steps,
                "wall_seconds": epoch_seconds,
                "samples_per_second": (
                    epoch_steps * args.global_batch_size / epoch_seconds
                ),
                "losses": {
                    key: value / max(epoch_steps, 1)
                    for key, value in summed_losses.items()
                },
                "lr": optimizer.param_groups[0]["lr"],
            })

        if stopped_for_smoke:
            break

        validation_loss = None
        if (epoch + 1) % 25 == 0:
            model.eval()
            validation_sum = torch.zeros(
                2, device=device, dtype=torch.float64
            )
            with torch.inference_mode():
                for batch in validation_loader:
                    components = model(
                        batch["images"].to(device, non_blocking=True),
                        batch["depths"].to(device, non_blocking=True),
                        batch["kpmaps"].to(device, non_blocking=True),
                        batch["targets"].to(device, non_blocking=True),
                        batch["vggt_target"].to(device, non_blocking=True),
                        batch["dpro_target"].to(device, non_blocking=True),
                        validation=True,
                    )
                    validation_sum[0] += (
                        components["loss_prediction"].double()
                        * args.local_batch_size
                    )
                    validation_sum[1] += args.local_batch_size
            dist.all_reduce(validation_sum, op=dist.ReduceOp.SUM)
            validation_loss = (
                validation_sum[0] / validation_sum[1]
            ).item()
            if rank == 0:
                epoch_records[-1]["validation_loss"] = validation_loss
                policy_path = (
                    output_root / "checkpoints"
                    / f"policy_epoch{epoch + 1}.pth"
                )
                full_path = (
                    output_root / "checkpoints"
                    / f"training_epoch{epoch + 1}.pth"
                )
                torch.save(model.module.policy.state_dict(), policy_path)
                torch.save({
                    "epoch": epoch + 1,
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "validation_loss": validation_loss,
                    "global_batch_size": args.global_batch_size,
                    "world_size": world,
                }, full_path)
        scheduler.step()
        if rank == 0:
            latest = epoch_records[-1]
            validation_text = (
                f", val={latest['validation_loss']:.6f}"
                if "validation_loss" in latest else ""
            )
            print(
                f"[epoch {epoch + 1:03d}/{args.epochs}] "
                f"steps={epoch_steps}, wall={epoch_seconds:.3f}s, "
                f"samples/s={latest['samples_per_second']:.3f}, "
                f"loss={latest['losses']['total']:.6f}"
                f"{validation_text}",
                flush=True,
            )
        dist.barrier()

    torch.cuda.synchronize(device)
    training_seconds_local = time.perf_counter() - training_start
    duration = torch.tensor(
        training_seconds_local, device=device, dtype=torch.float64
    )
    dist.all_reduce(duration, op=dist.ReduceOp.MAX)
    peak = torch.tensor([
        torch.cuda.max_memory_allocated(device) / 1024 ** 3,
        torch.cuda.max_memory_reserved(device) / 1024 ** 3,
    ], device=device, dtype=torch.float64)
    peaks = [torch.zeros_like(peak) for _ in range(world)]
    dist.all_gather(peaks, peak)

    rank_record = {
        "rank": rank,
        "local_rank": local_rank,
        "hostname": socket.gethostname(),
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_total_mib": torch.cuda.get_device_properties(device).total_memory
        / 1024 ** 2,
        "peak_allocated_gib": peak[0].item(),
        "peak_reserved_gib": peak[1].item(),
        "training_seconds": training_seconds_local,
    }
    atomic_json(output_root / f"training_rank{rank}.json", rank_record)
    dist.barrier()

    if rank == 0:
        total_training_seconds = duration.item()
        wall_from_launch = time.perf_counter() - run_start
        measured_steps = len(all_step_gpu_ms)
        summary = {
            "status": "smoke" if stopped_for_smoke else "complete",
            "hardware": {
                "world_size": world,
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_total_mib_each": torch.cuda.get_device_properties(
                    device
                ).total_memory / 1024 ** 2,
                "host_ram_note": "cache warmed into shared OS page cache",
            },
            "software": {
                "python": platform.python_version(),
                "pytorch": torch.__version__,
                "cuda": torch.version.cuda,
                "backend": dist.get_backend(),
            },
            "data": {
                "total_samples": len(dataset),
                "train_split": len(train_indices),
                "validation_split": len(validation_indices),
                "train_consumed_per_epoch": (
                    expected_train_steps * args.global_batch_size
                ),
                "validation_consumed": (
                    expected_validation_steps * args.global_batch_size
                ),
            },
            "configuration": {
                "epochs_requested": args.epochs,
                "epochs_completed": len(epoch_records),
                "global_batch_size": args.global_batch_size,
                "local_batch_size": args.local_batch_size,
                "warmup_epochs": args.warmup_epochs,
                "steps_per_epoch": expected_train_steps,
                "optimizer_updates_completed": completed_steps,
                "learning_rate": 0.0005,
                "seed": args.seed,
                "workers_per_rank": args.workers,
                "sample_order_verified": sample_order_verified,
            },
            "timing": {
                "cache_warm_seconds": cache_warm_seconds,
                "cache_warm_bytes": cache_warm_bytes,
                "training_seconds": total_training_seconds,
                "end_to_end_seconds": wall_from_launch,
                "gpu_hours": total_training_seconds * world / 3600.0,
                "global_samples_per_second": (
                    completed_steps * args.global_batch_size
                    / total_training_seconds
                ),
                "step_gpu_ms_mean": statistics.mean(all_step_gpu_ms)
                if all_step_gpu_ms else None,
                "step_gpu_ms_std": statistics.pstdev(all_step_gpu_ms)
                if len(all_step_gpu_ms) > 1 else 0.0,
                "step_gpu_ms_p50": percentile(all_step_gpu_ms, 50),
                "step_gpu_ms_p95": percentile(all_step_gpu_ms, 95),
                "measured_steps": measured_steps,
            },
            "memory": {
                "per_rank": [
                    {
                        "rank": index,
                        "peak_allocated_gib": item[0].item(),
                        "peak_reserved_gib": item[1].item(),
                    }
                    for index, item in enumerate(peaks)
                ],
                "max_peak_allocated_gib": max(
                    item[0].item() for item in peaks
                ),
                "max_peak_reserved_gib": max(
                    item[1].item() for item in peaks
                ),
            },
            "epochs": epoch_records,
        }
        atomic_json(output_root / "training_summary.json", summary)
        print(json.dumps(summary, indent=2))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
