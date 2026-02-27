"""
Thin bridge: grain IterDataset -> PyTorch tensors.

The grain pipeline (pipeline.py) already handles batching, shuffling,
and prefetching. This loader just converts numpy -> torch.Tensor.
"""

import torch
import numpy as np
from typing import Iterator
from ping_llm.data.pipeline import build_probe_chunk_dataset


def create_loader(
    arrayrecord_path: str,
    batch_size: int = 256,
    crop_size: int = 1024,
    shuffle: bool = True,
    shuffle_seed: int = 42,
    num_workers: int = 16,
    prefetch_buffer_size: int = 16,
    use_multiprocessing: bool = True,
    ram_budget_mb: int = 32768,
    device: str = "cpu",
) -> Iterator[dict[str, torch.Tensor]]:
    """
    Create an infinite iterator yielding PyTorch tensor batches.

    Each batch dict contains:
        inputs: [B, seq_len] long
        targets: [B, seq_len] long
        targets_segmentation: [B, seq_len] long (1=valid, 0=padding)

    The grain pipeline does all the heavy lifting (ArrayRecord reading,
    probe row sampling, tokenization, batching, mp_prefetch).
    We just convert numpy arrays to torch tensors.
    """
    dataset = build_probe_chunk_dataset(
        arrayrecord_path=arrayrecord_path,
        batch_size=batch_size,
        crop_size=crop_size,
        shuffle=shuffle,
        shuffle_seed=shuffle_seed,
        num_workers=num_workers,
        prefetch_buffer_size=prefetch_buffer_size,
        use_multiprocessing=use_multiprocessing,
        ram_budget_mb=ram_budget_mb,
    )

    for batch_np in dataset:
        yield {
            "inputs": torch.from_numpy(np.asarray(batch_np["inputs"])).long().to(device),
            "targets": torch.from_numpy(np.asarray(batch_np["targets"])).long().to(device),
            "targets_segmentation": torch.from_numpy(
                np.asarray(batch_np["targets_segmentation"])
            ).long().to(device),
        }
