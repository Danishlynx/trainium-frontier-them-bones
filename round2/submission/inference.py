"""
Reference Round 2 inference wrapper.

Participants may edit this file, but the scorer owns the CORE prompts, batching,
tokenizer, and metric computation. The wrapper must expose the same model that
was trained and return ordinary causal logits.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

import prepare
import train


class ModelWrapper(torch.nn.Module):
    """Minimal interface consumed by the trusted CORE evaluator."""

    def __init__(self, model: torch.nn.Module, max_seq_len: int):
        super().__init__()
        self.model = model
        self.max_seq_len = max_seq_len

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids)

    def get_eval_worker(self) -> dict[str, int | bool]:
        """Expose an opaque work assignment; the evaluator assumes no topology."""
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1
        # The reference model is replicated, so every rank is one independent
        # evaluation worker. Distributed submissions may group ranks however
        # they choose and return the same worker_id from ranks in one replica.
        return {
            "worker_id": rank,
            "worker_count": world_size,
            "is_leader": True,
        }

def load_for_eval(checkpoint_path: str, device: torch.device) -> ModelWrapper:
    """Load the submitted checkpoint as a replicated eval model on one rank."""
    prepare.ensure_tokenizer(build_if_missing=False)
    model = train.load_for_eval(checkpoint_path, device)
    max_seq_len = model.config.sequence_len
    if device.type == "neuron":
        model = torch.compile(model, backend="neuron", dynamic=False)
    return ModelWrapper(model, max_seq_len)
