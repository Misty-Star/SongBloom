"""单层 VQ codebook 的 streaming K-Means 训练器。"""

from __future__ import annotations

import math

import torch


class StreamingKMeans:
    def __init__(self, num_codes: int, embed_dim: int, device: str, seed: int):
        self.num_codes = int(num_codes)
        self.embed_dim = int(embed_dim)
        self.device = device
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)
        self.seed = int(seed)

        self.centers: torch.Tensor | None = None
        self.counts: torch.Tensor | None = None

    def initialize(self, init_samples: torch.Tensor) -> None:
        if init_samples.ndim != 2:
            raise ValueError(f"`init_samples` must be 2D, got {tuple(init_samples.shape)}")
        if init_samples.shape[0] < self.num_codes:
            raise ValueError(
                f"Need at least {self.num_codes} init samples, but only got {init_samples.shape[0]}."
            )

        init_samples = init_samples.detach().to(dtype=torch.float32, device="cpu")
        if init_samples.shape[1] != self.embed_dim:
            raise ValueError(
                f"Expected embedding dim {self.embed_dim}, got {init_samples.shape[1]}."
            )

        if init_samples.shape[0] == self.num_codes:
            centers = init_samples
        else:
            indices = torch.randperm(init_samples.shape[0], generator=self.generator)[: self.num_codes]
            centers = init_samples[indices]
        self.centers = centers.to(dtype=torch.float32, device=self.device).clone()
        self.counts = torch.zeros(self.num_codes, device=self.centers.device, dtype=torch.float32)

    def _ensure_initialized(self) -> None:
        if self.centers is None or self.counts is None:
            raise RuntimeError("StreamingKMeans must be initialized before use.")

    def assign(self, batch: torch.Tensor) -> torch.Tensor:
        self._ensure_initialized()
        batch = batch.to(self.centers.device, dtype=torch.float32)
        distances = torch.cdist(batch, self.centers)
        return distances.argmin(dim=-1)

    def partial_fit(self, batch: torch.Tensor) -> torch.Tensor:
        self._ensure_initialized()
        batch = batch.to(self.centers.device, dtype=torch.float32)
        assignments = self.assign(batch)

        unique_ids = assignments.unique()
        for cluster_id in unique_ids.tolist():
            mask = assignments == cluster_id
            points = batch[mask]
            if points.numel() == 0:
                continue
            self.counts[cluster_id] += points.shape[0]
            eta = points.shape[0] / self.counts[cluster_id].clamp_min(1.0)
            self.centers[cluster_id] = (1.0 - eta) * self.centers[cluster_id] + eta * points.mean(dim=0)
        return assignments

    def refresh_dead_codes(self, reserve: torch.Tensor) -> int:
        self._ensure_initialized()
        reserve = reserve.to(self.centers.device, dtype=torch.float32)
        dead = self.counts <= 0
        dead_count = int(dead.sum().item())
        if dead_count == 0:
            return 0
        if reserve.shape[0] < dead_count:
            raise ValueError(
                f"Need at least {dead_count} reserve samples, but only got {reserve.shape[0]}."
            )
        self.centers[dead] = reserve[:dead_count]
        self.counts[dead] = 1.0
        return dead_count

    def fit(
        self,
        samples: torch.Tensor,
        batch_size: int,
        num_steps: int,
        refresh_every: int = 0,
    ) -> "StreamingKMeans":
        if samples.ndim != 2:
            raise ValueError(f"`samples` must be 2D, got {tuple(samples.shape)}")
        if samples.shape[0] < self.num_codes:
            raise ValueError(
                f"Need at least {self.num_codes} sampled frames, but only got {samples.shape[0]}."
            )

        working = samples.detach().to(dtype=torch.float32, device="cpu")
        self.initialize(working)
        current_batch_size = min(max(1, int(batch_size)), working.shape[0])

        for step_index in range(int(num_steps)):
            indices = torch.randint(
                low=0,
                high=working.shape[0],
                size=(current_batch_size,),
                generator=self.generator,
            )
            batch = working[indices].to(self.centers.device, dtype=torch.float32)
            self.partial_fit(batch)

            if refresh_every > 0 and (step_index + 1) % refresh_every == 0:
                reserve_indices = torch.randperm(working.shape[0], generator=self.generator)[: self.num_codes]
                reserve = working[reserve_indices].to(self.centers.device, dtype=torch.float32)
                self.refresh_dead_codes(reserve)

        return self

    def refine_full(self, samples: torch.Tensor, batch_size: int) -> torch.Tensor:
        self._ensure_initialized()
        working = samples.detach().to(dtype=torch.float32, device="cpu")

        sums = torch.zeros_like(self.centers)
        new_counts = torch.zeros_like(self.counts)
        chunk = min(max(1, int(batch_size)), working.shape[0])

        for start in range(0, working.shape[0], chunk):
            batch = working[start:start + chunk].to(self.centers.device, dtype=torch.float32)
            assignments = self.assign(batch)
            unique_ids = assignments.unique()
            for cluster_id in unique_ids.tolist():
                mask = assignments == cluster_id
                points = batch[mask]
                if points.numel() == 0:
                    continue
                sums[cluster_id] += points.sum(dim=0)
                new_counts[cluster_id] += points.shape[0]

        empty = new_counts <= 0
        if empty.any():
            refill_ids = torch.randperm(working.shape[0], generator=self.generator)[: int(empty.sum().item())]
            refill = working[refill_ids].to(self.centers.device, dtype=torch.float32)
            sums[empty] = refill
            new_counts[empty] = 1.0

        self.centers = sums / new_counts.unsqueeze(-1)
        self.counts = new_counts
        return self.centers.detach().cpu()

    def quantization_mse(self, samples: torch.Tensor) -> float:
        self._ensure_initialized()
        working = samples.detach().to(dtype=torch.float32, device="cpu")
        chunk = min(max(1, 4096), working.shape[0])
        squared_error_sum = 0.0
        total_values = 0
        for start in range(0, working.shape[0], chunk):
            batch = working[start:start + chunk].to(self.centers.device, dtype=torch.float32)
            assignments = self.assign(batch)
            reconstructed = self.centers[assignments]
            squared_error_sum += float(torch.sum((batch - reconstructed) ** 2).item())
            total_values += int(batch.numel())
        return squared_error_sum / max(total_values, 1)

    def usage_stats(self) -> dict:
        self._ensure_initialized()
        counts = self.counts.detach().cpu().float()
        total = counts.sum().clamp_min(1.0)
        probs = counts / total
        nonzero = probs[probs > 0]
        usage_entropy = float((-(nonzero * nonzero.log())).sum().item()) if nonzero.numel() else 0.0
        return {
            "dead_code_ratio": float((counts <= 0).float().mean().item()),
            "top_1_usage_share": float(probs.max().item()) if probs.numel() else 0.0,
            "usage_entropy": usage_entropy,
            "perplexity": float(math.exp(usage_entropy)) if usage_entropy > 0 else 1.0,
        }

    def state_dict(self) -> dict:
        self._ensure_initialized()
        return {
            "num_codes": self.num_codes,
            "embed_dim": self.embed_dim,
            "seed": self.seed,
            "centers": self.centers.detach().cpu(),
            "counts": self.counts.detach().cpu(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.num_codes = int(state_dict["num_codes"])
        self.embed_dim = int(state_dict["embed_dim"])
        self.seed = int(state_dict.get("seed", self.seed))
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(self.seed)
        self.centers = state_dict["centers"].to(self.device, dtype=torch.float32)
        self.counts = state_dict["counts"].to(self.device, dtype=torch.float32)
