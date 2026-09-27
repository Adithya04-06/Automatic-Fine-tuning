"""Resource profiling utilities for tracking time, memory, and compound computational cost."""

import time
import torch
import torch.nn as nn


class ResourceProfiler:
    """Tracks dynamic hardware utilization such as peak GPU memory and epoch latency."""

    @staticmethod
    def get_peak_memory_mb() -> float:
        """Returns peak allocated GPU memory in Megabytes (or 0.0 if CUDA is unavailable)."""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 2)
        return 0.0

    @staticmethod
    def reset_peak_memory() -> None:
        """Resets the peak memory tracker on CUDA devices."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    @staticmethod
    def compute_compound_cost(
        mean_epoch_time: float,
        trainable_params: int,
        all_epoch_times: list,
        all_trainable_params: list,
        time_weight: float = 0.5,
        param_weight: float = 0.5,
    ) -> float:
        """Calculates a normalized compound cost score between 0.0 and 1.0."""
        min_time, max_time = min(all_epoch_times), max(all_epoch_times)
        min_p, max_p = min(all_trainable_params), max(all_trainable_params)

        norm_time = (mean_epoch_time - min_time) / (max_time - min_time + 1e-8)
        norm_param = (trainable_params - min_p) / (max_p - min_p + 1e-8)

        total_weight = time_weight + param_weight
        return (time_weight * norm_time + param_weight * norm_param) / total_weight
