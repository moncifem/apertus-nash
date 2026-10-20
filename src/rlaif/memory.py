"""Unified-memory telemetry for DGX Spark (GB10).

On UMA, GPU allocations come out of system RAM, so "used = MemTotal - MemAvailable" sees the
policy vLLM engine, the trainer and the judge together. Training usage is estimated as the
increase over the baseline taken BEFORE the trainer loads anything (judge already resident).
Blind spots: other host processes started later are attributed to training; transient spikes
shorter than the sample interval can be missed. This is a watchdog, not a hard partition.
"""
from __future__ import annotations

import threading
import time


def meminfo() -> dict[str, int]:
    out = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            parts = v.split()
            out[k] = int(parts[0]) * (1024 if len(parts) > 1 else 1)
    return out


class MemoryMonitor(threading.Thread):
    def __init__(self, mc):
        super().__init__(daemon=True)
        self.mc = mc
        m = meminfo()
        self.total = m["MemTotal"]
        self.baseline_used = m["MemTotal"] - m["MemAvailable"]
        self.peak_training = 0
        self.min_available = m["MemAvailable"]
        self.breach: str | None = None
        self._stop = threading.Event()

    def sample(self) -> dict:
        m = meminfo()
        used = m["MemTotal"] - m["MemAvailable"]
        training = max(0, used - self.baseline_used)
        self.peak_training = max(self.peak_training, training)
        self.min_available = min(self.min_available, m["MemAvailable"])
        if training > self.mc.training_ceiling_bytes:
            self.breach = self.breach or "MEMORY_BUDGET_EXCEEDED"
        elif m["MemAvailable"] < self.mc.min_host_available_bytes:
            self.breach = self.breach or "HOST_MEMORY_LOW"
        return {"available": m["MemAvailable"], "training_est": training}

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                self.sample()
            except Exception:  # noqa: BLE001
                pass
            time.sleep(self.mc.sample_interval_s)

    def stop(self) -> None:
        self._stop.set()

    def snapshot(self) -> dict:
        cur = self.sample()
        snap = {
            "mem_total": self.total, "baseline_used": self.baseline_used,
            "available": cur["available"], "training_est": cur["training_est"],
            "peak_training_est": self.peak_training, "min_available": self.min_available,
            "method": "MemTotal-MemAvailable minus pre-trainer baseline (UMA)",
        }
        try:
            import torch

            if torch.cuda.is_available():
                snap["torch_max_allocated"] = torch.cuda.max_memory_allocated()
                snap["torch_reserved"] = torch.cuda.memory_reserved()
        except Exception:  # noqa: BLE001
            pass
        return snap
