"""Application-level WAN + compute heterogeneity emulator.

When all nodes run on one host (local/Docker runs) the loopback link has ~0 latency and
unlimited bandwidth, which hides exactly the costs this project studies. The emulator
delays each transfer by  rtt/2 + bytes*8 / bandwidth  and pads local training to
base_compute_s * compute_factor * epochs, using per-client profiles in configs/profiles.json.

On real, separate cloud nodes set FEDCLOUD_NO_EMU=1 so only the true network is measured.
"""
from __future__ import annotations

import json
import os
import time


class NetEmu:
    def __init__(self, profile: dict | None, base_compute_s: float = 0.0):
        self.enabled = profile is not None and os.environ.get("FEDCLOUD_NO_EMU") != "1"
        self.p = profile or {}
        self.base = base_compute_s

    @classmethod
    def from_file(cls, path: str, client_id: int, offset: int = 0):
        """Client k uses profile (k + offset) mod P; offset lets you move the straggler role."""
        with open(path) as f:
            cfg = json.load(f)
        prof = cfg["clients"][(client_id + offset) % len(cfg["clients"])]
        return cls(prof, cfg.get("base_compute_s", 0.0))

    def transfer(self, n_bytes: int) -> float:
        """Block for one-way transfer time of n_bytes; return seconds slept."""
        if not self.enabled:
            return 0.0
        d = self.p["rtt_ms"] / 2000.0 + n_bytes * 8 / (self.p["bw_mbps"] * 1e6)
        time.sleep(d)
        return d

    def pad_compute(self, started: float, epochs: int) -> float:
        """Sleep so that local training lasts at least base*factor*epochs seconds."""
        if not self.enabled:
            return 0.0
        target = self.base * self.p["compute_factor"] * epochs
        rest = target - (time.time() - started)
        if rest > 0:
            time.sleep(rest)
            return rest
        return 0.0
