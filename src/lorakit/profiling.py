import os
import time
from collections import OrderedDict

import torch


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() not in ("0", "", "false", "no", "off")


class StepProfiler:
    """
    Lightweight per-section profiler for the training loop. Disabled by default.

    Controlled by the ``profile`` block in the training config, e.g.::

        train:
          profile:
            enabled: true     # default false
            warmup: 5         # steps excluded from the averages
            report_every: 0   # 0 = only report at the end

    The environment variable ``LORAKIT_PROFILE=1`` can force it on without
    editing the config (``LORAKIT_PROFILE_WARMUP`` / ``LORAKIT_PROFILE_EVERY``
    override the corresponding values when set).

    GPU sections are timed with CUDA events (so they account for asynchronous
    kernel execution) while CPU/host gaps such as waiting on the dataloader are
    timed with the wall clock.
    """

    def __init__(self, enabled=False, warmup=3, report_every=0):
        env_force = os.environ.get("LORAKIT_PROFILE")
        if env_force is not None:
            enabled = _as_bool(env_force)
        self.enabled = bool(enabled)
        self.cuda = self.enabled and torch.cuda.is_available()
        self.report_every = self._env_int("LORAKIT_PROFILE_EVERY", report_every)
        # Skip the first few steps from the averages: cuDNN autotuning / lazy
        # allocator growth / graph capture make them unrepresentatively slow.
        self.warmup = self._env_int("LORAKIT_PROFILE_WARMUP", warmup)
        self._totals = OrderedDict()
        self._counts = OrderedDict()
        self._events = []  # list of (label, start_event, end_event)
        self._open = {}  # label -> start event (GPU) or perf_counter (CPU)
        self._wall = {}
        self._pending = []  # (label, ms) buffered for the current step
        self._n_steps = 0  # total steps seen (incl. warmup)
        self._recorded = 0  # steps actually accumulated into totals

    @staticmethod
    def _env_int(name, default):
        try:
            return int(os.environ.get(name, str(default)))
        except ValueError:
            return default

    # -- GPU/CPU section timing (tic/toc, no re-indentation needed) --
    def tic(self, label):
        if not self.enabled:
            return
        if self.cuda:
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            self._open[label] = start
        else:
            self._open[label] = time.perf_counter()

    def toc(self, label):
        if not self.enabled or label not in self._open:
            return
        if self.cuda:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self._events.append((label, self._open.pop(label), end))
        else:
            self._pending.append((label, (time.perf_counter() - self._open.pop(label)) * 1000.0))

    def _add(self, label, ms):
        self._totals[label] = self._totals.get(label, 0.0) + ms
        self._counts[label] = self._counts.get(label, 0) + 1

    # -- wall-clock timing for host gaps (e.g. dataloader wait) --
    def wall_start(self, label):
        if self.enabled:
            self._wall[label] = time.perf_counter()

    def wall_stop(self, label):
        if self.enabled and label in self._wall:
            self._pending.append((label, (time.perf_counter() - self._wall.pop(label)) * 1000.0))

    def step_end(self):
        """Flush CUDA event timings for the step and accumulate (past warmup)."""
        if not self.enabled:
            return
        if self.cuda and self._events:
            torch.cuda.synchronize()
            for label, start, end in self._events:
                self._pending.append((label, start.elapsed_time(end)))
            self._events.clear()
        if self._n_steps >= self.warmup:
            for label, ms in self._pending:
                self._add(label, ms)
            self._recorded += 1
        self._pending.clear()
        self._n_steps += 1
        if self.report_every and self._n_steps % self.report_every == 0:
            self.report(prefix=f"[profile @ step {self._n_steps}]")

    def report(self, prefix="[profile]"):
        if not self.enabled or self._recorded == 0:
            return
        print(
            f"\n{prefix} averages over {self._recorded} step(s) "
            f"(first {self.warmup} warmup step(s) excluded):"
        )
        total_ms = 0.0
        rows = []
        for label, total in self._totals.items():
            per_step = total / self._recorded
            rows.append((label, per_step))
            total_ms += per_step
        width = max(len(label) for label, _ in rows)
        for label, per_step in rows:
            pct = (per_step / total_ms * 100.0) if total_ms > 0 else 0.0
            print(f"  {label.ljust(width)} : {per_step:8.2f} ms/step  ({pct:5.1f}%)")
        print(f"  {'TOTAL (sum)'.ljust(width)} : {total_ms:8.2f} ms/step")
        if self.cuda:
            mem = torch.cuda.max_memory_allocated() / (1024**3)
            reserved = torch.cuda.max_memory_reserved() / (1024**3)
            print(f"  peak CUDA memory: {mem:.2f} GiB allocated / {reserved:.2f} GiB reserved")
        print()
