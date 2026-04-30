"""
GLM-Image inference timer.

Wraps a MODEL with a function-wrapper that records elapsed time per
diffusion step + total. Output is a TIMING_REPORT string that can be
fed to a `ShowText`-style preview node, plus pass-through MODEL.
"""

# MANUAL bug-fix (Apr 2026): Native GLM-Image — inference timing.

import time
import threading

import torch


_RECORDS_LOCK = threading.Lock()
_RECORDS = {}  # patcher id -> {steps, total_s, started_at}


class TimingHandle:
    """Mutable record handed back via the model_options wrapper."""

    def __init__(self):
        self.step_times = []
        self.t0 = None
        self.t_last = None

    def reset(self):
        self.step_times = []
        self.t0 = None
        self.t_last = None

    def begin(self):
        self.t0 = time.perf_counter()
        self.t_last = self.t0

    def step(self):
        now = time.perf_counter()
        if self.t_last is None:
            self.t_last = now
            self.t0 = self.t0 or now
        self.step_times.append(now - self.t_last)
        self.t_last = now

    def report(self):
        if not self.step_times:
            return "no steps recorded"
        n = len(self.step_times)
        total = (self.t_last - self.t0) if (self.t_last and self.t0) else sum(self.step_times)
        per = sum(self.step_times) / n
        fastest = min(self.step_times)
        slowest = max(self.step_times)
        steps_per_s = (n / total) if total > 0 else 0.0
        return (
            f"GLM-Image inference timing\n"
            f"  steps         : {n}\n"
            f"  total elapsed : {total:.3f} s\n"
            f"  avg / step    : {per*1000:.1f} ms ({steps_per_s:.2f} step/s)\n"
            f"  fastest step  : {fastest*1000:.1f} ms\n"
            f"  slowest step  : {slowest*1000:.1f} ms\n"
        )


def install_timer(model_patcher, handle: TimingHandle):
    """Install a model_function_wrapper that times each forward."""
    handle.reset()
    handle.begin()

    orig_options = model_patcher.model_options
    prior_wrapper = orig_options.get("model_function_wrapper", None)

    def _wrapper(apply_model, args):
        out = apply_model(args["input"], args["timestep"], **args["c"]) \
            if prior_wrapper is None else prior_wrapper(apply_model, args)
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        handle.step()
        return out

    new_options = dict(orig_options)
    new_options["model_function_wrapper"] = _wrapper
    model_patcher.model_options = new_options
    return model_patcher
