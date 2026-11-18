"""Run recording: one folder per run holding every log, trace and measurement.

runs/<run_id>/
    manifest.json        config, environment, model revisions, input file hashes, start/end times
    run.log              human-readable log
    events.jsonl         one structured record per pipeline stage (time, energy, memory, metadata)
    traces/              one JSON per query with prompt, retrieved chunks, tokens, timings, energy
    codecarbon/          raw CodeCarbon output (one tracker per measured block)
    ...                  CSV tables written by run_pipeline.py
"""
import contextlib
import datetime as dt
import hashlib
import json
import logging
import os
import platform
import socket
import subprocess
import sys
import time

import torch


def now_iso():
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="milliseconds")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _json_default(o):
    if hasattr(o, "item"):
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default, ensure_ascii=False)


def resolve_device(name):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


class GpuEnergyMeter:
    """NVIDIA board energy counter (millijoules) through NVML. Inactive on non-NVIDIA hardware."""

    def __init__(self):
        self.handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
        except Exception:  # noqa: BLE001
            self.handle = None

    @property
    def available(self):
        return self.handle is not None

    def read_mj(self):
        return self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle) if self.handle else None

    def power_w(self):
        return self.nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000 if self.handle else None


class CarbonMeter:
    """A fresh CodeCarbon tracker per measured block; results are returned as a flat dict."""

    def __init__(self, output_dir, measure_power_secs=1.0, country_iso_code=""):
        self.output_dir = output_dir
        self.measure_power_secs = measure_power_secs
        self.country_iso_code = country_iso_code
        os.makedirs(output_dir, exist_ok=True)
        logging.getLogger("codecarbon").setLevel(logging.ERROR)

    @contextlib.contextmanager
    def track(self, project_name):
        from codecarbon import EmissionsTracker, OfflineEmissionsTracker
        kwargs = dict(project_name=project_name, output_dir=self.output_dir, save_to_file=True,
                      measure_power_secs=self.measure_power_secs, tracking_mode="process", log_level="error")
        if self.country_iso_code:
            tracker = OfflineEmissionsTracker(country_iso_code=self.country_iso_code, **kwargs)
        else:
            tracker = EmissionsTracker(**kwargs)
        result = {}
        tracker.start()
        try:
            yield result
        finally:
            emissions = tracker.stop()
            d = tracker.final_emissions_data
            result.update({
                "co2_kg": emissions, "energy_kwh": d.energy_consumed, "cpu_energy_kwh": d.cpu_energy,
                "gpu_energy_kwh": d.gpu_energy, "ram_energy_kwh": d.ram_energy, "cc_duration_s": d.duration,
                "cpu_power_w": d.cpu_power, "gpu_power_w": d.gpu_power, "ram_power_w": d.ram_power,
                "country": d.country_name, "region": d.region,
                "carbon_intensity_g_per_kwh": (emissions / d.energy_consumed * 1000) if d.energy_consumed else None,
            })


class RunRecorder:
    """Owns the run folder, the logger, the event stream and the energy meters."""

    def __init__(self, root, tag, device):
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_id = f"{stamp}_{tag}" if tag else stamp
        self.dir = os.path.join(root, self.run_id)
        os.makedirs(os.path.join(self.dir, "traces"), exist_ok=True)
        self.device = device
        self.gpu = GpuEnergyMeter()
        self._events = open(os.path.join(self.dir, "events.jsonl"), "a")
        self._t0 = time.perf_counter()
        self.log = logging.getLogger("rag")
        self.log.setLevel(logging.INFO)
        self.log.handlers.clear()
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        for h in (logging.FileHandler(os.path.join(self.dir, "run.log")), logging.StreamHandler(sys.stdout)):
            h.setFormatter(fmt)
            self.log.addHandler(h)

    def path(self, *parts):
        p = os.path.join(self.dir, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    def event(self, kind, **data):
        rec = {"ts": now_iso(), "t_rel_s": round(time.perf_counter() - self._t0, 4), "event": kind, **data}
        self._events.write(json.dumps(rec, default=_json_default, ensure_ascii=False) + "\n")
        self._events.flush()

    @contextlib.contextmanager
    def stage(self, name, **meta):
        """Time a pipeline stage and record its GPU energy and peak memory in events.jsonl."""
        info = {}
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        e0 = self.gpu.read_mj()
        sync(self.device)
        start = time.perf_counter()
        self.event("stage_start", stage=name, **meta)
        try:
            yield info
        finally:
            sync(self.device)
            elapsed = time.perf_counter() - start
            e1 = self.gpu.read_mj()
            info["duration_s"] = elapsed
            if e0 is not None:
                info["gpu_energy_j"] = (e1 - e0) / 1000
            if self.device == "cuda":
                info["gpu_peak_mem_mb"] = torch.cuda.max_memory_allocated() / 2**20
            self.event("stage_end", stage=name, **meta, **info)

    def close(self):
        self._events.close()
        for h in self.log.handlers:
            h.close()


def _version(pkg):
    try:
        from importlib.metadata import version
        return version(pkg)
    except Exception:  # noqa: BLE001
        return None


def capture_environment(device, gpu_meter):
    env = {
        "hostname": socket.gethostname(), "platform": platform.platform(), "python": sys.version.split()[0],
        "device": device, "cpu": platform.processor() or platform.machine(), "cpu_count": os.cpu_count(),
        "packages": {p: _version(p) for p in ["torch", "transformers", "sentence-transformers", "faiss-cpu",
                                              "faiss-gpu", "codecarbon", "pypdf", "accelerate", "nvidia-ml-py"]},
    }
    try:
        import psutil
        env["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except Exception:  # noqa: BLE001
        pass
    if platform.system() == "Darwin":
        env["cpu"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                    text=True).stdout.strip() or env["cpu"]
    elif os.path.exists("/proc/cpuinfo"):
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                env["cpu"] = line.split(":", 1)[1].strip()
                break
    if device == "cuda":
        env["gpu"] = torch.cuda.get_device_name(0)
        env["cuda"] = torch.version.cuda
        env["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
        if gpu_meter.available:
            env["nvidia_driver"] = gpu_meter.nvml.nvmlSystemGetDriverVersion()
    return env


def model_revision(repo_id):
    """Commit hash of the locally cached model snapshot (pins exactly which weights were used)."""
    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(repo_id, local_files_only=True)
        return os.path.basename(path)
    except Exception:  # noqa: BLE001
        return None
