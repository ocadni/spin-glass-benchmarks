"""Machine specifications collection."""

from __future__ import annotations

import platform
import socket

import numpy as np
import psutil
import torch

from experiments.src.results import MachineSpecs

try:
    import cpuinfo

    HAS_CPUINFO = True
except ImportError:
    HAS_CPUINFO = False


class MachineSpecsCollector:
    """Collect comprehensive machine specifications for reproducibility."""

    @staticmethod
    def collect() -> MachineSpecs:
        """Collect comprehensive machine specifications."""
        cpu_info = MachineSpecsCollector._collect_cpu_info()
        gpu_info = MachineSpecsCollector._collect_gpu_info()

        return MachineSpecs(
            hostname=socket.gethostname(),
            cpu=cpu_info,
            memory_gb=round(psutil.virtual_memory().total / (1024**3), 2),
            gpu=gpu_info,
            os={
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
            },
            python=platform.python_version(),
            torch=torch.__version__,
            numpy=np.__version__,
            cuda_version=torch.version.cuda if torch.cuda.is_available() else None,
        )

    @staticmethod
    def _collect_cpu_info() -> dict:
        """Collect CPU information."""
        cpu_info = {
            "physical_cores": psutil.cpu_count(logical=False),
            "total_cores": psutil.cpu_count(logical=True),
        }

        if HAS_CPUINFO:
            cpu_data = cpuinfo.get_cpu_info()
            cpu_info["model"] = cpu_data.get("brand_raw", "Unknown")
        else:
            cpu_info["model"] = platform.processor() or "Unknown"

        cpu_freq = psutil.cpu_freq()
        if cpu_freq:
            cpu_info["frequency_mhz"] = cpu_freq.current
        else:
            cpu_info["frequency_mhz"] = None

        return cpu_info

    @staticmethod
    def _collect_gpu_info() -> dict:
        """Collect GPU information."""
        if not torch.cuda.is_available():
            return {"available": False}

        props = torch.cuda.get_device_properties(0)
        capability = torch.cuda.get_device_capability(0)

        return {
            "available": True,
            "name": torch.cuda.get_device_name(0),
            "compute_capability": f"{capability[0]}.{capability[1]}",
            "memory_gb": round(props.total_memory / (1024**3), 2),
        }
