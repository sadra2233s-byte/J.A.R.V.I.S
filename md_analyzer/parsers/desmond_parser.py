import re
import os
from typing import Optional
from md_analyzer.models import SimulationConfig


class DesmondConfigParser:
    """فقط می‌خواند و پارس می‌کند. هیچ‌چیز نمی‌سازد."""

    def parse(self, filepath: str) -> SimulationConfig:
        cfg = SimulationConfig()
        if not os.path.exists(filepath):
            return cfg

        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception as e:
            cfg.raw_metadata["خطای خواندن فایل"] = str(e)
            return cfg

        # تعداد اتم‌ها
        if m := re.search(r'#\s*N\s+atoms\s*=\s*(\d+)', content, re.I):
            cfg.atom_count = int(m.group(1))

        # ensemble
        if m := re.search(r'ensemble\s*=\s*\{[^}]*class\s*=\s*"(\w+)"', content, re.S):
            cfg.ensemble = m.group(1)
        elif m := re.search(r'class\s*=\s*"(NVT|NPT|NPAT|NPH)"', content, re.I):
            cfg.ensemble = m.group(1)

        # thermostat method
        if m := re.search(r'method\s*=\s*"(Nose-Hoover|NH|Langevin|Berendsen)"', content, re.I):
            cfg.thermostat = m.group(1)

        # temperature
        if m := re.search(r'temperature\s*=\s*\[\[\s*"([\d.]+)"', content, re.I):
            cfg.target_temp_k = float(m.group(1))

        # pressure
        if m := re.search(r'pressure\s*=\s*"([\d.]+)"', content, re.I):
            cfg.target_pressure_bar = float(m.group(1))

        # timestep
        if m := re.search(r'timestep\s*=\s*\[\s*"([\d.]+)"', content, re.I):
            cfg.timestep_fs = float(m.group(1)) * 1000.0

        # simulation time
        if m := re.search(r'time\s*=\s*"([\d.]+)"', content, re.I):
            cfg.simulation_time_ns = float(m.group(1)) / 1000.0

        # box dimensions (3x3 matrix flattened)
        box_pat = r'box\s*=\s*\[\s*"([\d.]+)"\s*"[\d.]+"\s*"[\d.]+"\s*"[\d.]+"\s*"([\d.]+)"\s*"[\d.]+"\s*"[\d.]+"\s*"[\d.]+"\s*"([\d.]+)"'
        if m := re.search(box_pat, content, re.I):
            cfg.box_dimensions_angstrom = (m.group(1), m.group(2), m.group(3))

        return cfg
