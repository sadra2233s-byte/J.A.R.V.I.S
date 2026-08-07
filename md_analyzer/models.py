from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import pandas as pd


@dataclass
class SimulationConfig:
    atom_count: Optional[int] = None
    ensemble: Optional[str] = None
    thermostat: Optional[str] = None
    target_temp_k: Optional[float] = None
    target_pressure_bar: Optional[float] = None
    timestep_fs: Optional[float] = None
    simulation_time_ns: Optional[float] = None
    box_dimensions_angstrom: Optional[tuple] = None  # (x, y, z)
    raw_metadata: Dict[str, str] = field(default_factory=dict)

    def to_fa_dict(self) -> Dict[str, str]:
        """برای نمایش در جدول PDF"""
        d = {}
        if self.atom_count:
            d["تعداد کل اتم‌ها"] = f"{self.atom_count} اتم"
        if self.ensemble:
            d["انسامبل ترمودینامیکی"] = self.ensemble.upper()
        if self.thermostat:
            d["الگوریتم ترموستات"] = self.thermostat
        if self.target_temp_k:
            d["دمای هدف تنظیم‌شده"] = f"{self.target_temp_k} K"
        if self.target_pressure_bar:
            d["فشار هدف تنظیم‌شده"] = f"{self.target_pressure_bar} bar"
        if self.timestep_fs:
            d["گام زمانی محاسبات"] = f"{self.timestep_fs:.1f} fs"
        if self.simulation_time_ns:
            d["مدت شبیه‌سازی کانفیگ"] = f"{self.simulation_time_ns:.2f} ns"
        if self.box_dimensions_angstrom:
            x, y, z = self.box_dimensions_angstrom
            d["ابعاد باکس شبیه‌سازی"] = f"{x} × {y} × {z} Å³"
        d.update(self.raw_metadata)
        return d


@dataclass
class EnergyData:
    df: pd.DataFrame
    available_metrics: List[str]
    time_column: str = "Time"
    total_time_ns: Optional[float] = None

    def get_metric_stats(self) -> Dict[str, str]:
        stats = {}
        for col in self.available_metrics:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                mean = self.df[col].mean()
                std = self.df[col].std()
                stats[f"میانگین {col}"] = f"{mean:.2f} ± {std:.2f}"
        return stats


@dataclass
class ReportArtifact:
    pdf_path: Optional[str] = None
    html_path: Optional[str] = None
    interactive_html_path: Optional[str] = None
    plot_paths: List[str] = field(default_factory=list)
    summary_text: str = ""
    error_log: List[str] = field(default_factory=list)

    def to_agent_json(self) -> str:
        import json
        payload = {
            "status": "success" if not self.error_log else "partial_success",
            "pdf_path": self.pdf_path,
            "interactive_html_path": self.interactive_html_path,
            "plots": [p for p in self.plot_paths],
            "summary": self.summary_text,
            "errors": self.error_log,
            "instruction_to_model": (
                "گزارش PDF و نمودار تعاملی HTML ساخته شد. "
                "فقط مسیر PDF و HTML را اعلام کن؛ نیازی به جدول یا متن طولانی نیست."
            ),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)
