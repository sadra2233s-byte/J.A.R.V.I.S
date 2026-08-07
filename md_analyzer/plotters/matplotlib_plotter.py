import os
from typing import List, Optional
import matplotlib
matplotlib.use('Agg')  # سرور بدون GUI
import matplotlib.pyplot as plt
import seaborn as sns
from md_analyzer.models import EnergyData

sns.set_theme(style="whitegrid")


class MatplotlibPlotter:
    def __init__(self, output_dir: str, dpi: int = 300):
        self.output_dir = output_dir
        self.dpi = dpi
        os.makedirs(output_dir, exist_ok=True)

    def _match_column(self, metric: str, columns: List[str]) -> Optional[str]:
        m = metric.lower()
        for col in columns:
            c = col.lower()
            if m in c or c in m:
                return col
        # fuzzy aliases
        aliases = {
            "e_p": ["e_p", "potential", "pot"],
            "e_k": ["e_k", "kinetic", "kin"],
            "e_tot": ["e_tot", "total", "tot"],
        }
        for key, variants in aliases.items():
            if any(v in m for v in variants):
                for col in columns:
                    if any(v in col.lower() for v in variants):
                        return col
        return None

    def plot_combined(self, data: EnergyData, metrics: List[str]) -> Optional[str]:
        df = data.df.dropna(how='all')
        time = df[data.time_column] / 1000.0 if data.time_column == "Time" else df.index

        fig, ax = plt.subplots(figsize=(9, 4.5), dpi=self.dpi)
        colors = ['#1f77b4', '#e74c3c', '#2ca02c', '#9467bd', '#ff7f0e']
        added = 0

        for idx, metric in enumerate(metrics):
            col = self._match_column(metric, data.available_metrics)
            if not col:
                continue
            c = colors[idx % len(colors)]
            ax.plot(time, df[col], color=c, alpha=0.3, linewidth=0.8)
            smoothed = df[col].rolling(window=10, min_periods=1).mean()
            ax.plot(time, smoothed, color=c, linewidth=2.0, label=col)
            added += 1

        if added == 0:
            plt.close(fig)
            return None

        ax.set_title("Combined Energy Metrics", fontsize=12, fontweight='bold')
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Energy (kcal/mol)")
        ax.legend(loc='best')
        plt.tight_layout()

        path = os.path.join(self.output_dir, "combined_metrics_chart.png")
        fig.savefig(path)
        plt.close(fig)
        return path

    def plot_individual(self, data: EnergyData, metrics: List[str]) -> List[str]:
        paths = []
        df = data.df.dropna(how='all')
        time = df[data.time_column] / 1000.0 if data.time_column == "Time" else df.index

        for metric in metrics:
            col = self._match_column(metric, data.available_metrics)
            if not col:
                continue

            fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=self.dpi)
            ax.plot(time, df[col], color='#2b5c8f', alpha=0.4, linewidth=0.8)
            smoothed = df[col].rolling(window=10, min_periods=1).mean()
            ax.plot(time, smoothed, color='#e74c3c', linewidth=2.0, label='Moving Avg')
            ax.set_title(f"Trajectory: {col}", fontsize=12, fontweight='bold')
            ax.set_xlabel("Time (ns)")
            ax.set_ylabel(col)
            ax.legend()
            plt.tight_layout()

            path = os.path.join(self.output_dir, f"chart_{col}.png")
            fig.savefig(path)
            plt.close(fig)
            paths.append(path)

        return paths
