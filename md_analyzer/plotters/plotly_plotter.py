import os
from typing import List, Optional
import plotly.graph_objects as go
from md_analyzer.models import EnergyData


class PlotlyPlotter:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _match_column(self, metric: str, columns: List[str]) -> Optional[str]:
        m = metric.lower()
        for col in columns:
            c = col.lower()
            if m in c or c in m:
                return col
        return None

    def generate_html(self, data: EnergyData, metrics: List[str], filename: str = "interactive_chart.html") -> Optional[str]:
        df = data.df.dropna(how='all')
        time = df[data.time_column] / 1000.0 if data.time_column == "Time" else df.index

        fig = go.Figure()
        count = 0
        for metric in metrics:
            col = self._match_column(metric, data.available_metrics)
            if not col:
                continue
            fig.add_trace(go.Scatter(
                x=time, y=df[col], mode='lines', name=col,
                hovertemplate=f"{col}: %{{y:.2f}}<br>Time: %{{x:.2f}} ns"
            ))
            count += 1

        if count == 0:
            for col in data.available_metrics[:3]:
                fig.add_trace(go.Scatter(x=time, y=df[col], mode='lines', name=col))

        fig.update_layout(
            title="Interactive MD Energy Analysis",
            xaxis_title="Time (ns)",
            yaxis_title="Energy (kcal/mol)",
            template="plotly_white",
            hovermode="x unified",
        )

        path = os.path.join(self.output_dir, filename)
        fig.write_html(path)
        return path
