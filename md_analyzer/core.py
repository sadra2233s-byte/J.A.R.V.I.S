import os
from typing import List, Optional
from md_analyzer.models import SimulationConfig, EnergyData, ReportArtifact
from md_analyzer.parsers.desmond_parser import DesmondConfigParser
from md_analyzer.parsers.energy_parser import EnergyFileParser
from md_analyzer.plotters.matplotlib_plotter import MatplotlibPlotter
from md_analyzer.plotters.plotly_plotter import PlotlyPlotter
from md_analyzer.report.nlg import PersianNLG
from md_analyzer.report.pdf_generator import PDFReportGenerator


class DesmondProAnalyzer:
    def __init__(self, target_dir: str, output_dir: Optional[str] = None):
        self.target_dir = os.path.expanduser(target_dir)
        self.output_dir = output_dir or self.target_dir
        self.config_parser = DesmondConfigParser()
        self.energy_parser = EnergyFileParser()
        self.matplotlib_plotter = MatplotlibPlotter(self.output_dir)
        self.plotly_plotter = PlotlyPlotter(self.output_dir)
        self.nlg = PersianNLG()
        self.pdf_gen = PDFReportGenerator()
        self.artifact = ReportArtifact()
        self.config = SimulationConfig()
        self.energy = None

    def scan_and_parse(self):
        if not os.path.exists(self.target_dir):
            self.artifact.error_log.append(f"پوشه یافت نشد: {self.target_dir}")
            return self.artifact

        files = os.listdir(self.target_dir)
        ext_counts = {}
        for f in files:
            if os.path.isfile(os.path.join(self.target_dir, f)):
                ext = os.path.splitext(f)[1].lower()
                ext_counts[ext] = ext_counts.get(ext, 0) + 1

        self.artifact.raw_metadata = {
            "محتویات پوشه": ", ".join([f"{v} فایل {k}" for k, v in ext_counts.items()])
        }

        config = SimulationConfig()
        for f in files:
            if f.endswith(('.log', '.multisim.log', '.cfg')):
                cfg = self.config_parser.parse(os.path.join(self.target_dir, f))
                if not config.atom_count and cfg.atom_count:
                    config.atom_count = cfg.atom_count
                if not config.ensemble and cfg.ensemble:
                    config.ensemble = cfg.ensemble
                if not config.thermostat and cfg.thermostat:
                    config.thermostat = cfg.thermostat
                if not config.target_temp_k and cfg.target_temp_k:
                    config.target_temp_k = cfg.target_temp_k
                if not config.target_pressure_bar and cfg.target_pressure_bar:
                    config.target_pressure_bar = cfg.target_pressure_bar
                if not config.timestep_fs and cfg.timestep_fs:
                    config.timestep_fs = cfg.timestep_fs
                if not config.simulation_time_ns and cfg.simulation_time_ns:
                    config.simulation_time_ns = cfg.simulation_time_ns
                if not config.box_dimensions_angstrom and cfg.box_dimensions_angstrom:
                    config.box_dimensions_angstrom = cfg.box_dimensions_angstrom
                config.raw_metadata.update(cfg.raw_metadata)

        energy = None
        ene_files = [f for f in files if f.endswith('.ene')]
        if ene_files:
            energy = self.energy_parser.parse(os.path.join(self.target_dir, ene_files[0]))
            if energy:
                config.raw_metadata.update(energy.get_metric_stats())
                if energy.total_time_ns:
                    config.raw_metadata["زمان کل شبیه‌سازی (از ENE)"] = f"{energy.total_time_ns:.2f} ns"

        self.config = config
        self.energy = energy
        return self.artifact

    def generate_charts(self, metrics: List[str], combine: bool = True):
        if not self.energy:
            self.artifact.error_log.append("داده انرژی موجود نیست.")
            return

        if combine:
            p = self.matplotlib_plotter.plot_combined(self.energy, metrics)
            if p:
                self.artifact.plot_paths.append(p)

        singles = self.matplotlib_plotter.plot_individual(self.energy, metrics)
        self.artifact.plot_paths.extend(singles)

    def generate_interactive(self, metrics: List[str], filename: str = "interactive_chart.html"):
        if not self.energy:
            return
        p = self.plotly_plotter.generate_html(self.energy, metrics, filename)
        self.artifact.interactive_html_path = p

    def generate_pdf(self, output_name: str = "MD_Simulation_Report.pdf"):
        text = self.nlg.generate(self.config, self.energy)
        self.artifact.summary_text = text
        meta = self.config.to_fa_dict()
        meta.update(self.artifact.raw_metadata)

        # آماده‌سازی آمار کامل برای جدول
        energy_stats = []
        if self.energy and self.energy.df is not None:
            df = self.energy.df
            info = {
                'E_p': ('انرژی پتانسیل', '$E_{pot}$', 'kcal/mol'),
                'E_k': ('انرژی جنبشی', '$E_{kin}$', 'kcal/mol'),
                'E_tot': ('انرژی کل', '$E_{tot}$', 'kcal/mol'),
                'Temp': ('دما', '$T$', 'K'),
                'Press': ('فشار', '$P$', 'bar'),
            }
            for col in self.energy.available_metrics:
                if col not in df.columns:
                    continue
                vals = df[col].dropna()
                if len(vals) == 0:
                    continue
                name, symbol, unit = info.get(col, (col, col, ''))
                energy_stats.append({
                    'name': name,
                    'symbol': symbol,
                    'mean': f"{vals.mean():.2f}",
                    'std': f"{vals.std():.2f}",
                    'min': f"{vals.min():.2f}",
                    'max': f"{vals.max():.2f}",
                    'unit': unit,
                })

        pdf, html = self.pdf_gen.generate(
            self.output_dir, meta, text, self.artifact.plot_paths,
            energy_stats=energy_stats, output_name=output_name
        )
        self.artifact.pdf_path = pdf
        self.artifact.html_path = html
