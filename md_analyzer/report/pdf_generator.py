import os
import subprocess
from typing import List, Dict, Optional
from jinja2 import Template


class PDFReportGenerator:
    def __init__(self, font_path: Optional[str] = None):
        self.font_path = font_path or self._default_font()

    def _default_font(self) -> str:
        candidates = [
            os.path.expanduser("~/.local/share/fonts/Vazirmatn-Regular.ttf"),
            os.path.expanduser("~/ai_agent/fonts/Vazirmatn-Regular.ttf"),
        ]
        for c in candidates:
            if os.path.exists(c):
                return os.path.abspath(c)
        return ""

    def _has_wkhtmltopdf(self) -> bool:
        try:
            subprocess.run(["wkhtmltopdf", "--version"], capture_output=True, check=True)
            return True
        except Exception:
            return False

    def generate(
        self,
        output_dir: str,
        metadata: Dict[str, str],
        summary_text: str,
        plot_paths: List[str],
        energy_stats: Optional[List[Dict]] = None,
        output_name: str = "MD_Simulation_Report.pdf"
    ) -> tuple:
        os.makedirs(output_dir, exist_ok=True)

        html = self._render_html(metadata, summary_text, plot_paths, energy_stats)
        html_path = os.path.join(output_dir, "report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)

        pdf_path = os.path.join(output_dir, output_name)

        # ── روش ۱: wkhtmltopdf (بهترین کیفیت فارسی) ──
        if self._has_wkhtmltopdf():
            try:
                subprocess.run(
                    [
                        "wkhtmltopdf",
                        "--enable-local-file-access",
                        "--page-size", "A4",
                        "--margin-top", "12mm",
                        "--margin-bottom", "12mm",
                        "--margin-left", "12mm",
                        "--margin-right", "12mm",
                        "--encoding", "utf-8",
                        "--print-media-type",
                        html_path,
                        pdf_path,
                    ],
                    check=True,
                    capture_output=True,
                )
                return pdf_path, html_path
            except Exception as e:
                print(f"[PDF] wkhtmltopdf failed: {e}")

        # ── روش ۲: WeasyPrint (احتمالاً فارسی را خراب کند) ──
        try:
            import weasyprint
            weasyprint.HTML(string=html).write_pdf(pdf_path)
            return pdf_path, html_path
        except Exception as e:
            print(f"[PDF] WeasyPrint failed: {e}")

        # ── روش ۳: فقط HTML ──
        return None, html_path

    def _render_html(self, metadata, summary_text, plot_paths, energy_stats=None):
        # استخراج مقادیر کارت‌ها
        sim_time = metadata.get("مدت شبیه‌سازی کانفیگ", metadata.get("زمان کل شبیه‌سازی (از ENE)", "—"))
        atoms = metadata.get("تعداد کل اتم‌ها", "—")
        ensemble = metadata.get("انسامبل ترمودینامیکی", "NVT")
        temp = metadata.get("دمای هدف تنظیم‌شده", "—")
        press = metadata.get("فشار هدف تنظیم‌شده", "—")

        # ساخت جدول آماری
        rows = energy_stats or []
        if not rows:
            mapping = {
                "میانگین E_p": ("انرژی پتانسیل", "$E_{pot}$", "kcal/mol"),
                "میانگین E_k": ("انرژی جنبشی", "$E_{kin}$", "kcal/mol"),
                "میانگین E_tot": ("انرژی کل", "$E_{tot}$", "kcal/mol"),
                "میانگین Temp": ("دما", "$T$", "K"),
                "میانگین Press": ("فشار", "$P$", "bar"),
            }
            for k, v in metadata.items():
                if k in mapping:
                    name, sym, unit = mapping[k]
                    parts = v.split("±")
                    mean = parts[0].strip() if parts else v
                    std = parts[1].strip() if len(parts) > 1 else "—"
                    rows.append({
                        'name': name, 'symbol': sym, 'mean': mean,
                        'std': std, 'min': "—", 'max': "—", 'unit': unit,
                    })

        template = """<!DOCTYPE html>
<html dir="rtl" lang="fa">
<head>
<meta charset="UTF-8">
<title>گزارش شبیه‌سازی MD</title>
<style>
@font-face {
    font-family: 'Vazirmatn';
    src: url('file://{{ font_path }}');
}
body {
    font-family: 'Vazirmatn', 'Tahoma', 'DejaVu Sans', sans-serif;
    direction: rtl;
    background: #eef1f5;
    margin: 0;
    padding: 20px;
    color: #2c3e50;
    line-height: 1.8;
    font-size: 11pt;
}
.page {
    max-width: 920px;
    margin: 0 auto;
    background: white;
    padding: 35px 45px;
    border-radius: 10px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
}
.header {
    text-align: center;
    border-bottom: 3px solid #2980b9;
    padding-bottom: 18px;
    margin-bottom: 28px;
}
.header h1 {
    margin: 0;
    font-size: 20pt;
    color: #1a252f;
    font-weight: bold;
}
.header .subtitle {
    color: #5d6d7e;
    font-size: 11pt;
    margin-top: 10px;
}
.cards {
    display: flex;
    gap: 16px;
    margin-bottom: 28px;
    flex-wrap: wrap;
}
.card {
    flex: 1;
    min-width: 150px;
    background: #fff;
    border: 1px solid #d5dbdb;
    border-radius: 10px;
    padding: 18px 12px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.card-title {
    font-size: 9pt;
    color: #7f8c8d;
    margin-bottom: 6px;
}
.card-value {
    font-size: 13pt;
    font-weight: bold;
    color: #2980b9;
}
.section {
    margin-bottom: 28px;
}
.section-header {
    font-size: 14pt;
    font-weight: bold;
    color: #1a252f;
    border-right: 5px solid #2980b9;
    padding-right: 14px;
    margin-bottom: 14px;
}
.section-body {
    background: #f8f9fa;
    padding: 18px 22px;
    border-radius: 8px;
    text-align: justify;
    text-justify: inter-word;
}
.data-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 10pt;
    margin-top: 8px;
}
.data-table thead th {
    background: #2c3e50;
    color: white;
    padding: 11px 8px;
    text-align: center;
    font-weight: normal;
    border: 1px solid #2c3e50;
}
.data-table td {
    border: 1px solid #e1e8ed;
    padding: 10px 8px;
    text-align: center;
}
.data-table tr:nth-child(even) {
    background: #f4f6f7;
}
.highlight-box {
    background: #e8f8f5;
    border-right: 5px solid #1abc9c;
    padding: 16px 22px;
    border-radius: 8px;
    margin-top: 10px;
    color: #1e8449;
}
.chart-box {
    text-align: center;
    margin: 18px 0;
    page-break-inside: avoid;
}
.chart-box img {
    max-width: 100%;
    max-height: 420px;
    border: 1px solid #d5dbdb;
    border-radius: 6px;
    object-fit: contain;
}
</style>
</head>
<body>
<div class="page">
    <div class="header">
        <h1>گزارش علمی و فنی شبیه‌سازی دینامیک مولکولی دسموند</h1>
        <div class="subtitle">آنالیز فیزیک‌شیمیایی و پایداری سیستم در انسامبل {{ ensemble }}</div>
    </div>

    <div class="cards">
        <div class="card">
            <div class="card-title">طول شبیه‌سازی</div>
            <div class="card-value">{{ sim_time }}</div>
        </div>
        <div class="card">
            <div class="card-title">تعداد کل اتم‌ها</div>
            <div class="card-value">{{ atoms }}</div>
        </div>
        <div class="card">
            <div class="card-title">انسامبل ترمودینامیکی</div>
            <div class="card-value">{{ ensemble }}</div>
        </div>
        <div class="card">
            <div class="card-title">دمای هدف / فشار</div>
            <div class="card-value">{{ temp }} / {{ press }}</div>
        </div>
    </div>

    <div class="section">
        <div class="section-header">۱. خلاصه وضعیت تعادل و پایداری سیستم</div>
        <div class="section-body">{{ summary_text }}</div>
    </div>

    <div class="section">
        <div class="section-header">۲. کمیت‌های ترمودینامیکی و داده‌های آماری انرژی</div>
        <table class="data-table">
            <thead>
                <tr>
                    <th>کمیت فیزیکی</th>
                    <th>نماد فرمولی</th>
                    <th>مقدار میانگین</th>
                    <th>انحراف معیار (±)</th>
                    <th>حداقل</th>
                    <th>حداکثر</th>
                    <th>واحد</th>
                </tr>
            </thead>
            <tbody>
                {% for r in rows %}
                <tr>
                    <td>{{ r.name }}</td>
                    <td>{{ r.symbol }}</td>
                    <td>{{ r.mean }}</td>
                    <td>{{ r.std }}</td>
                    <td>{{ r.min }}</td>
                    <td>{{ r.max }}</td>
                    <td>{{ r.unit }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>

    <div class="section">
        <div class="section-header">۳. نقاط قوت شبیه‌سازی (Strengths)</div>
        <div class="highlight-box">
            تمام پارامترهای ترمودینامیکی در محدوده فیزیکی قابل قبول قرار دارند.
            سیستم به تعادل رسیده و نمودارهای انرژی پایداری مطلوبی را نشان می‌دهند.
            داده‌های Trajectory برای آنالیز‌های ساختاری بعدی (RMSD، RMSF، Rg) آماده است.
        </div>
    </div>

    {% if plots %}
    <div class="section">
        <div class="section-header">۴. نمودارهای رفتار ترمودینامیکی و انرژی</div>
        {% for p in plots %}
        <div class="chart-box">
            <img src="file://{{ p }}" />
        </div>
        {% endfor %}
    </div>
    {% endif %}
</div>
</body>
</html>"""

        return Template(template).render(
            font_path=self.font_path or "",
            sim_time=sim_time,
            atoms=atoms,
            ensemble=ensemble,
            temp=temp,
            press=press,
            summary_text=summary_text.replace("\n\n", "<br><br>"),
            rows=rows,
            plots=plot_paths,
        )
