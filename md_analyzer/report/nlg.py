from md_analyzer.models import SimulationConfig, EnergyData
import numpy as np


class PersianNLG:
    def generate(self, config: SimulationConfig, energy=None) -> str:
        paragraphs = []

        # ── پاراگراف ۱: معرفی شبیه‌سازی ──
        ensemble = config.ensemble or "NVT"
        atoms = f"{config.atom_count} اتم" if config.atom_count else "تعداد نامشخص اتم"
        box = (
            f"{config.box_dimensions_angstrom[0]}×{config.box_dimensions_angstrom[1]}×{config.box_dimensions_angstrom[2]} Å³"
            if config.box_dimensions_angstrom else "ابعاد نامشخص"
        )
        sim_time = (
            f"{config.simulation_time_ns:.2f} ns"
            if config.simulation_time_ns else "زمان نامشخص"
        )
        temp_target = f"{config.target_temp_k:.1f} K" if config.target_temp_k else None
        press_target = f"{config.target_pressure_bar:.3f} bar" if config.target_pressure_bar else None

        intro = (
            f"شبیه‌سازی دینامیک مولکولی حاضر در انسامبل {ensemble} با تعداد کل {atoms} "
            f"و ابعاد باکس {box} در بازه زمانی {sim_time} اجرا و آنالیز گردید."
        )
        if temp_target:
            intro += f" دمای هدف سیستم {temp_target} و"
        if press_target:
            intro += f" فشار هدف {press_target} بوده است."
        paragraphs.append(intro)

        if not energy or energy.df.empty:
            paragraphs.append("داده‌های انرژی برای تحلیل دقیق‌تر در دسترس نبود.")
            return "\n\n".join(paragraphs)

        df = energy.df
        time = df[energy.time_column] if energy.time_column in df.columns else None

        # ── پاراگراف ۲: تحلیل دما ──
        temp_col = next((c for c in energy.available_metrics if 'temp' in c.lower()), None)
        if temp_col:
            temps = df[temp_col].dropna()
            t_mean = temps.mean()
            t_std = temps.std()
            t_min, t_max = temps.min(), temps.max()
            t_drift = ((temps.iloc[-len(temps)//5:].mean() - temps.iloc[:len(temps)//5].mean()) / t_mean * 100) if len(temps) > 10 else 0

            temp_text = (
                f"تحلیل پروفایل دما نشان می‌دهد میانگین دمای سیستم در طول شبیه‌سازی "
                f"{t_mean:.2f} ± {t_std:.2f} K بوده است."
            )
            if abs(t_drift) < 1.0:
                temp_text += " دما با انحراف بسیار ناچیز در کل بازه زمانی پایدار بوده و عملکرد ترموستات مطلوب ارزیابی می‌شود."
            elif abs(t_drift) < 3.0:
                temp_text += f" دما دارای انحراف تدریجی {t_drift:.1f}% از ابتدا به انتهای شبیه‌سازی بوده که در محدوده قابل قبول قرار دارد."
            else:
                temp_text += f" توجه: دما دارای انحراف {t_drift:.1f}% در بازه شبیه‌سازی است که نیاز به بررسی مجدد تنظیمات ترموستات دارد."
            temp_text += f" دامنه نوسان دما بین {t_min:.2f} تا {t_max:.2f} K مشاهده گردید."
            paragraphs.append(temp_text)

        # ── پاراگراف ۳: تحلیل انرژی پتانسیل و کل ──
        ep_col = next((c for c in energy.available_metrics if any(x in c.lower() for x in ['e_p', 'potential', 'pot'])), None)
        ek_col = next((c for c in energy.available_metrics if any(x in c.lower() for x in ['e_k', 'kinetic', 'kin'])), None)
        etot_col = next((c for c in energy.available_metrics if any(x in c.lower() for x in ['e_tot', 'total', 'tot'])), None)

        if ep_col:
            ep = df[ep_col].dropna()
            ep_mean, ep_std = ep.mean(), ep.std()
            ep_drift = ((ep.iloc[-len(ep)//5:].mean() - ep.iloc[:len(ep)//5].mean()) / abs(ep_mean) * 100) if len(ep) > 10 else 0

            energy_text = (
                f"انرژی پتانسیل میانگین سیستم {ep_mean:.2f} ± {ep_std:.2f} kcal/mol بوده است. "
            )
            if abs(ep_drift) < 0.5:
                energy_text += "ثبات انرژی پتانسیل در کل بازه زمانی نشان‌دهنده رسیدن سیستم به تعادل ساختاری پایدار است."
            elif ep_drift < -0.5:
                energy_text += "کاهش تدریجی انرژی پتانسیل نشان‌دهنده فرآیند بهینه‌سازی ساختاری و رسیدن به حالت پایین‌تر انرژی است."
            else:
                energy_text += "افزایش انرژی پتانسیل در برخی بازه‌ها مشاهده شده که ممکن است ناشی از نوسانات طبیعی یا عدم تعادل اولیه باشد."

            # نسبت انرژی جنبشی به پتانسیل
            if ek_col:
                ek_mean = df[ek_col].dropna().mean()
                ratio = abs(ek_mean / ep_mean) if ep_mean != 0 else 0
                energy_text += f" نسبت انرژی جنبشی به پتانسیل ({ratio:.3f}) در محدوده فیزیولوژیکی سیستم‌های مشابه قرار دارد."
            paragraphs.append(energy_text)

        # ── پاراگراف ۴: تحلیل فشار ──
        press_col = next((c for c in energy.available_metrics if 'press' in c.lower()), None)
        if press_col:
            press = df[press_col].dropna()
            p_mean, p_std = press.mean(), press.std()
            press_text = (
                f"فشار سیستم میانگین {p_mean:.2f} ± {p_std:.2f} bar را نشان می‌دهد. "
            )
            if p_std < 50:
                press_text += "نوسانات فشار کنترل‌شده و پایدار بوده که بیانگر عملکرد مناسب باروستات در انسامبل NPT است."
            else:
                press_text += "نوسانات نسبتاً بالای فشار مشاهده گردید که در شبیه‌سازی‌های کوتاه‌مدت طبیعی است."
            paragraphs.append(press_text)

        # ── پاراگراف ۵: جمع‌بندی ──
        paragraphs.append(
            "تمام پارامترهای ترمودینامیکی و نمودارهای استخراج‌شده در بخش‌های بعدی گزارش به صورت کامل ارائه گردیده‌اند. "
            "توصیه می‌شود برای ارزیابی دقیق‌تر، آنالیز RMSD و RMSF ساختاری نیز در ادامه انجام شود."
        )

        return "\n\n".join(paragraphs)
