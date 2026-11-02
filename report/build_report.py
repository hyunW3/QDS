"""report_template.html + figs/*.png -> report.html (single file).

Figures are embedded as base64. A Noto Sans CJK subset (only the characters the page uses) is embedded as
@font-face so Korean renders even on machines without Korean fonts. Needs fontTools (ships with matplotlib):
run with the lingua venv python.
"""
import base64, io, re
from pathlib import Path
from fontTools import subset
from fontTools.ttLib import TTFont

D = Path(__file__).resolve().parents[1]
html = (D / "report/report_template.html").read_text()
FIGS = {"fig1": "fig1_sqnr_rht", "fig2": "fig2_alpha_per_layer", "fig3": "fig3_db_budget",
        "fig4": "fig4_level_bias", "fig5": "fig5_across_models", "fig6": "fig6_synthetic_outliers",
        "fig7": "fig7_int4_vs_e1m2", "fig8": "fig8_cause_distribution", "fig9": "fig9_gap_to_optimal", "fig10": "fig10_rotation_sweep", "fig11": "fig11_best_configs", "fig12": "fig12_flip_grids", "fig13": "fig13_flip_results", "fig14": "fig14_flip_vs_large", "fig15": "fig15_stage1", "fig16": "fig16_ptq_quality", "fig18": "fig18_latency", "fig20": "fig20_ptq8b", "fig21": "fig21_anatomy", "fig22": "fig22_sensitivity", "fig23": "fig23_level_placement", "fig24": "fig24_dfp_decomp", "fig25": "fig25_rotation_gain"}

# --- font subset from the visible text (before images are inlined)
text = re.sub(r"<[^>]+>", " ", html)
chars = "".join(sorted(set(text) | set("0123456789")))
font = TTFont(str(next((D / "fonts").glob("*.[ot]tf"))))
opts = subset.Options(); opts.flavor = "woff"; opts.layout_features = ["*"]; opts.name_IDs = ["*"]
sub = subset.Subsetter(opts); sub.populate(text=chars); sub.subset(font)
buf = io.BytesIO(); font.flavor = "woff"; font.save(buf)
face = ("@font-face{font-family:'ReportKR';font-weight:400;src:url(data:font/woff;base64,"
        + base64.b64encode(buf.getvalue()).decode() + ") format('woff');}\n")
html = html.replace("<style>\n", "<style>\n" + face, 1)
html = html.replace('font:15px/1.65 -apple-system', "font:15px/1.65 'ReportKR',-apple-system", 1)
html = html.replace("code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;", "code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,monospace,'ReportKR';", 1)

for k, name in FIGS.items():
    html = html.replace("{{" + k + "}}", base64.b64encode((D / "figs" / f"{name}.png").read_bytes()).decode())
assert "{{" not in html
(D / "report.html").write_text(html)
print("wrote", D / "report.html", f"{len(html)/1e6:.2f} MB (font subset {len(buf.getvalue())/1e3:.0f} KB, {len(chars)} chars)")
