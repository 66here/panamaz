"""Design system — CSS kustom + tema Plotly untuk tampilan SaaS analytics."""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from .settings import PALETTE, SERIES_COLORS


def apply_css() -> None:
    """Suntik stylesheet global (idempotent per rerun)."""
    p = PALETTE
    css = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {{
  --pei-bg: {p['bg']};
  --pei-surface: {p['surface']};
  --pei-surface-hi: {p['surface_hi']};
  --pei-accent: {p['accent']};
  --pei-text: {p['text']};
  --pei-muted: {p['muted']};
  --pei-border: {p['border']};
  --pei-shadow: 0 10px 30px rgba(2, 8, 20, 0.45);
}}

/* ------------------------------------------------- global shell -------- */
.stApp {{
  background:
    radial-gradient(1200px 500px at 85% -10%, rgba(32, 211, 194, 0.05), transparent 60%),
    radial-gradient(900px 420px at -10% 110%, rgba(76, 141, 255, 0.05), transparent 55%),
    {p['bg']};
  color: {p['text']};
  font-family: 'Inter', -apple-system, 'Segoe UI', system-ui, sans-serif;
}}
.stApp header {{ background: transparent; }}
header[data-testid="stHeader"] {{ background: transparent; height: 0px; }}
footer {{ visibility: hidden; }}
#MainMenu {{ visibility: hidden; }}

.block-container {{
  padding: 1.7rem 2.4rem 3.2rem !important;
  max-width: 1560px;
}}

h1, h2, h3, h4 {{ color: {p['text']}; font-family: 'Inter', sans-serif; }}
a {{ color: {p['accent']}; }}
p {{ color: #C9D5E5; }}
p, li, td, th {{ font-family: 'Inter', sans-serif; }}

::selection {{ background: rgba(32, 211, 194, 0.28); }}
*::-webkit-scrollbar {{ width: 9px; height: 9px; }}
*::-webkit-scrollbar-thumb {{ background: rgba(140, 160, 184, 0.25); border-radius: 8px; }}
*::-webkit-scrollbar-track {{ background: transparent; }}

/* ------------------------------------------------- sidebar ------------- */
section[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, #0A0F1C 0%, {p['bg']} 55%);
  border-right: 1px solid {p['border']};
  min-width: 272px; max-width: 272px;
}}
section[data-testid="stSidebar"] .block-container {{ padding: 1.4rem 1.25rem 2rem !important; }}

.pei-brand {{ display: flex; align-items: center; gap: 12px; padding: 4px 2px 14px 2px; }}
.pei-brand-logo {{
  width: 44px; height: 44px; border-radius: 12px; flex: 0 0 44px;
  background: linear-gradient(145deg, #16233A, #0B1220);
  border: 1px solid rgba(32, 211, 194, 0.35);
  display: flex; align-items: center; justify-content: center;
  box-shadow: inset 0 0 18px rgba(32, 211, 194, 0.12), 0 6px 18px rgba(2, 8, 20, 0.5);
}}
.pei-brand-name {{ font-size: 0.62rem; letter-spacing: 0.24em; color: {p['muted']}; font-weight: 600; line-height: 1.1; }}
.pei-brand-title {{ font-size: 1.02rem; font-weight: 800; color: {p['text']}; line-height: 1.15; }}
.pei-brand-title span {{ color: {p['accent']}; }}
.pei-nav-label {{
  font-size: 0.66rem; letter-spacing: 0.22em; color: {p['muted']};
  font-weight: 700; margin: 14px 4px 6px 4px; text-transform: uppercase;
}}

/* nav item */
[data-testid="stPageLink"] {{ margin: 0; padding: 0; }}
[data-testid="stPageLink"] a {{
  display: flex; align-items: center; gap: 10px;
  padding: 0.56rem 0.75rem !important;
  margin: 2px 0 !important;
  border-radius: 10px !important;
  border: 1px solid transparent !important;
  color: {p['muted']} !important;
  font-size: 0.9rem !important; font-weight: 500 !important;
  text-decoration: none !important;
  transition: background 0.16s ease, color 0.16s ease, border-color 0.16s ease;
}}
[data-testid="stPageLink"] a:hover {{
  background: rgba(32, 211, 194, 0.07) !important;
  color: #E7EEF7 !important;
}}
[data-testid="stPageLink"] a span {{ font-size: 1rem; }}

.pei-side-card {{
  background: rgba(17, 28, 46, 0.75);
  border: 1px solid {p['border']};
  border-radius: 12px; padding: 12px 14px; margin-top: 16px;
}}
.pei-side-card b {{ color: #DDE7F3; font-size: 0.8rem; }}
.pei-side-foot {{ color: {p['muted']}; font-size: 0.72rem; margin-top: 14px; line-height: 1.5; }}

/* ------------------------------------------------- hero ---------------- */
.pei-hero {{ margin: 0 0 1.25rem 0; }}
.pei-eyebrow {{
  display: inline-flex; align-items: center; gap: 8px;
  font-size: 0.68rem; font-weight: 700; letter-spacing: 0.2em;
  color: {p['accent']}; text-transform: uppercase; margin-bottom: 6px;
}}
.pei-eyebrow::before {{
  content: ""; width: 22px; height: 2px; border-radius: 2px;
  background: linear-gradient(90deg, {p['accent']}, transparent);
}}
.pei-hero h1 {{ font-size: 1.65rem; font-weight: 800; margin: 0 0 6px 0; letter-spacing: -0.02em; }}
.pei-hero p {{ color: {p['muted']}; font-size: 0.92rem; margin: 0; max-width: 860px; line-height: 1.55; }}
.pei-chips {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }}
.pei-chip {{
  font-size: 0.74rem; font-weight: 600; color: #CFE4E1;
  background: rgba(32, 211, 194, 0.08);
  border: 1px solid rgba(32, 211, 194, 0.22);
  padding: 0.28rem 0.7rem; border-radius: 999px;
}}
.pei-chip.neutral {{ color: #C9D5E5; background: rgba(140, 160, 184, 0.08); border-color: rgba(140, 160, 184, 0.22); }}

/* ------------------------------------------------- cards / KPI --------- */
.kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; }}
.kpi-grid.cols-3 {{ grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); }}
.kpi-card {{
  position: relative; overflow: hidden;
  background: linear-gradient(180deg, #14203A 0%, {p['surface']} 100%);
  border: 1px solid {p['border']};
  border-radius: 16px;
  padding: 16px 18px 15px 18px;
  box-shadow: var(--pei-shadow);
  transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
}}
.kpi-card:hover {{
  transform: translateY(-3px);
  border-color: rgba(32, 211, 194, 0.4);
  box-shadow: 0 14px 34px rgba(2, 8, 20, 0.55);
}}
.kpi-card::before {{
  content: ""; position: absolute; inset: 0 0 auto 0; height: 2px;
  background: linear-gradient(90deg, rgba(32, 211, 194, 0.65), transparent 65%);
  opacity: 0; transition: opacity 0.2s ease;
}}
.kpi-card:hover::before {{ opacity: 1; }}
.kpi-top {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; }}
.kpi-label {{ font-size: 0.66rem; font-weight: 700; letter-spacing: 0.14em; color: {p['muted']}; text-transform: uppercase; }}
.kpi-icon {{
  width: 34px; height: 34px; border-radius: 10px; flex: 0 0 34px;
  background: rgba(32, 211, 194, 0.12); color: {p['accent']};
  display: flex; align-items: center; justify-content: center;
}}
.kpi-icon.blue {{ background: rgba(76, 141, 255, 0.12); color: {p['blue']}; }}
.kpi-icon.amber {{ background: rgba(255, 182, 92, 0.12); color: {p['amber']}; }}
.kpi-icon.purple {{ background: rgba(167, 139, 250, 0.13); color: {p['purple']}; }}
.kpi-icon.red {{ background: rgba(255, 107, 107, 0.12); color: {p['red']}; }}
.kpi-icon.green {{ background: rgba(61, 220, 151, 0.12); color: {p['green']}; }}
.kpi-value {{ font-size: 1.5rem; font-weight: 750; color: {p['text']}; letter-spacing: -0.01em; line-height: 1.15; }}
.kpi-value small {{ font-size: 0.85rem; font-weight: 600; color: #B9C6D8; }}
.kpi-sub {{ font-size: 0.76rem; color: {p['muted']}; margin-top: 4px; line-height: 1.4; }}

/* ------------------------------------------------- section header ------ */
.pei-sec {{ display: flex; align-items: center; gap: 10px; margin: 1.6rem 0 0.7rem 0; }}
.pei-sec .bar {{ width: 4px; height: 18px; border-radius: 3px; background: {p['accent']}; box-shadow: 0 0 12px rgba(32, 211, 194, 0.5); }}
.pei-sec h3 {{ margin: 0; font-size: 1.02rem; font-weight: 700; letter-spacing: -0.01em; }}
.pei-sec .hint {{ color: {p['muted']}; font-size: 0.76rem; margin-left: 4px; }}

/* ------------------------------------------------- notes / badges ------ */
.pei-note {{
  display: flex; gap: 12px; align-items: flex-start;
  border-radius: 12px; padding: 13px 16px; font-size: 0.86rem; line-height: 1.55;
  border: 1px solid; margin: 0.6rem 0;
}}
.pei-note .n-icon {{ flex: 0 0 18px; margin-top: 2px; }}
.pei-note.info {{ background: rgba(76, 141, 255, 0.08); border-color: rgba(76, 141, 255, 0.28); color: #CFE0FF; }}
.pei-note.warn {{ background: rgba(255, 182, 92, 0.07); border-color: rgba(255, 182, 92, 0.3); color: #FFD9A8; }}
.pei-note.danger {{ background: rgba(255, 107, 107, 0.07); border-color: rgba(255, 107, 107, 0.3); color: #FFC9C9; }}
.pei-note.ok {{ background: rgba(61, 220, 151, 0.07); border-color: rgba(61, 220, 151, 0.28); color: #C4F2DE; }}
.pei-note.teal {{ background: rgba(32, 211, 194, 0.07); border-color: rgba(32, 211, 194, 0.28); color: #C7EFEA; }}
.pei-note b {{ color: inherit; }}

.badge {{
  display: inline-flex; align-items: center; gap: 6px;
  padding: 0.16rem 0.6rem; border-radius: 999px;
  font-size: 0.7rem; font-weight: 700; letter-spacing: 0.02em;
  border: 1px solid;
  white-space: nowrap;
}}
.badge.ok {{ color: #7BE8C8; background: rgba(61, 220, 151, 0.1); border-color: rgba(61, 220, 151, 0.35); }}
.pei-side-card .badge {{ white-space: normal; line-height: 1.4; }}
.badge.warn {{ color: #FFD9A8; background: rgba(255, 182, 92, 0.1); border-color: rgba(255, 182, 92, 0.4); }}
.badge.err {{ color: #FFC9C9; background: rgba(255, 107, 107, 0.1); border-color: rgba(255, 107, 107, 0.4); }}
.badge.muted {{ color: {p['muted']}; background: rgba(140, 160, 184, 0.08); border-color: rgba(140, 160, 184, 0.28); }}
.badge.teal {{ color: #7BE8DC; background: rgba(32, 211, 194, 0.1); border-color: rgba(32, 211, 194, 0.4); }}

/* ------------------------------------------------- empty state --------- */
.pei-empty {{
  border: 1.5px dashed rgba(140, 160, 184, 0.3);
  border-radius: 16px;
  background: rgba(17, 28, 46, 0.45);
  padding: 30px 26px;
  text-align: center;
  margin: 0.6rem 0;
}}
.pei-empty .e-icon {{
  width: 46px; height: 46px; margin: 0 auto 12px auto; border-radius: 13px;
  background: rgba(140, 160, 184, 0.09); color: {p['muted']};
  display: flex; align-items: center; justify-content: center;
}}
.pei-empty h4 {{ color: #DDE7F3; font-size: 0.98rem; margin: 0 0 6px 0; font-weight: 700; }}
.pei-empty p {{ color: {p['muted']}; font-size: 0.83rem; margin: 0 auto 10px auto; max-width: 620px; line-height: 1.55; }}
.pei-empty .e-file {{
  display: inline-block; font-family: 'JetBrains Mono', 'Fira Code', monospace;
  font-size: 0.76rem; color: #7BE8DC; background: rgba(32, 211, 194, 0.07);
  border: 1px solid rgba(32, 211, 194, 0.25); border-radius: 8px; padding: 5px 10px; margin: 2px 0 4px 0;
}}
.pei-empty ul {{ text-align: left; display: inline-block; margin: 8px auto 0 auto; padding-left: 18px; color: {p['muted']}; font-size: 0.8rem; line-height: 1.7; }}

/* ------------------------------------------------- tooltip ikon info --- */
.tip {{ position: relative; display: inline-flex; color: {p['muted']}; cursor: help; vertical-align: middle; }}
.tip:hover {{ color: {p['accent']}; }}
.tip .tip-body {{
  visibility: hidden; opacity: 0; position: absolute; bottom: calc(100% + 8px); left: 50%;
  transform: translateX(-50%); width: 280px; z-index: 999;
  background: #0A0F1C; color: #D9E4F2; border: 1px solid rgba(32, 211, 194, 0.3);
  border-radius: 10px; padding: 10px 12px; font-size: 0.74rem; line-height: 1.5;
  box-shadow: 0 12px 30px rgba(2, 8, 20, 0.6); text-align: left; font-weight: 400;
  transition: opacity 0.15s ease;
}}
.tip:hover .tip-body {{ visibility: visible; opacity: 1; }}
.tip .tip-body b {{ color: {p['accent']}; }}

/* ------------------------------------------------- figure card --------- */
.fig-card {{ background: #FDFEFF; border: 1px solid rgba(148, 163, 184, 0.25); border-radius: 14px; padding: 10px; box-shadow: var(--pei-shadow); }}
.fig-cap {{ color: {p['muted']}; font-size: 0.76rem; margin-top: 8px; line-height: 1.5; }}
.fig-cap b {{ color: #C9D5E5; }}

/* ------------------------------------------------- widgets ------------- */
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {{
  background-color: #0F1930 !important;
  border: 1px solid rgba(140, 160, 184, 0.22) !important;
  border-radius: 10px !important;
  color: {p['text']} !important;
  transition: border-color 0.15s ease;
}}
div[data-baseweb="select"] > div:hover {{ border-color: rgba(32, 211, 194, 0.45) !important; }}
div[data-baseweb="popover"] > div {{ background-color: #0F1930; }}
[data-baseweb="popover"] [role="listbox"] {{ background-color: #0F1930; }}
[data-baseweb="menu"] {{ background-color: #0F1930; }}
[data-baseweb="menu"] li {{ color: #D9E4F2; }}
[data-baseweb="checkbox"] span {{ color: #C9D5E5; }}

.stButton button, .stDownloadButton button {{
  background: linear-gradient(180deg, rgba(32, 211, 194, 0.16), rgba(32, 211, 194, 0.08)) !important;
  color: {p['accent']} !important;
  border: 1px solid rgba(32, 211, 194, 0.4) !important;
  border-radius: 10px !important;
  font-weight: 600 !important;
  padding: 0.4rem 1rem !important;
  transition: all 0.16s ease !important;
}}
.stButton button:hover, .stDownloadButton button:hover {{
  transform: translateY(-1px);
  box-shadow: 0 8px 20px rgba(32, 211, 194, 0.15);
  border-color: rgba(32, 211, 194, 0.7) !important;
}}

[data-testid="stExpander"] {{
  border: 1px solid {p['border']} !important;
  border-radius: 14px !important;
  background: rgba(17, 28, 46, 0.5);
  overflow: hidden;
}}
[data-testid="stExpander"] summary {{
  font-size: 0.9rem; font-weight: 600; color: #DDE7F3; padding: 6px 4px;
}}
[data-testid="stExpander"] summary:hover {{ color: {p['accent']}; }}
[data-testid="stExpanderDetails"] {{ padding-top: 2px; }}

[data-testid="stDataFrame"] {{
  border: 1px solid {p['border']};
  border-radius: 12px;
  overflow: hidden;
  background: {p['surface']};
}}
[data-testid="stDataFrame"] .glideCell, [data-testid="stDataFrame"] .glideHeaderCell {{ font-size: 0.82rem; }}

[data-testid="stTabs"] [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {p['border']}; }}
[data-testid="stTabs"] [data-baseweb="tab"] {{
  padding: 8px 16px; border-radius: 9px 9px 0 0; color: {p['muted']};
  font-weight: 600; font-size: 0.88rem;
}}
[data-testid="stTabs"] [aria-selected="true"] {{ color: {p['accent']}; background: rgba(32, 211, 194, 0.07); box-shadow: inset 0 -2px 0 {p['accent']}; }}

[data-testid="stPills"] button {{
  border-radius: 999px !important;
  border: 1px solid rgba(140, 160, 184, 0.25) !important;
  background: rgba(17, 28, 46, 0.6) !important;
  color: {p['muted']} !important; font-weight: 600 !important; font-size: 0.82rem !important;
  padding: 0.25rem 0.9rem !important;
  transition: all 0.15s ease;
}}
[data-testid="stPills"] button:hover {{ border-color: rgba(32, 211, 194, 0.5) !important; color: #E7EEF7 !important; }}
[data-testid="stPills"] button[data-selected="true"] {{
  background: rgba(32, 211, 194, 0.14) !important;
  border-color: rgba(32, 211, 194, 0.55) !important;
  color: {p['accent']} !important;
}}

[data-testid="stMetricValue"] {{ font-family: 'Inter', sans-serif; }}

[data-testid="stImage"] img, .stImage img {{ border-radius: 10px; }}
caption, .stCaption {{ color: {p['muted']} !important; }}

.stTooltipIcon {{ color: {p['muted']}; }}

/* slider */
div[data-testid="stSlider"] [role="slider"] {{
  background-color: {p['accent']}; border: 2px solid #0B1220;
}}
div[data-testid="stSlider"] [data-baseweb="slider"] > div {{ color: {p['muted']}; }}
</style>
"""
    st.markdown(css, unsafe_allow_html=True)


def apply_plotly_theme() -> None:
    """Registrasi template Plotly 'pei' (dipakai semua chart)."""
    if "pei" in pio.templates:
        return
    p = PALETTE
    t = go.layout.Template()
    t.layout = go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, -apple-system, 'Segoe UI', sans-serif", color="#D9E4F2", size=12.5),
        colorway=SERIES_COLORS,
        margin=dict(l=8, r=14, t=34, b=8),
        hoverlabel=dict(
            bgcolor="#0A0F1C",
            bordercolor="rgba(32,211,194,0.45)",
            font=dict(color="#E7EEF7", family="Inter", size=12.5),
        ),
        title=dict(font=dict(size=14.5, color=p["text"], family="Inter"), x=0.01, xanchor="left"),
        legend=dict(
            bgcolor="rgba(10,15,28,0.55)",
            bordercolor="rgba(148,163,184,0.18)",
            borderwidth=1,
            font=dict(size=11.5),
        ),
        xaxis=dict(
            gridcolor="rgba(140,160,184,0.10)",
            zerolinecolor="rgba(140,160,184,0.18)",
            linecolor="rgba(140,160,184,0.25)",
            tickfont=dict(size=11, color=p["muted"]),
            title_font=dict(size=12, color=p["muted"]),
        ),
        yaxis=dict(
            gridcolor="rgba(140,160,184,0.10)",
            zerolinecolor="rgba(140,160,184,0.18)",
            linecolor="rgba(140,160,184,0.25)",
            tickfont=dict(size=11, color=p["muted"]),
            title_font=dict(size=12, color=p["muted"]),
        ),
        coloraxis=dict(
            colorbar=dict(
                bgcolor="rgba(0,0,0,0)", outlinecolor="rgba(148,163,184,0.2)",
                tickfont=dict(size=10.5, color=p["muted"]), thickness=13,
            )
        ),
    )
    pio.templates["pei"] = t
    pio.templates.default = "pei"
