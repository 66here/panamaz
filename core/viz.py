"""Komponen UI reusable: ikon SVG, kartu KPI, badge, empty state, chart helpers."""
from __future__ import annotations

import html as _html
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .settings import PALETTE, FAMILY_COLORS, METRIC_DEFS

# ------------------------------------------------------------ SVG icons ---
# Ikon minimalis gaya stroke (lucide-like), warna mengikuti konteks (currentColor)
_PATHS = {
    "bolt": "M13 2 L4.5 13.5 H11 L10 22 L19.5 10 H13 Z",
    "database": "M12 3c-4.4 0-8 1.3-8 3s3.6 3 8 3 8-1.3 8-3-3.6-3-8-3Z M4 6v6c0 1.7 3.6 3 8 3s8-1.3 8-3V6 M4 12v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6",
    "activity": "M22 12h-4l-3 9L9 3l-3 9H2",
    "target": "M12 12m-9 0a9 9 0 1 0 18 0a9 9 0 1 0-18 0 M12 12m-5 0a5 5 0 1 0 10 0a5 5 0 1 0-10 0 M12 12m-1 0a1 1 0 1 0 2 0a1 1 0 1 0-2 0",
    "layers": "M12 2 2 7l10 5 10-5-10-5Z M2 12l10 5 10-5 M2 17l10 5 10-5",
    "trophy": "M8 21h8 M12 17v4 M7 4h10v5a5 5 0 0 1-10 0V4Z M7 6H4a1 1 0 0 0-1 1c0 2.5 1.8 4 4.3 4.3 M17 6h3a1 1 0 0 1 1 1c0 2.5-1.8 4-4.3 4.3",
    "flask": "M9 3h6 M10 3v6L5 19a2 2 0 0 0 1.8 2.8h10.4A2 2 0 0 0 19 19l-5-10V3 M7.5 14h9",
    "brain": "M9.5 3A3.5 3.5 0 0 0 6 6.5c-1.8.6-3 2.2-3 4a4 4 0 0 0 2 3.5c-.3 2.2 1.4 4.5 4 4.5 1 0 2-.5 2.5-1V3.8C11 3.3 10.3 3 9.5 3Z M14.5 3A3.5 3.5 0 0 1 18 6.5c1.8.6 3 2.2 3 4a4 4 0 0 1-2 3.5c.3 2.2-1.4 4.5-4 4.5-1 0-2-.5-2.5-1V3.8C13 3.3 13.7 3 14.5 3Z",
    "gauge": "M12 14l3.5-3.5 M12 21a9 9 0 1 1 9-9 8.5 8.5 0 0 1-.5 3 M3.5 15A9 9 0 0 1 12 21",
    "cpu": "M6 6h12v12H6Z M9 1v3 M15 1v3 M9 20v3 M15 20v3 M1 9h3 M1 15h3 M20 9h3 M20 15h3 M10 10h4v4h-4Z",
    "calendar": "M5 4h14a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2Z M16 2v4 M8 2v4 M3 10h18",
    "clock": "M12 12m-9 0a9 9 0 1 0 18 0a9 9 0 1 0-18 0 M12 7v5l3 3",
    "download": "M12 3v12 M7 10l5 5 5-5 M4 21h16",
    "info": "M12 12m-9 0a9 9 0 1 0 18 0a9 9 0 1 0-18 0 M12 8h.01 M11 12h1v5h1",
    "alert": "M12 3 1.5 21h21L12 3Z M12 10v5 M12 18h.01",
    "check": "M12 12m-9 0a9 9 0 1 0 18 0a9 9 0 1 0-18 0 M8.5 12.5l2.5 2.5 4.5-5",
    "x": "M12 12m-9 0a9 9 0 1 0 18 0a9 9 0 1 0-18 0 M9 9l6 6 M15 9l-6 6",
    "minus": "M12 12m-9 0a9 9 0 1 0 18 0a9 9 0 1 0-18 0 M8 12h8",
    "file": "M14 2H7a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V7l-5-5Z M14 2v5h5 M9 13h6 M9 17h6",
    "chart": "M3 21h18 M6 17V9 M11 17V5 M16 17v-7 M21 17v-3",
    "spark": "M12 2v4 M12 18v4 M2 12h4 M18 12h4 M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8Z M5.6 5.6l2 2 M16.4 16.4l2 2 M18.4 5.6l-2 2 M7.6 16.4l-2 2",
    "sun": "M12 12m-4 0a4 4 0 1 0 8 0a4 4 0 1 0-8 0 M12 2v2 M12 20v2 M4.9 4.9l1.4 1.4 M17.7 17.7l1.4 1.4 M2 12h2 M20 12h2 M4.9 19.1l1.4-1.4 M17.7 6.3l1.4-1.4",
    "cloud": "M17.5 19a4.5 4.5 0 0 0 .4-9A7 7 0 0 0 4.7 12.5 3.5 3.5 0 0 0 6 19.5h11.5Z",
    "wind": "M9.6 4.6A2 2 0 1 1 12 7H2 M12.6 19.4A2 2 0 1 0 15 17H2 M17.5 8a2.5 2.5 0 1 1 2 4H2",
    "droplet": "M12 3s6 6.2 6 10.5a6 6 0 0 1-12 0C6 9.2 12 3 12 3Z",
    "table": "M3 5h18v14H3Z M3 10h18 M9 10v9 M15 10v9",
    "image": "M3 5h18v14H3Z M3 15l5-5 4 4 3-3 6 6 M8.5 9a1 1 0 1 0 0 .01",
    "book": "M4 19.5A2.5 2.5 0 0 1 6.5 17H20 M4 19.5A2.5 2.5 0 0 0 6.5 22H20V2H6.5A2.5 2.5 0 0 0 4 4.5v15Z",
    "search": "M11 11m-8 0a8 8 0 1 0 16 0a8 8 0 1 0-16 0 M21 21l-4.3-4.3",
    "filter": "M3 5h18l-7 8v6l-4-2v-4L3 5Z",
    "trend": "M3 17l6-6 4 4 8-8 M15 7h6v6",
    "zap-off": "M2 2l20 20 M13 3l-2.5 6 M5.5 10.5 4 15h5l-1 6 4.5-4.2",
}


def icon(name: str, size: int = 16, stroke: float = 1.8) -> str:
    """Kembalikan inline SVG ikon minimalis."""
    path = _PATHS.get(name, _PATHS["info"])
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="currentColor" stroke-width="{stroke}" stroke-linecap="round" '
        f'stroke-linejoin="round"><path d="{path}"/></svg>'
    )


def esc(text) -> str:
    return _html.escape(str(text))


def set_current(key: str) -> None:
    """Tandai halaman aktif (dipakai app.py untuk highlight navigasi)."""
    st.session_state["pei_current"] = key


# ------------------------------------------------------------ komponen ----
def tip(def_key: str | None = None, text: str | None = None) -> str:
    """Ikon info dengan tooltip CSS (definisi metrik / istilah)."""
    body = text if text is not None else METRIC_DEFS.get(def_key, "")
    return (
        f'<span class="tip">{icon("info", 13)}'
        f'<span class="tip-body">{body}</span></span>'
    )


def page_header(title: str, description: str, eyebrow: str, chips: list[str] | None = None):
    _chips = ""
    if chips:
        _chips = '<div class="pei-chips">' + "".join(
            f'<span class="pei-chip{" neutral" if i % 2 else ""}">{esc(c)}</span>'
            for i, c in enumerate(chips)
        ) + "</div>"
    st.markdown(
        f'<div class="pei-hero"><div class="pei-eyebrow">{esc(eyebrow)}</div>'
        f"<h1>{esc(title)}</h1><p>{description}</p>{_chips}</div>",
        unsafe_allow_html=True,
    )


def section(title: str, hint: str = ""):
    h = f'<span class="hint">{esc(hint)}</span>' if hint else ""
    st.markdown(
        f'<div class="pei-sec"><div class="bar"></div><h3>{esc(title)}</h3>{h}</div>',
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, sub: str = "", icon_name: str = "activity",
             tone: str = "", tip_text: str | None = None) -> str:
    t = ""
    if tip_text:
        t = f'<span class="tip">{icon("info", 13)}<span class="tip-body">{tip_text}</span></span>'
    return (
        f'<div class="kpi-card"><div class="kpi-top">'
        f'<div class="kpi-label">{esc(label)}</div>'
        f'<div class="kpi-icon {tone}">{icon(icon_name, 17)}</div></div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{sub} {t}</div></div>'
    )


def kpi_row(cards: list[str]):
    st.markdown(f'<div class="kpi-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def badge(text: str, kind: str = "muted") -> str:
    return f'<span class="badge {kind}">{esc(text)}</span>'


def note(message: str, kind: str = "info", icon_name: str | None = None, title: str = ""):
    ic = icon_name or {"info": "info", "warn": "alert", "danger": "alert", "ok": "check", "teal": "spark"}[kind]
    ttl = f"<b>{esc(title)}</b> — " if title else ""
    st.markdown(
        f'<div class="pei-note {kind}"><div class="n-icon">{icon(ic, 16)}</div>'
        f"<div>{ttl}{message}</div></div>",
        unsafe_allow_html=True,
    )


def empty_state(title: str, message: str, file_hint: str | None = None,
                steps: list[str] | None = None, icon_name: str = "database"):
    steps_html = ""
    if steps:
        steps_html = "<ul>" + "".join(f"<li>{esc(s)}</li>" for s in steps) + "</ul>"
    fh = f'<div><span class="e-file">{esc(file_hint)}</span></div>' if file_hint else ""
    st.markdown(
        f'<div class="pei-empty"><div class="e-icon">{icon(icon_name, 22)}</div>'
        f"<h4>{esc(title)}</h4><p>{message}</p>{fh}{steps_html}</div>",
        unsafe_allow_html=True,
    )


def fig_card(path: Path | str, caption: str, use_column_width: bool = True):
    p = Path(path)
    if not p.exists():
        return
    st.markdown('<div class="fig-card">', unsafe_allow_html=True)
    st.image(str(p))
    st.markdown(f'<div class="fig-cap">{caption}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------------------------------- chart helpers ----
def base(fig: go.Figure, height: int = 320, x_title: str | None = None,
         y_title: str | None = None, hover_x: str = "x", title: str | None = None):
    fig.update_layout(
        height=height,
        xaxis_title=x_title, yaxis_title=y_title,
        hovermode="x unified" if hover_x == "x" else "closest",
        title_text=title,
    )
    return fig


def line(x, y, name: str, color: str | None = None, width: float = 2.0,
         dash: str | None = None, fill: str | None = None, opacity: float = 1.0,
         showlegend: bool = True, hovertemplate: str | None = None) -> go.Scatter:
    return go.Scatter(
        x=x, y=y, name=name, mode="lines", line=dict(width=width, dash=dash),
        fill=fill, opacity=opacity,
        marker=dict(color=color), line_color=color,
        showlegend=showlegend,
        hovertemplate=hovertemplate or ("%{x}<br>" + f"{name}: " + "%{y:,.2f}<extra></extra>"),
    )


def hist_fig(values, color: str = "#20D3C2", height: int = 320, nbins: int = 60,
             mean: float | None = None, median: float | None = None,
             x_title: str = "", y_title: str = "Frekuensi"):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=values, nbinsx=nbins, marker=dict(color=color, opacity=0.75),
        hovertemplate="Rentang %{x}<br>Jumlah %{y}<extra></extra>",
    ))
    if mean is not None:
        fig.add_vline(x=mean, line_width=1.4, line_dash="dash", line_color="#FFB65C",
                      annotation_text=f"mean {mean:,.0f}", annotation_position="top right",
                      annotation_font=dict(size=10.5, color="#FFB65C"))
    if median is not None:
        fig.add_vline(x=median, line_width=1.4, line_dash="dot", line_color="#A78BFA",
                      annotation_text=f"median {median:,.0f}", annotation_position="top left",
                      annotation_font=dict(size=10.5, color="#A78BFA"))
    return base(fig, height, x_title=x_title, y_title=y_title, hover_x="closest")


def bar_fig(labels, values, color: str = "#20D3C2", height: int = 320,
            x_title: str = "", y_title: str = "", hover_fmt: str = "%{y:,.2f}",
            text_vals: bool = True, color_list: list[str] | None = None):
    mk = dict(color=color, opacity=0.85, line=dict(width=0))
    if color_list:
        mk = dict(color=color_list, opacity=0.9, line=dict(width=0))
    fig = go.Figure(go.Bar(
        x=list(labels), y=list(values), marker=mk,
        text=[f"{v:,.1f}" if isinstance(v, float) else str(v) for v in values] if text_vals else None,
        textposition="outside", textfont=dict(size=10.5, color="#B9C6D8"),
        hovertemplate="%{x}<br>" + hover_fmt.replace("%{y:,.2f}", "%{y:,.2f}") + "<extra></extra>",
    ))
    fig.update_layout(height=height, xaxis_title=x_title, yaxis_title=y_title)
    fig.update_xaxes(type="category")
    return fig


def heatmap_fig(z: pd.DataFrame, height: int = 360, colorscale: str = "Teal",
                fmt: str = ".1f", x_title: str = "", y_title: str = "",
                reversescale: bool = False):
    fig = go.Figure(go.Heatmap(
        z=z.values, x=[str(c) for c in z.columns], y=[str(i) for i in z.index],
        colorscale=colorscale, reversescale=reversescale,
        text=z.values, texttemplate=f"%{{text:{fmt}}}", textfont=dict(size=10.5),
        hovertemplate=f"{x_title or 'X'}: %{{x}}<br>{y_title or 'Y'}: %{{y}}<br>Nilai: %{{z:{fmt}}}<extra></extra>",
        xgap=2, ygap=2,
    ))
    fig.update_layout(height=height, xaxis_title=x_title, yaxis_title=y_title)
    fig.update_xaxes(side="top", type="category")
    fig.update_yaxes(type="category", autorange="reversed")
    return fig


def fmt_num(v, dec: int = 0) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:,.{dec}f}"


def download_csv(df: pd.DataFrame, filename: str, label: str = "Unduh data terfilter (CSV)", key: str | None = None):
    if df is None or len(df) == 0:
        return
    st.download_button(
        label, df.to_csv(index=False).encode("utf-8"),
        file_name=filename, mime="text/csv", key=key,
    )


def family_color(family: str) -> str:
    return FAMILY_COLORS.get(family, "#20D3C2")
