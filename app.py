"""PANAMA ENERGY INTELLIGENCE — dashboard analisis peramalan beban listrik Panama.

Struktur modular:
- core/settings.py     : konfigurasi terpusat (palet, roster, kebijakan Chronos)
- core/theme.py        : design system (CSS + tema Plotly)
- core/data_loader.py  : satu sumber baca file hasil notebook (outputs/)
- core/viz.py          : komponen UI (kartu KPI, empty state, chart helpers)
- views/*.py           : 8 halaman dashboard
"""
from __future__ import annotations

import streamlit as st

from core import settings, theme, viz
from core.data_loader import get_hub
from views import (benchmark, cost, data_explorer, drivers, efficiency,
                   forecasting, overview, statistical)

st.set_page_config(
    page_title="Panama Energy Intelligence",
    page_icon=str(settings.ASSETS_DIR / "logo.png"),
    layout="wide",
    initial_sidebar_state="expanded",
)

theme.apply_css()
theme.apply_plotly_theme()

PAGES = [
    (overview.render, "Overview", ":material/dashboard:", "overview"),
    (data_explorer.render, "Data Explorer", ":material/database:", "data-explorer"),
    (forecasting.render, "Forecasting Analysis", ":material/monitoring:", "forecasting-analysis"),
    (benchmark.render, "Model Benchmark", ":material/emoji_events:", "model-benchmark"),
    (statistical.render, "Statistical Testing", ":material/science:", "statistical-testing"),
    (drivers.render, "Driver Analysis", ":material/lightbulb:", "driver-analysis"),
    (efficiency.render, "Data Efficiency", ":material/speed:", "data-efficiency"),
    (cost.render, "Computational Cost", ":material/memory:", "computational-cost"),
]

page_objs = [st.Page(fn, title=t, icon=i, url_path=u) for fn, t, i, u in PAGES]
nav = st.navigation(page_objs, position="hidden")

# ------------------------------------------------------- sidebar (atas) ----
with st.sidebar:
    st.markdown(
        f"""
        <div class="pei-brand">
          <div class="pei-brand-logo">{viz.icon('bolt', 22, 2.2)}</div>
          <div>
            <div class="pei-brand-name">PANAMA ENERGY</div>
            <div class="pei-brand-title">INTEL<span>LIGENCE</span></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="pei-nav-label">Navigasi</div>', unsafe_allow_html=True)

# --------------------------------------------------------------- main ----
try:
    nav.run()
except Exception as exc:  # error handling global — halaman tidak pernah "putih"
    viz.page_header("Terjadi Kesalahan", "Halaman gagal dirender karena kesalahan tak terduga. "
                    "Cek pesan di bawah — umumnya terkait file hasil yang belum lengkap.", "Error")
    st.error(f"{type(exc).__name__}: {exc}")
    with st.expander("Detail teknis (untuk debugging)"):
        st.exception(exc)
    viz.note(
        "Perbaiki file hasil di <code>outputs/results/</code> atau hubungi pengelola dashboard. "
        "Halaman lain tetap dapat diakses lewat navigasi.", "warn", "alert",
    )

# ------------------------------------------- sidebar (bawah) + nav aktif ----
cur = st.session_state.get("pei_current", "overview")
cur_css = 'a[href=""]' if cur == "overview" else f'a[href="{cur}"]'
with st.sidebar:
    st.markdown(
        f"<style>{cur_css}{{"
        f"background: linear-gradient(90deg, rgba(32,211,194,.16), rgba(32,211,194,.03)) !important;"
        f"color: {settings.PALETTE['accent']} !important;"
        f"border: 1px solid rgba(32,211,194,.3) !important;"
        f"font-weight: 600 !important;}}</style>",
        unsafe_allow_html=True,
    )
    for obj, t, i, u in zip(page_objs, [p[1] for p in PAGES], [p[2] for p in PAGES], [p[3] for p in PAGES]):
        st.page_link(obj, label=t, icon=i)

    # status data ringkas
    hub = get_hub()
    _, n_ess, n_tot = hub.counts()
    chronos_badge = (
        '<span class="badge warn">Chronos-2: Not Available / Failed Execution</span>'
        if not hub.chronos_on()
        else '<span class="badge ok">Chronos-2: tersedia</span>'
    )
    st.markdown(
        f"""
        <div class="pei-side-card">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
            <b>Sumber data</b>
            <span class="badge {'ok' if n_ess == n_tot else 'muted'}">{n_ess}/{n_tot} file utama</span>
          </div>
          <div style="font-size:.74rem;color:{settings.PALETTE['muted']};line-height:1.55">
            Folder <code>outputs/results/</code> dicek otomatis. File hasil notebook yang belum ada
            ditampilkan sebagai <i>empty state</i> informatif — dashboard tidak mengarang angka.
          </div>
          <div style="margin-top:10px">{chronos_badge}</div>
        </div>
        <div class="pei-side-foot">
          {viz.icon('bolt', 10)} {settings.APP_NAME} v{settings.APP_VERSION}<br>
          Forecasting beban listrik nasional Panama 2015–2020 · sarima · ets · arimax · rf · xgboost ·
          lightgbm · lstm · gru · cnn-1d · chronos-2
        </div>
        """,
        unsafe_allow_html=True,
    )
