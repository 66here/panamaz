"""Halaman Driver Analysis — interpretabilitas: SHAP, kelompok fitur, PDP, permutasi DL."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core import viz
from core.data_loader import get_hub
from core.settings import PALETTE, WEATHER_VARS, CITIES

GROUP_DEFS = {
    "Historical Load": "lag_0/1/24/168, rolling mean & std 24/168, same_hour_lastday, same_hour_lastweek",
    "Calendar": "hour, dow, month, is_weekend, holiday, Holiday_ID, school, flag anomali",
    "Weather Tocumen": "T2M_toc, QV2M_toc, TQL_toc, W2M_toc",
    "Weather Santiago": "T2M_san, QV2M_san, TQL_san, W2M_san",
    "Weather David": "T2M_dav, QV2M_dav, TQL_dav, W2M_dav",
    "Weather PCA": "wpc1–wpc7 (95% varians cuaca)",
}

NOTE_SRC = (
    "Analisis faktor penggerak memakai model ML terbaik pada set fitur <b>F3</b> (SHAP, TreeExplainer) "
    "dan model DL terbaik (permutation importance), sesuai konfigurasi notebook (SHAP_FSET=F3, h=1 &amp; 24). "
    "Chronos dikecualikan dari analisis interpretabilitas."
)


def render():
    viz.set_current("driver-analysis")
    hub = get_hub()
    viz.page_header(
        "Driver Analysis",
        "Interpretabilitas model: fitur mana yang paling menggerakkan prediksi beban listrik. "
        "Menampilkan SHAP feature importance, kontribusi per kelompok fitur, kurva suhu–beban (PDP), "
        "permutation importance model deep learning, dan loading PCA cuaca.",
        "Model Interpretability",
        chips=list(GROUP_DEFS),
    )
    viz.note(NOTE_SRC, "info", "brain", "Sumber analisis")

    # ------------------------------------------------- kelompok fitur ------
    viz.section("Kelompok Fitur", "pemetaan fitur → kelompok sesuai notebook")
    grp_cols = st.columns(3)
    tone = {"Historical Load": "blue", "Calendar": "", "Weather Tocumen": "amber",
            "Weather Santiago": "amber", "Weather David": "amber", "Weather PCA": "purple"}
    ico = {"Historical Load": "activity", "Calendar": "calendar", "Weather Tocumen": "sun",
           "Weather Santiago": "sun", "Weather David": "sun", "Weather PCA": "layers"}
    for i, (g, d) in enumerate(GROUP_DEFS.items()):
        with grp_cols[i % 3]:
            st.markdown(
                f'<div class="kpi-card" style="min-height:104px"><div class="kpi-top">'
                f'<div class="kpi-label">{viz.esc(g)}</div>'
                f'<div class="kpi-icon {tone.get(g, "")}">{viz.icon(ico.get(g, "info"), 17)}</div></div>'
                f'<div class="kpi-sub" style="font-size:.78rem">{viz.esc(d)}</div></div>',
                unsafe_allow_html=True,
            )

    # --------------------------------------------------------- SHAP --------
    viz.section("SHAP Feature Importance", "kontribusi absolut rata-rata per fitur dan kelompok")
    grp = hub.get("importance_by_group")
    dli = hub.get("dl_perm_importance")
    if grp is not None:
        gmat = grp.set_index("kelompok").astype(float)
        c1, c2 = st.columns(2)
        with c1:
            import plotly.graph_objects as go
            fig = go.Figure()
            for col in gmat.columns:
                fig.add_trace(go.Bar(name=col, x=list(gmat.index), y=gmat[col].values,
                                     hovertemplate="%{x}<br>" + col + ": %{y:.1f}%<extra></extra>"))
            fig.update_layout(barmode="group", height=360, yaxis_title="% kontribusi",
                              xaxis=dict(tickangle=-18),
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = go.Figure()
            for i, col in enumerate(gmat.columns):
                fig2.add_trace(go.Bar(name=col, y=list(gmat.index), x=gmat[col].values,
                                      orientation="h", marker=dict(color=PALETTE["accent"] if i == 0 else PALETTE["blue"]),
                                      opacity=0.85, hovertemplate="%{y}<br>" + col + ": %{x:.1f}%<extra></extra>"))
            fig2.update_layout(barmode="stack", height=360, xaxis_title="% kontribusi",
                               legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
            st.plotly_chart(fig2, use_container_width=True)
        viz.download_csv(gmat.reset_index(), "shap_group_pct.csv", key="dl_shap")
        top_grp = gmat.sum(axis=1).sort_values(ascending=False)
        viz.note(
            "Interpretasi otomatis: kelompok dengan kontribusi SHAP terbesar (menjumlahkan kedua horizon) adalah "
            f"<b>{viz.esc(top_grp.index[0])}</b> ({top_grp.iloc[0]:.1f}% agregat), disusul "
            f"{', '.join(viz.esc(f'{x} ({y:.1f}%)') for x, y in top_grp.iloc[1:3].items())}.", "teal", "spark", "Bacaan cepat")
    else:
        viz.empty_state(
            "Tabel SHAP per kelompok belum tersedia",
            "Kontribusi interaktif per kelompok akan aktif setelah file hasil tersedia. Gambar SHAP asli dari run notebook ditampilkan di bawah.",
            "outputs/results/importance_by_group_pct.csv",
            ["Jalankan notebook (sel 14: faktor penggerak) hingga selesai",
             "Salin results/importance_by_group_pct.csv ke outputs/results/"],
            icon_name="brain",
        )

    # gambar SHAP asli
    figs = hub.figures(["drv01_shap_summary_h1", "drv01_shap_summary_h24"])
    if figs:
        cap = {
            "drv01_shap_summary_h1": "<b>drv01_shap_summary_h1.png</b> — Ringkasan SHAP model ML terbaik (F3), horizon 1 jam. Titik = sampel; sumbu x = dampak ke prediksi (MWh).",
            "drv01_shap_summary_h24": "<b>drv01_shap_summary_h24.png</b> — Ringkasan SHAP, horizon 24 jam. Perhatikan pergeseran dominasi fitur saat horizon memanjang.",
        }
        c1, c2 = st.columns(2)
        names = [n for n, _ in figs]
        if "drv01_shap_summary_h1" in names:
            with c1:
                viz.fig_card(hub.figure("drv01_shap_summary_h1"), cap["drv01_shap_summary_h1"])
        if "drv01_shap_summary_h24" in names:
            with c2:
                viz.fig_card(hub.figure("drv01_shap_summary_h24"), cap["drv01_shap_summary_h24"])

    # kontribusi kelompok (gambar notebook)
    f = hub.figure("drv04_kontribusi_kelompok")
    if f:
        viz.fig_card(f, "<b>drv04_kontribusi_kelompok.png</b> — Kontribusi kelompok fitur ML (SHAP, %) pada horizon 1 dan 24 jam — riwayat beban mendominasi, disusul kalender dan cuaca.")

    # ------------------------------------------------- PDP suhu -------------
    viz.section("PDP Temperatur T2M", "hubungan suhu–beban yang dipelajari model")
    f1 = hub.figure("drv03_pdp_suhu_h1")
    f24 = hub.figure("drv03_pdp_suhu_h24")
    if f1 or f24:
        c1, c2 = st.columns(2)
        with c1:
            if f1:
                viz.fig_card(f1, "<b>drv03_pdp_suhu_h1.png</b> — Partial dependence suhu (grid 25 titik, 3 kota), horizon 1 jam.")
        with c2:
            if f24:
                viz.fig_card(f24, "<b>drv03_pdp_suhu_h24.png</b> — PDP suhu, horizon 24 jam; kurva lebih landai karena informasi lag beban mendominasi.")
        viz.note(
            "Kurva naik pada rentang suhu tinggi (≈26–30 °C) menunjukkan efek pendinginan ruang (AC) terhadap beban "
            "puncak siang — konsisten dengan iklim tropis Panama. Nilai pasti kurva berasal dari notebook "
            "(matplotlib, sel 14).", "teal", "sun", "Interpretasi",
        )

    # dependence plots per kota
    dep = hub.figures(["drv02_shap_dep_T2M_toc_h1", "drv02_shap_dep_T2M_toc_h24",
                       "drv02_shap_dep_T2M_san_h1", "drv02_shap_dep_T2M_san_h24",
                       "drv02_shap_dep_T2M_dav_h1", "drv02_shap_dep_T2M_dav_h24"])
    if dep:
        with st.expander("SHAP dependence plot T2M per kota (Tocumen · Santiago · David)"):
            tabs = st.tabs(["Tocumen h=1", "Tocumen h=24", "Santiago h=1", "Santiago h=24", "David h=1", "David h=24"])
            caps = {
                "drv02_shap_dep_T2M_toc_h1": "Dependence SHAP T2M Tocumen, h=1",
                "drv02_shap_dep_T2M_toc_h24": "Dependence SHAP T2M Tocumen, h=24",
                "drv02_shap_dep_T2M_san_h1": "Dependence SHAP T2M Santiago, h=1",
                "drv02_shap_dep_T2M_san_h24": "Dependence SHAP T2M Santiago, h=24",
                "drv02_shap_dep_T2M_dav_h1": "Dependence SHAP T2M David, h=1",
                "drv02_shap_dep_T2M_dav_h24": "Dependence SHAP T2M David, h=24",
            }
            for tab, (n, p) in zip(tabs, dep):
                with tab:
                    viz.fig_card(p, f"<b>{n}.png</b> — {caps.get(n, n)}")

    # --------------------------------------------- permutation importance DL --
    viz.section("Permutation Importance — Deep Learning", "kenaikan RMSE saat kelompok fitur diacak (test)")
    if dli is not None:
        import plotly.graph_objects as go
        dli = dli.sort_values(dli.columns[1], ascending=True) if len(dli.columns) > 1 else dli
        val_col = [c for c in dli.columns if "RMSE" in c][0]
        fig = go.Figure(go.Bar(
            y=list(dli["fitur"]), x=dli[val_col], orientation="h",
            marker=dict(color=PALETTE["purple"], opacity=0.85),
            hovertemplate="%{y}<br>+%{x:.2f} MWh<extra></extra>",
        ))
        fig.update_layout(height=300, x_title="Kenaikan RMSE (MWh)", y_title="Kelompok fitur")
        st.plotly_chart(fig, use_container_width=True)
        viz.download_csv(dli, "dl_permutation_importance.csv", key="dl_perm")
    else:
        viz.empty_state(
            "Permutation importance DL belum tersedia",
            "Nilai interaktif aktif setelah file tersedia; gambar statis dari notebook di bawah.",
            "outputs/results/dl_permutation_importance.csv", icon_name="cpu",
        )
        f = hub.figure("drv05_dl_permutasi")
        if f:
            viz.fig_card(f, "<b>drv05_dl_permutasi.png</b> — Kenaikan RMSE (MWh) model DL terbaik saat tiap kelompok fitur diacak; riwayat beban jauh paling dominan.")

    # ------------------------------------------------------ PCA cuaca -------
    viz.section("Loading PCA Cuaca", "basis komponen untuk set fitur F4")
    pca = hub.get("pca_loadings")
    if pca is not None:
        pmat = pca.set_index("variabel").astype(float)
        fig = viz.heatmap_fig(pmat, height=430, colorscale="RdBu", fmt=".2f",
                              x_title="Komponen", y_title="Variabel cuaca", reversescale=True)
        fig.update_traces(zmin=-1, zmax=1)
        st.plotly_chart(fig, use_container_width=True)
        pc1 = pmat["wpc1"]
        top1 = pc1.abs().sort_values(ascending=False).head(4)
        city_counts = {c: sum(1 for v in top1.index if v.endswith("_" + k)) for k, c in CITIES.items()}
        city_txt = ", ".join(f"<b>{c}</b> ({n} variabel)" for c, n in city_counts.items() if n)
        viz.note(
            f"Komponen <b>wpc1</b> — penyumbang varians terbesar — didominasi "
            f"{', '.join(f'<b>{viz.esc(v)}</b> ({pc1[v]:+.2f})' for v in top1.index)}; distribusi kota pada 4 variabel teratas: {city_txt}. "
            "Komponen ini mewakili sinyal kelembapan-temperatur bersama ketiga kota.", "teal", "layers", "Interpretasi otomatis")
    else:
        viz.empty_state("Loading PCA belum tersedia", "File loading komponen cuaca belum ada di outputs.", "outputs/results/pca_weather_loadings.csv", icon_name="layers")
