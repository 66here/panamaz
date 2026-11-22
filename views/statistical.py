"""Halaman Statistical Testing — uji Diebold-Mariano antar model terbaik."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import viz
from core.data_loader import get_hub
from core.settings import ALPHA_DM, HORIZONS, METRIC_DEFS


def render():
    viz.set_current("statistical-testing")
    hub = get_hub()
    viz.page_header(
        "Statistical Testing",
        "Uji Diebold-Mariano (DM) untuk membandingkan akurasi peramalan antara model terbaik "
        "dan model pembanding pada setiap horizon. Hasil signifikan dan tidak signifikan "
        "ditandai dengan warna berbeda.",
        "Forecast Accuracy Significance",
        chips=[f"alpha = {ALPHA_DM}", "Loss: squared error", "Varians HAC (Newey-West)"],
    )

    # --------------------------------------------------------- metodologi ---
    viz.section("Hipotesis & Metodologi")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-top"><div class="kpi-label">H₀</div>'
            f'<div class="kpi-icon blue">{viz.icon("minus", 17)}</div></div>'
            f'<div class="kpi-sub" style="font-size:.86rem;color:#C9D5E5">Kedua model memiliki <b>akurasi peramalan yang sama</b> '
            f'(tidak ada perbedaan sistematis pada loss kuadrat).</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-top"><div class="kpi-label">H₁</div>'
            f'<div class="kpi-icon amber">{viz.icon("trend", 17)}</div></div>'
            f'<div class="kpi-sub" style="font-size:.86rem;color:#C9D5E5">Kedua model memiliki <b>akurasi peramalan yang berbeda</b> '
            f'(d=u₁²−u₂², DM ≠ 0).</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-top"><div class="kpi-label">Aturan Keputusan</div>'
            f'<div class="kpi-icon">{viz.icon("check", 17)}</div></div>'
            f'<div class="kpi-sub" style="font-size:.86rem;color:#C9D5E5">p-value &lt; <b>{ALPHA_DM}</b> → tolak H₀ '
            f'(perbedaan akurasi signifikan). Statistik DM &gt; 0 → pembanding <i>lebih buruk</i> dari model terbaik.</div></div>',
            unsafe_allow_html=True,
        )
    viz.note(
        f"Protokol sesuai notebook: per horizon, model dengan RMSE test terbaik diuji terhadap <b>10 pembanding teratas</b>. "
        f"{viz.esc('')}Loss = galat kuadrat; varians memakai estimasi HAC dengan lag truncation "
        f"K = max(⌈(h−1)/3⌉, ⌈24/3⌉) untuk mengoreksi autokorelsi galat (origin tiap 3 jam). {viz.tip('DM')}",
        "info", "flask", "Desain uji",
    )

    # ------------------------------------------------------------- data -----
    dm = hub.get("dm_tests")
    if dm is None:
        viz.section("Hasil Uji")
        viz.empty_state(
            "Tabel uji DM belum tersedia",
            "Uji Diebold-Mariano dijalankan notebook setelah semua model menghasilkan prediksi pada regime test. "
            "Tanpa file hasil, dashboard hanya menampilkan metodologi di atas — tanpa angka fiktif.",
            "outputs/results/dm_tests.csv",
            ["Jalankan notebook utama (Run All) — sel 13 menghasilkan dm_tests.csv",
             "Salin ke outputs/results/",
             "Skema: horizon · terbaik · pembanding · RMSE_terbaik · RMSE_pembanding · DM_stat · p_value"],
            icon_name="flask",
        )
        return

    # ------------------------------------------------------------- aktif ----
    viz.section("Hasil Uji", f"signifikan = tolak H₀ (p < {ALPHA_DM})")
    horizons_avail = sorted(dm["horizon"].unique())
    h_sel = st.pills("Horizon", horizons_avail, default=horizons_avail,
                     key="dm_h", selection_mode="multi")
    dmv = dm[dm.horizon.isin(h_sel)] if h_sel else dm.copy()

    n_sig = int(dmv["signifikan"].sum())
    n_tot = len(dmv)
    viz.kpi_row([
        viz.kpi_card("Pasangan Diuji", str(n_tot), "model pembanding × horizon", "flask", "blue"),
        viz.kpi_card("Signifikan", str(n_sig), f"p < {ALPHA_DM} — akurasi berbeda nyata", "check", "green"),
        viz.kpi_card("Tidak Signifikan", str(n_tot - n_sig), "gagal tolak H₀ — akurasi setara", "minus", "amber"),
    ])

    # tabel dengan warna status
    def _row_html(r: pd.Series) -> str:
        sig = bool(r["signifikan"])
        if pd.notna(r["DM_stat"]) and r["DM_stat"] > 0:
            dir_txt = "pembanding lebih buruk"
        else:
            dir_txt = "pembanding lebih baik"
        badge = viz.badge("Signifikan — tolak H₀", "err") if sig else viz.badge("Tidak signifikan", "muted")
        bg = ' style="background:rgba(255,107,107,0.055)"' if sig else ""
        return (
            f'<tr{bg}>'
            f'<td>{int(r["horizon"])} jam</td>'
            f'<td><b style="color:#7BE8DC">{viz.esc(r["terbaik"])}</b></td>'
            f'<td>{viz.esc(r["pembanding"])}</td>'
            f'<td style="text-align:right">{r["RMSE_terbaik"]:.2f}</td>'
            f'<td style="text-align:right">{r["RMSE_pembanding"]:.2f}</td>'
            f'<td style="text-align:right">{r["DM_stat"]:.3f}</td>'
            f'<td style="text-align:right">{r["p_value"]:.4f}</td>'
            f'<td>{badge}<br><span style="font-size:.68rem;color:#8CA0B8">{dir_txt}</span></td></tr>'
        )

    header = (
        '<table class="pei-table"><thead><tr>'
        '<th>Horizon</th><th>Model terbaik</th><th>Pembanding</th><th style="text-align:right">RMSE terbaik</th>'
        '<th style="text-align:right">RMSE pembanding</th><th style="text-align:right">DM stat</th>'
        '<th style="text-align:right">p-value</th><th>Kesimpulan</th></tr></thead><tbody>'
    )
    rows_html = header + "".join(_row_html(r) for _, r in dmv.iterrows()) + "</tbody></table>"
    st.markdown(rows_html + _table_css(), unsafe_allow_html=True)
    viz.download_csv(dmv, "dm_tests_filtered.csv", key="dl_dm")

    # ringkasan signifikansi per horizon
    viz.section("Ringkasan per Horizon", "jumlah pasangan signifikan vs tidak")
    gg = dm.groupby("horizon")["signifikan"].agg(["sum", "count"]).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Signifikan", x=gg.horizon.astype(str), y=gg["sum"],
                         marker_color=PALETTE["red"], opacity=0.85, text=gg["sum"],
                         textposition="outside", textfont=dict(size=11, color="#FFC9C9")))
    fig.add_trace(go.Bar(name="Tidak signifikan", x=gg.horizon.astype(str), y=gg["count"] - gg["sum"],
                         marker_color="#3D5273", opacity=0.9, text=gg["count"] - gg["sum"],
                         textposition="outside", textfont=dict(size=11, color="#B9C6D8")))
    fig.update_layout(barmode="stack", height=300, xaxis_title="Horizon (jam)",
                      yaxis_title="Jumlah pasangan",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    st.plotly_chart(fig, use_container_width=True)
    viz.note(
        "Interpretasi: <b>Signifikan</b> berarti selisih RMSE antara model terbaik dan pembanding bukan karena kebetulan "
        f"sampling (p &lt; {ALPHA_DM}). Untuk horizon pendek selisih biasanya lebih mudah signifikan karena galat jauh lebih kecil "
        "dan konsisten; pada horizon panjang, model-model cenderung saling menyusul.", "teal", "info", "Cara membaca hasil",
    )


def _table_css() -> str:
    return """<style>
.pei-table {width:100%; border-collapse:collapse; font-size:.82rem;}
.pei-table th {text-align:left; padding:9px 12px; color:#8CA0B8; font-weight:600; border-bottom:1px solid rgba(148,163,184,.18); text-transform:uppercase; font-size:.66rem; letter-spacing:.08em;}
.pei-table td {padding:9px 12px; color:#C9D5E5; border-bottom:1px solid rgba(148,163,184,.08); vertical-align:top;}
.pei-table tr:hover td {background:rgba(32,211,194,.04);}
</style>"""
