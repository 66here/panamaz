"""Halaman Data Efficiency — analisis cold start terhadap jumlah data training."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import viz
from core.data_loader import get_hub
from core.settings import HORIZONS, PALETTE

COLD_LABELS = {
    30: "1 bulan", 90: "3 bulan", 365: "1 tahun", 1095: "3 tahun",
}
FAM_COLORS = {"Cold-ML": "#3DDC97", "Cold-DL": "#F472B6", "Chronos": "#20D3C2"}


def render():
    viz.set_current("data-efficiency")
    hub = get_hub()
    viz.page_header(
        "Data Efficiency",
        "Analisis cold start: seberapa banyak data training yang benar-benar dibutuhkan? Model ML dan DL "
        "terbaik (dipilih dari RMSE validasi) dilatih ulang hanya dengan data terbaru — 1 bulan, 3 bulan, "
        "1 tahun, 3 tahun — lalu dibandingkan dengan full training.",
        "Cold Start Analysis",
        chips=["1 bulan", "3 bulan", "1 tahun", "3 tahun", "Full training"],
    )

    reg = hub.get("regime_summary")
    full_days = None
    if reg is not None:
        try:
            tr = reg.set_index("regime").loc["train"]
            full_days = int(pd.to_numeric(tr["jumlah_jam"])) // 24
        except Exception:
            full_days = None

    # kartu konfigurasi
    viz.section("Konfigurasi Uji", "jumlah data training terbaru yang dipakai untuk retraining")
    cards = []
    for days, lab in COLD_LABELS.items():
        cards.append(viz.kpi_card(lab, f"{days} hari", "data terbaru sebelum TRAIN_END", "clock", "blue"))
    cards.append(viz.kpi_card("Full Training", f"≈{viz.fmt_num(full_days)} hari" if full_days else "penuh",
                              "70% pertama kronologis (train regime)", "database", "green"))
    viz.kpi_row(cards)

    cold = hub.get("cold_start_rmse")
    h_sel = st.pills("Horizon (jam)", HORIZONS, default=HORIZONS[2], key="cs_h") or HORIZONS[2]

    if cold is None:
        viz.section("Kurva Cold Start", "RMSE test terhadap jumlah data training")
        viz.empty_state(
            "Hasil cold start belum tersedia",
            "Kurva interaktif RMSE vs jumlah data training aktif setelah file hasil tersedia. "
            "Model ML/DL dan Chronos hanya dibandingkan jika hasilnya benar-benar ada — saat ini "
            "eksperimen Chronos gagal dieksekusi sehingga tidak ikut dalam grafik.",
            "outputs/results/cold_start_rmse.csv",
            ["Jalankan notebook (sel 11: efisiensi data) hingga selesai",
             "Salin results/cold_start_rmse.csv ke outputs/results/",
             "Skema: family · horizon · hari · RMSE (termasuk baris full-training)"],
            icon_name="gauge",
        )
        f = hub.figure("eval03_cold_start")
        if f:
            viz.fig_card(f, "<b>eval03_cold_start.png</b> — Cold start dari notebook (horizon 24 jam): Cold-ML vs Cold-DL "
                            "(model dilatih sendiri). Titik paling kanan = full training.")
        viz.note(
            "Catatan: perbandingan dengan Chronos pada kurva ini baru muncul bila hasil Chronos tersedia "
            "(status saat ini: Not Available / Failed Execution — kendala CUDA dan dependency torchao).",
            "warn", "alert", "Perbandingan Chronos",
        )
        return

    # ------------------------------------------------------------- aktif ---
    viz.section("Kurva Cold Start", f"RMSE test pada horizon {h_sel} jam (sumbu-x log)")
    d = cold[cold.horizon == h_sel].copy()
    if d.empty:
        viz.note("Tidak ada baris untuk horizon ini; pilih horizon lain.", "warn", "alert")
        return
    fig = go.Figure()
    for fam, g in d.groupby("family"):
        g = g.dropna(subset=["hari"]).sort_values("hari")
        is_cold = fam.startswith("Cold")
        fig.add_trace(go.Scatter(
            x=g["hari"], y=g["RMSE"], mode="lines+markers", name=fam,
            line=dict(color=FAM_COLORS.get(fam, PALETTE["accent"]), width=2.2 if is_cold else 1.6,
                      dash=None if is_cold else "dash"),
            marker=dict(size=8 if is_cold else 6),
            hovertemplate="%{x:,.0f} hari<br>RMSE %{y:.2f} MWh<extra>" + fam + "</extra>",
        ))
    if full_days:
        fig.add_vline(x=full_days, line_width=1, line_dash="dot", line_color="rgba(140,160,184,0.5)",
                      annotation_text=f"full ≈ {full_days} hari", annotation_position="top left",
                      annotation_font=dict(size=10.5, color="#8CA0B8"))
    fig.update_layout(height=420, xaxis_title="Hari data training terbaru (log)",
                      yaxis_title=f"RMSE test (h={h_sel}, MWh)",
                      xaxis=dict(type="log"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    st.plotly_chart(fig, use_container_width=True)

    # tabel nilai
    viz.section("Tabel Nilai", "RMSE per family × jumlah hari data")
    piv = d.pivot_table(index="family", columns="hari", values="RMSE")
    piv.columns = [f"≈{viz.fmt_num(c)} hari" if pd.notna(c) else "full" for c in piv.columns]
    st.dataframe(piv.round(2), use_container_width=True,
                 column_config={c: st.column_config.NumberColumn(c, format="%.2f") for c in piv.columns})
    viz.download_csv(d, f"cold_start_h{h_sel}.csv", key="dl_cs")

    # insight otomatis
    best_fam = d.groupby("family")["RMSE"].min().idxmin()
    full_row = d[d.family == best_fam].sort_values("hari")
    if len(full_row) > 1:
        f100 = full_row.iloc[-1]["RMSE"]
        m30 = full_row[full_row.hari <= 30]
        if len(m30):
            gain = (m30.iloc[0]["RMSE"] - f100) / m30.iloc[0]["RMSE"] * 100
            viz.note(
                f"Interpretasi otomatis: pada horizon {h_sel} jam, family terbaik adalah <b>{viz.esc(best_fam)}</b>. "
                f"Dari {viz.fmt_num(m30.iloc[0]['hari'])} hari data menuju full training, RMSE membaik sekitar "
                f"<b>{gain:.0f}%</b> ({m30.iloc[0]['RMSE']:.1f} → {f100:.1f} MWh); setelah ≈1 tahun data, "
                "perbaikan tambahan umumnya kecil.", "teal", "spark", "Bacaan cepat")
