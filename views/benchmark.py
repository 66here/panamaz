"""Halaman Model Benchmark — ranking, perbandingan kelompok, heatmap, ablasi fitur.

PENTING: sesuai hasil eksperimen saat ini, Chronos-2 (Small & Base) GAGAL dieksekusi
(kendala CUDA + dependency torchao). Status ditampilkan apa adanya sebagai
"Not Available / Failed Execution" tanpa angka fiktif.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import viz
from core.data_loader import get_hub
from core.settings import (CHRONOS_NOTE, CHRONOS_PROTOCOL, FEATURE_SETS, HORIZONS,
                           MODEL_ROSTER, PALETTE)

MAIN_FAMILIES = ["Konvensional", "ML", "DL"]


def _chronos_banner(hub):
    on = hub.chronos_on()
    if not on:
        viz.note(
            f"<b>Chronos-2 Small &amp; Chronos-2 Base — Not Available / Failed Execution.</b><br>"
            f"{CHRONOS_NOTE}<br><br>"
            "<b>Protokol yang dirancang</b> (untuk transparansi apa yang akan dibandingkan begitu run berhasil):<br>"
            + "".join(f"<br>• <b>{k}:</b> {v}" for k, v in CHRONOS_PROTOCOL.items())
            + "<br><br>Karena itu, seluruh ranking, heatmap, dan grafik di halaman ini hanya memuat kelompok "
            "<b>Konvensional, ML, dan DL</b>. Tidak ada metrik Chronos yang ditampilkan atau diimprovisasi.",
            "warn", "alert", "Status eksperimen Chronos-2",
        )
    else:
        viz.note("Chronos-2 aktif: hasil zero-shot/fine-tuned terbaca dari file hasil.", "ok", "check", "Status eksperimen Chronos-2")


def render():
    viz.set_current("model-benchmark")
    hub = get_hub()
    viz.page_header(
        "Model Benchmark",
        "Perbandingan empat kelompok model — konvensional (SARIMA, ETS, ARIMAX), machine learning "
        "(Random Forest, XGBoost, LightGBM), deep learning (LSTM, GRU, CNN-1D dilated), dan model fondasi "
        "(Chronos-2 Small/Base) — berdasarkan RMSE, MAE, serta ablasi set fitur F0–F4.",
        "Model Comparison",
        chips=["SARIMA · ETS · ARIMAX", "RF · XGBoost · LightGBM", "LSTM · GRU · CNN-1D", "Chronos-2 (gagal dieksekusi)"],
    )

    _chronos_banner(hub)

    m = hub.get("metrics_all")
    horizon = st.pills("Horizon pembanding (jam)", HORIZONS, default=HORIZONS[2], key="bm_h") or HORIZONS[2]

    if m is None:
        # ------------------------------------------------ fallback figure + empty state
        viz.section("Peringkat Model", "regime test — dihitung dari metrics_all.csv")
        viz.empty_state(
            "Tabel metrik belum tersedia",
            "Peringkat, heatmap, dan tabel ranking akan aktif otomatis setelah hasil eksekusi notebook dimuat ke folder outputs. "
            "Dashboard tidak mengarang angka peringkat.",
            "outputs/results/metrics_all.csv",
            ["Jalankan notebook utama (Run All, FAST_DEBUG=False)",
             "Salin results/metrics_all.csv (dan ablation_fset_rmse.csv) ke outputs/results/",
             "Semua grafik halaman ini otomatis terisi dari file tersebut"],
            icon_name="trophy",
        )
        viz.section("Gambar Perbandingan dari Run Notebook (statis)")
        c1, c2 = st.columns(2)
        with c1:
            f = hub.figure("eval01_rmse_horizon")
            if f:
                viz.fig_card(f, "<b>eval01_rmse_horizon.png</b> — RMSE model terbaik tiap kelompok (test) per horizon. "
                                "Semakin landai kurvanya, semakin tahan model terhadap horizon panjang.")
        with c2:
            f = hub.figure("eval02_ablasi_fitur")
            if f:
                viz.fig_card(f, "<b>eval02_ablasi_fitur.png</b> — Ablasi set fitur LightGBM: perbaikan RMSE (%) vs F0 per set fitur dan horizon.")
        viz.section("Roster Model", "empat kelompok sesuai desain eksperimen")
        _roster_table()
        viz.section("Ablasi Set Fitur F0–F4")
        _fset_help()
        return

    # ---------------------------------------------------------------- aktif ---
    lb = m[(m.regime_group == "test")].copy()
    if not hub.chronos_on():
        lb = lb[~lb.model_id.astype(str).str.startswith("Chronos")]
    lb = lb[lb.family.isin(MAIN_FAMILIES + (["Foundation"] if hub.chronos_on() else []))]

    # kartu model terbaik per horizon
    viz.section("Model Terbaik per Horizon", "ditonjolkan — RMSE terendah pada regime test")
    best = lb.loc[lb.groupby("horizon")["RMSE"].idxmin()].sort_values("horizon")
    cols = st.columns(len(best))
    for i, (_, r) in enumerate(best.iterrows()):
        with cols[i]:
            st.markdown(
                f'<div class="kpi-card">'
                f'<div class="kpi-top"><div class="kpi-label">HORIZON {int(r["horizon"])} JAM</div>'
                f'<div class="kpi-icon" style="background:rgba(255,182,92,.13);color:{PALETTE["amber"]}">{viz.icon("trophy", 17)}</div></div>'
                f'<div class="kpi-value" style="font-size:1.15rem">{viz.esc(str(r["model_id"]))}</div>'
                f'<div class="kpi-sub">{viz.esc(str(r["family"]))} · RMSE <b style="color:{PALETTE["accent"]}">{r["RMSE"]:.2f}</b> MWh · '
                f'MAE {r["MAE"]:.2f}</div></div>',
                unsafe_allow_html=True,
            )

    # ranking table (sortable)
    viz.section("Tabel Ranking", f"regime test · horizon {horizon} jam — klik header untuk mengurutkan")
    t = lb[lb.horizon == horizon].copy()
    if t.empty:
        viz.empty_state("Tidak ada baris untuk horizon ini", "Pilih horizon lain.", icon_name="table")
    else:
        t = t.sort_values("RMSE")
        t.insert(0, "rank", range(1, len(t) + 1))
        st.dataframe(
            t[["rank", "model_id", "family", "model", "fset", "RMSE", "MAE", "MAPE", "sMAPE", "n"]],
            hide_index=True, use_container_width=True, height=380,
            column_config={
                "rank": st.column_config.NumberColumn("#", width="small"),
                "RMSE": st.column_config.NumberColumn("RMSE (MWh)", format="%.2f"),
                "MAE": st.column_config.NumberColumn("MAE (MWh)", format="%.2f"),
                "MAPE": st.column_config.NumberColumn("MAPE (%)", format="%.2f"),
                "sMAPE": st.column_config.NumberColumn("sMAPE (%)", format="%.2f"),
                "n": st.column_config.NumberColumn("n obs", format="%d"),
            },
        )
        viz.download_csv(t, f"ranking_test_h{horizon}.csv", key="dl_bm")

    # grouped bar MAE & RMSE
    viz.section("Perbandingan MAE & RMSE", "grouped bar — semua model per horizon")
    piv_mae = lb.pivot_table(index="model_id", columns="horizon", values="MAE")
    piv_rmse = lb.pivot_table(index="model_id", columns="horizon", values="RMSE")
    if not piv_rmse.empty:
        fig = go.Figure()
        for h in HORIZONS:
            if h in piv_rmse.columns:
                fig.add_trace(go.Bar(
                    name=f"RMSE h={h}", x=list(piv_rmse.index), y=piv_rmse[h].values,
                    marker_color=PALETTE["accent"] if h == 1 else (PALETTE["blue"] if h == 6 else (PALETTE["amber"] if h == 24 else PALETTE["purple"])),
                    opacity=0.88, hovertemplate="%{x}<br>RMSE h=" + str(h) + ": %{y:.2f}<extra></extra>",
                ))
        for h in HORIZONS:
            if h in piv_mae.columns:
                fig.add_trace(go.Bar(
                    name=f"MAE h={h}", x=list(piv_mae.index), y=piv_mae[h].values,
                    marker_color=["#0E6E63" if h == 1 else "#2C5A9E" if h == 6 else "#8A5B22" if h == 24 else "#5A4494"][0],
                    opacity=0.88, visible="legendonly",
                    hovertemplate="%{x}<br>MAE h=" + str(h) + ": %{y:.2f}<extra></extra>",
                ))
        fig.update_layout(barmode="group", height=430, xaxis_title="model_id",
                          yaxis_title="Metrik (MWh)",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("MAE ditampilkan sebagai seri sekunder (klik legenda untuk mengaktifkan). Semakin rendah semakin baik.")

    # heatmap model x horizon
    viz.section("Heatmap Performa", "RMSE model × horizon (test) — model terbaik tiap horizon diberi tepi teal")
    if not piv_rmse.empty:
        z = piv_rmse.sort_values(HORIZONS[2] if HORIZONS[2] in piv_rmse.columns else piv_rmse.columns[0])
        fig = viz.heatmap_fig(z, height=430, colorscale="Tealgrn", fmt=".1f",
                              x_title="Horizon (jam)", y_title="model_id")
        best_idx = {r["model_id"]: h for h, r in best.set_index("horizon").iterrows()}
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            "".join(f'<span class="badge teal" style="margin:2px 6px 2px 0">h={h} → {viz.esc(mid)}</span> '
                    for h, mid in best_idx.items()),
            unsafe_allow_html=True,
        )

    # ablasi fitur
    viz.section("Perbandingan Set Fitur F0–F4", "model ML — RMSE per set fitur dan horizon")
    _fset_help()
    ab = hub.get("ablation_fset")
    if ab is not None:
        mlab = lb["model_id"].astype(str)
        try:
            imp = pd.DataFrame(np.nan, index=ab.index, columns=ab.columns, dtype=float)
            for mdl in ab.index.get_level_values(0).unique():
                if (mdl, "F0") in ab.index:
                    f0 = ab.loc[(mdl, "F0")].astype(float).values
                    imp.loc[mdl] = ((f0 - ab.loc[mdl].astype(float).values) / f0 * 100)
            fig = viz.heatmap_fig(imp, height=360, colorscale="RdYlGn", fmt=".1f",
                                  x_title="Horizon (jam)", y_title="model · set fitur")
            fig.update_layout(title="Perbaikan RMSE (%) dibanding F0 — positif = lebih baik")
            st.plotly_chart(fig, use_container_width=True)
            viz.download_csv(imp.reset_index(), "ablation_fset_improvement.csv", key="dl_abl")
        except Exception:
            st.dataframe(ab, use_container_width=True)
    else:
        viz.empty_state(
            "Tabel ablasi belum tersedia",
            "Perbandingan interaktif F0–F4 aktif setelah file ablasi tersedia. Gambar statis di bawah dari notebook.",
            "outputs/results/ablation_fset_rmse.csv", icon_name="layers",
        )
        f = hub.figure("eval02_ablasi_fitur")
        if f:
            viz.fig_card(f, "<b>eval02_ablasi_fitur.png</b> — Perbaikan RMSE (%) vs F0 (LightGBM) per set fitur dan horizon.")

    # Chronos vs trained (terkunci status)
    viz.section("Chronos-2 vs Model Dilatih Sendiri", "tertunda status eksekusi")
    cmp_df = hub.get("chronos_vs_trained") if hub.chronos_on() else None
    if cmp_df is not None:
        st.dataframe(cmp_df, hide_index=True, use_container_width=True)
    else:
        viz.empty_state(
            "Perbandingan Chronos belum dapat ditampilkan",
            CHRONOS_NOTE + " File perbandingan (chronos_vs_trained.csv) hanya dimuat setelah eksperimen Chronos berhasil.",
            "outputs/results/chronos_vs_trained.csv", icon_name="zap-off",
        )


def _roster_table():
    rows = []
    for mr in MODEL_ROSTER:
        rows.append({
            "Kelompok": mr["family"], "Model": mr["label"], "Deskripsi": mr["desc"],
            "Status": ("Not Available / Failed Execution" if mr["family"] == "Foundation" else "Hasil menunggu file metrics_all.csv"),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _fset_help():
    st.markdown(
        "".join(f'<span class="pei-chip neutral" style="margin:2px 8px 2px 0"><b>{k}</b> · {viz.esc(v.split(" — ")[0])}</span> '
                for k, v in FEATURE_SETS.items()),
        unsafe_allow_html=True,
    )
    st.caption("F0 = riwayat beban · F1 = + kalender · F2 = + cuaca Tocumen · F3 = + cuaca 3 kota · F4 = + PCA cuaca (95% varians)")
