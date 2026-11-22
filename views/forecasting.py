"""Halaman Forecasting Analysis — aktual vs prediksi, residual, distribusi error."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import viz
from core.data_loader import get_hub
from core.settings import (FEATURE_SETS, HORIZONS, METRIC_DEFS, MODEL_ROSTER,
                           PALETTE, TARGET_DESC)

FAM_TONE = {"Konvensional": "#FFB65C", "ML": "#4C8DFF", "DL": "#A78BFA"}
SERIES_PALETTE = ["#20D3C2", "#4C8DFF", "#FFB65C", "#A78BFA", "#FF6B6B", "#3DDC97", "#F472B6"]


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    e = y - p
    ae = np.abs(e)
    return {
        "RMSE": float(np.sqrt(np.mean(e ** 2))),
        "MAE": float(ae.mean()),
        "MAPE": float(np.mean(ae / np.abs(y) * 100)),
        "sMAPE": float(np.mean(2 * ae / (np.abs(y) + np.abs(p)) * 100)),
        "n": len(y),
        "bias": float(e.mean()),
    }


def render():
    viz.set_current("forecasting-analysis")
    hub = get_hub()
    viz.page_header(
        "Forecasting Analysis",
        "Analisis hasil prediksi pada regime evaluasi: aktual vs prediksi, residual, "
        "sebaran error, dan perbandingan antarmodel. Seluruh grafik mengikuti filter "
        "horizon, model, set fitur, dan rentang tanggal yang dipilih.",
        "Prediction & Error Analysis",
        chips=[f"Horizon {', '.join(map(str, HORIZONS))} jam", "Set fitur F0–F4", "Origin evaluasi tiap 3 jam"],
    )

    pred = hub.get("predictions_eval")

    # ------------------------------------------------------------- filter ---
    non_chronos = [m for m in MODEL_ROSTER if m["family"] != "Foundation"]
    st.markdown('<div class="pei-sec"><div class="bar"></div><h3>Filter</h3>'
                f'<span class="hint">{viz.tip("eval_note")}</span></div>', unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns([0.9, 1.5, 1.2, 1.1])
    with f1:
        horizon = st.pills("Horizon (jam)", HORIZONS, default=24, key="fc_h")
        if horizon is None:
            horizon = 24
    with f2:
        if pred is not None:
            avail_models = sorted(pred["model_id"].astype(str).unique())
            model_sel = st.multiselect("Model (model_id|fset)", avail_models,
                                       default=[m for m in avail_models if not m.startswith("Chronos")][:3],
                                       max_selections=6, key="fc_m")
        else:
            model_sel = st.multiselect(
                "Model", [m["label"] for m in non_chronos],
                default=[non_chronos[1]["label"]], disabled=True,
                help="Aktif otomatis setelah file prediksi tersedia", key="fc_m_d")
    with f3:
        if pred is not None:
            fsets = sorted(pred["model_id"].astype(str).str.split("|").str[1].dropna().unique())
            fset_sel = st.multiselect("Set fitur", fsets, default=fsets, key="fc_f")
        else:
            fset_sel = st.multiselect("Set fitur", list(FEATURE_SETS), default=["F1"], disabled=True, key="fc_f_d")
    with f4:
        if pred is not None:
            regimes = ["test", "val"]
            reg_sel = st.segmented_control("Regime", regimes, selection_mode="single", default="test", key="fc_r")
            reg_sel = reg_sel or "test"
        else:
            reg_sel = st.segmented_control("Regime", ["test", "val"], selection_mode="single",
                                           default="test", disabled=True, key="fc_r_d")
            reg_sel = reg_sel or "test"

    if pred is not None:
        dmin, dmax = pred["target_time"].min(), pred["target_time"].max()
        d0_default = dmin.date()
        d1_default = min(dmax.date(), (dmin + pd.Timedelta(days=60)).date())
        drange = st.date_input("Rentang tanggal evaluasi", value=(d0_default, d1_default),
                               min_value=dmin.date(), max_value=dmax.date(), key="fc_d")
    else:
        drange = None

    # -------------------------------------------------- gerbang ketersediaan ---
    if pred is None:
        viz.section("Hasil Prediksi", "aktif setelah file tersedia")
        viz.empty_state(
            "File prediksi belum tersedia",
            "Halaman ini membutuhkan file prediksi keluaran notebook utama. Saat ini dashboard tidak "
            "menampilkan grafik maupun metrik agar tidak menampilkan angka yang tidak benar. "
            "Di bawah tersedia contoh hasil (gambar statis) dari run notebook sebelumnya.",
            "outputs/results/predictions_eval.csv.gz",
            ["Jalankan notebook utama hingga selesai (Run All, FAST_DEBUG = False)",
             "Salin folder results (atau file predictions_eval.csv.gz) ke outputs/results/",
             "Muat ulang halaman — filter dan grafik akan aktif otomatis",
             "Skema yang diharapkan: origin · horizon · regime · model_id · family · y_true · y_pred · target_time"],
            icon_name="chart",
        )
        viz.section("Contoh Hasil dari Run Notebook (statis)")
        c1, c2, c3 = st.columns(3)
        with c1:
            f = hub.figure("eval05_contoh_sebelum_pandemi")
            if f:
                viz.fig_card(f, "<b>eval05_contoh_sebelum_pandemi.png</b> — Aktual vs prediksi model terbaik, horizon 24 jam, Desember 2019 (sebelum pandemi).")
        with c2:
            f = hub.figure("eval06_contoh_pandemi")
            if f:
                viz.fig_card(f, "<b>eval06_contoh_pandemi.png</b> — Aktual vs prediksi, horizon 24 jam, April 2020 (periode pandemi COVID-19).")
        with c3:
            f = hub.figure("eval08_residual")
            if f:
                viz.fig_card(f, "<b>eval08_residual.png</b> — Analisis residual model terbaik: MAE per jam, RMSE per bulan (Mar–Jun 2020 = pandemi), MAE per tipe hari.")
        viz.section("Definisi Metrik")
        with st.expander("RMSE · MAE · MAPE · sMAPE"):
            st.markdown(
                f"- **RMSE** {viz.tip('RMSE')}\n\n- **MAE** {viz.tip('MAE')}\n\n"
                f"- **MAPE** {viz.tip('MAPE')}\n\n- **sMAPE** {viz.tip('sMAPE')}\n\n"
                f"- **Catatan evaluasi** {viz.tip('eval_note')}",
                unsafe_allow_html=True,
            )
        return

    # ------------------------------------------------------------- aktif ---
    dfp = pred[pred["regime"].astype(str) == reg_sel].copy()
    if model_sel:
        dfp = dfp[dfp["model_id"].astype(str).isin(model_sel)]
    if fset_sel:
        dfp = dfp[dfp["model_id"].astype(str).str.split("|").str[1].isin(fset_sel)]
    dfp = dfp[dfp["horizon"] == int(horizon)]
    if isinstance(drange, tuple) and len(drange) == 2:
        d0, d1 = pd.Timestamp(drange[0]), pd.Timestamp(drange[1]) + pd.Timedelta(hours=23)
        dfp = dfp[(dfp["target_time"] >= d0) & (dfp["target_time"] <= d1)]

    if dfp.empty:
        viz.note("Tidak ada baris prediksi untuk kombinasi filter saat ini. Perluas rentang tanggal, pilih lebih banyak model, atau ubah horizon.", "warn", "alert", "Filter terlalu sempit")
        return

    # KPI metrik per model terpilih
    models_in = sorted(dfp["model_id"].astype(str).unique())
    per_model = {m: _metrics(dfp[dfp.model_id == m]["y_true"].values,
                             dfp[dfp.model_id == m]["y_pred"].values) for m in models_in}
    agg = _metrics(dfp["y_true"].values, dfp["y_pred"].values)
    viz.kpi_row([
        viz.kpi_card("RMSE", f"{agg['RMSE']:.2f}", f"MWh · {len(models_in)} model · n={viz.fmt_num(agg['n'])}", "activity", "blue", tip_text=METRIC_DEFS["RMSE"]),
        viz.kpi_card("MAE", f"{agg['MAE']:.2f}", "MWh (rata-rata kesalahan absolut)", "chart", "", tip_text=METRIC_DEFS["MAE"]),
        viz.kpi_card("MAPE", f"{agg['MAPE']:.2f}", "persen (relatif terhadap aktual)", "trend", "amber", tip_text=METRIC_DEFS["MAPE"]),
        viz.kpi_card("sMAPE", f"{agg['sMAPE']:.2f}", "persen (versi simetris)", "spark", "purple", tip_text=METRIC_DEFS["sMAPE"]),
    ])
    viz.note(
        "KPI di atas dihitung langsung dari baris prediksi yang lolos filter (regime "
        f"<b>{reg_sel}</b>, horizon <b>{horizon} jam</b>, {viz.fmt_num(agg['n'])} titik). "
        f"{viz.tip('RMSE')}{viz.tip('MAE')}{viz.tip('MAPE')}{viz.tip('sMAPE')} Arahkan kursor ke ikon untuk definisi metrik.",
        "teal", "info", "Cara baca",
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Aktual vs Prediksi", "Residual", "Scatter", "Distribusi Error", "Perbandingan Model"])
    with tab1:
        fig = go.Figure()
        mids = models_in
        for i, m in enumerate(mids):
            g = dfp[dfp.model_id == m].sort_values("target_time")
            fig.add_trace(viz.line(g["target_time"], g["y_pred"], m, SERIES_PALETTE[i % len(SERIES_PALETTE)], width=1.5))
        g0 = dfp[dfp.model_id == mids[0]].sort_values("target_time")
        fig.add_trace(viz.line(g0["target_time"], g0["y_true"], "Aktual", "#E7EEF7", width=1.8))
        fig.update_layout(height=420, yaxis_title="Beban (MWh)", legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        st.plotly_chart(fig, use_container_width=True)
        viz.download_csv(dfp.assign(err=dfp.y_true - dfp.y_pred),
                         f"prediksi_{reg_sel}_h{horizon}.csv", key="dl_fc")
    with tab2:
        fig = go.Figure()
        for i, m in enumerate(mids):
            g = dfp[dfp.model_id == m].sort_values("target_time")
            err = g["y_true"] - g["y_pred"]
            fig.add_trace(go.Bar(x=g["target_time"], y=err, name=m,
                                 marker_color=SERIES_PALETTE[i % len(SERIES_PALETTE)], opacity=0.75,
                                 hovertemplate="%{x}<br>error %{y:.1f} MWh<extra>" + m + "</extra>"))
        fig.add_hline(y=0, line_color="rgba(231,238,247,0.5)", line_width=1)
        fig.update_layout(height=380, yaxis_title="Error (y_true − y_pred, MWh)", barmode="overlay")
        st.plotly_chart(fig, use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            g = dfp.sort_values("target_time")
            hour_err = g.assign(err=g.y_true - g.y_pred, hr=g.target_time.dt.hour).groupby("hr")["err"].mean()
            fig2 = viz.bar_fig([f"{h:02d}" for h in hour_err.index], hour_err.values,
                               PALETTE["blue"], height=300, x_title="Jam target", y_title="Bias rata-rata (MWh)", text_vals=False)
            fig2.update_layout(title="Bias rata-rata per jam (nilai positif = prediksi terlalu rendah)")
            st.plotly_chart(fig2, use_container_width=True)
        with c2:
            st.markdown("**Ringkasan residual per model**")
            rr = pd.DataFrame(per_model).T
            st.dataframe(rr[["RMSE", "MAE", "bias", "n"]].rename(columns={"bias": "bias (MWh)", "n": "n obs"}).round(2),
                         use_container_width=True)
    with tab3:
        fig = go.Figure()
        for i, m in enumerate(mids):
            g = dfp[dfp.model_id == m]
            fig.add_trace(go.Scatter(
                x=g["y_true"], y=g["y_pred"], mode="markers", name=m,
                marker=dict(size=4.5, color=SERIES_PALETTE[i % len(SERIES_PALETTE)], opacity=0.45),
                hovertemplate="aktual %{x:,.0f}<br>prediksi %{y:,.0f}<extra>" + m + "</extra>"))
        lo = float(min(dfp.y_true.min(), dfp.y_pred.min())) * 0.97
        hi = float(max(dfp.y_true.max(), dfp.y_pred.max())) * 1.03
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="y = x",
                                 line=dict(color="rgba(231,238,247,0.55)", width=1.2, dash="dash"),
                                 hoverinfo="skip", showlegend=True))
        fig.update_layout(height=420, x_title="Aktual (MWh)", y_title="Prediksi (MWh)", hover_x="closest")
        st.plotly_chart(fig, use_container_width=True)
    with tab4:
        err_all = (dfp["y_true"] - dfp["y_pred"]).dropna()
        fig = viz.hist_fig(err_all.values, PALETTE["blue"], nbins=70, mean=float(err_all.mean()),
                           median=float(err_all.median()), x_title="Error (MWh)", y_title="Frekuensi")
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Bias rata-rata {err_all.mean():+.2f} MWh · std {err_all.std():.2f} MWh · "
                   f"p95 |error| {err_all.abs().quantile(0.95):.1f} MWh")
    with tab5:
        st.markdown("**Prediksi antarmodel pada 3 hari terakhir rentang terpilih**")
        win = dfp["target_time"].max() - pd.Timedelta(days=3)
        dw = dfp[dfp.target_time >= win]
        fig = go.Figure()
        for i, m in enumerate(mids):
            g = dw[dw.model_id == m].sort_values("target_time")
            fig.add_trace(viz.line(g["target_time"], g["y_pred"], m, SERIES_PALETTE[i % len(SERIES_PALETTE)], width=1.6))
        g0 = dw[dw.model_id == mids[0]].sort_values("target_time")
        fig.add_trace(viz.line(g0["target_time"], g0["y_true"], "Aktual", "#E7EEF7", width=2.0))
        fig.update_layout(height=380, yaxis_title="Beban (MWh)", legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Metrik per model (mengikuti seluruh filter)**")
        cmp_df = pd.DataFrame(per_model).T[["RMSE", "MAE", "MAPE", "sMAPE", "n"]]
        st.dataframe(cmp_df.round(2), use_container_width=True,
                     column_config={"RMSE": st.column_config.NumberColumn("RMSE (MWh)", format="%.2f"),
                                    "MAE": st.column_config.NumberColumn("MAE (MWh)", format="%.2f"),
                                    "MAPE": st.column_config.NumberColumn("MAPE (%)", format="%.2f"),
                                    "sMAPE": st.column_config.NumberColumn("sMAPE (%)", format="%.2f")})
