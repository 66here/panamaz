"""Halaman Overview — executive dashboard."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import viz
from core.data_loader import get_hub
from core.settings import (ANOM_DATES, DATA_PERIOD, HORIZONS, PALETTE, SPLIT_FRAC,
                           TARGET_DESC, TOTAL_OBS)

DAY_NAMES = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]


def _series_fig(s: pd.Series, source_note: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(viz.line(s.index, s.values, "Beban aktual (per jam)", "#5B6E8C", width=1.0, showlegend=True))
    roll = s.rolling(24 * 7, min_periods=24).mean()
    fig.add_trace(viz.line(roll.index, roll.values, "Rata-rata 7 hari", PALETTE["accent"], width=2.0))
    fig.update_layout(
        height=400,
        yaxis_title="Beban (MWh)",
        xaxis=dict(rangeselector=dict(
            buttons=[
                dict(count=1, label="1 bulan", step="month", stepmode="backward"),
                dict(count=3, label="3 bulan", step="month", stepmode="backward"),
                dict(count=1, label="1 tahun", step="year", stepmode="backward"),
                dict(step="all", label="Semua"),
            ],
            bgcolor="rgba(17,28,46,0.7)", activecolor="rgba(32,211,194,0.25)",
            font=dict(size=11),
        )),
    )
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    return fig


def render():
    viz.set_current("overview")
    hub = get_hub()
    viz.page_header(
        "Overview", 
        "Ikhtisar eksekutif proyek peramalan beban listrik nasional Panama. "
        "Halaman ini merangkum dataset, pola beban, status pipeline hasil eksperimen, "
        "dan model dengan performa terbaik pada setiap horizon.",
        "Executive Summary",
        chips=[DATA_PERIOD, "Resolusi per jam", "Target: nat_demand (MWh)", f"Horizon {', '.join(map(str, HORIZONS))} jam"],
    )

    # ------------------------------------------------------------ KPI ---
    series, src_label, src_file = hub.actuals()
    n_obs = TOTAL_OBS
    reg = hub.get("regime_summary")
    reg_note = ""
    if reg is not None:
        n_obs = int(pd.to_numeric(reg["jumlah_jam"], errors="coerce").sum())
        reg_note = "dari rekap split notebook"
    kpis = [
        viz.kpi_card("Total Observations", viz.fmt_num(n_obs), reg_note or "jam pengamatan per jam", "database", "blue"),
        viz.kpi_card("Data Period", "Jan 2015 – Jun 2020", "3 Jan 2015 s.d. 27 Jun 2020", "calendar", ""),
        viz.kpi_card("Sampling Frequency", "Hourly", "1 jam, tanpa gap (divalidasi notebook)", "clock", "amber"),
        viz.kpi_card("Target Variable", "nat_demand", TARGET_DESC, "activity", "purple"),
        viz.kpi_card("Forecast Horizons", str(len(HORIZONS)), "1 · 6 · 24 · 72 jam ke depan", "target", "green"),
        viz.kpi_card("Model Families", "4", "Konvensional · ML · DL · Foundation", "layers", ""),
    ]
    viz.kpi_row(kpis)

    # ------------------------------------------- historical time series ---
    viz.section("Beban Listrik Nasional — Seri Waktu", src_label if src_label else "")
    if len(series) > 0:
        st.plotly_chart(_series_fig(series, src_label), use_container_width=True)
        cov = f"{series.index.min():%d %b %Y} – {series.index.max():%d %b %Y}"
        viz.note(
            f"Sumber interaktif: <b>{src_label}</b> (file <code>{src_file}</code>). "
            f"Cakupan data yang tersedia saat ini: <b>{cov}</b> ({viz.fmt_num(len(series))} jam). "
            "Muat file <code>results/data_with_regime.csv.gz</code> dari notebook untuk visualisasi periode penuh 2015–2020.",
            "teal", "info", "Catatan sumber data",
        )
        with st.expander("Gambar dokumentasi dari notebook — seri lengkap 2015–2020 (statis)", expanded=False):
            f = hub.figure("eda01_series")
            if f:
                viz.fig_card(f, "<b>eda01_series.png</b> — Beban listrik nasional Panama 2015–2020 (MWh). "
                                "Garis tipis: per jam; garis tebal: rata-rata 7 hari; area berwarna: regime validasi & test; titik merah: anomali berflag.")
            else:
                viz.empty_state("Gambar notebook tidak ditemukan", "Gambar statis hasil EDA notebook belum tersedia di folder outputs/figures.", "outputs/figures/eda01_series.png")
    else:
        viz.empty_state(
            "Data aktual belum tersedia",
            "Seri beban interaktif membutuhkan file dataset dari notebook. Tanpa file tersebut, dashboard tidak menampilkan grafik agar tidak menampilkan angka palsu.",
            "outputs/results/data_with_regime.csv.gz",
            ["Jalankan notebook Panama_forecasting_ADL_v2.ipynb (Run All)",
             "Salin folder results ke outputs/ pada proyek dashboard ini",
             "Dashboard akan otomatis membaca file saat halaman dimuat ulang"],
        )
        f = hub.figure("eda01_series")
        if f:
            viz.fig_card(f, "<b>eda01_series.png</b> — Gambar statis seri lengkap dari notebook (referensi).")

    # ------------------------------------------------------- pola beban ---
    viz.section("Pola Beban", "Rata-rata, sebaran, dan profil waktu")
    c1, c2, c3 = st.columns([1.1, 1.1, 1.1])
    with c1:
        st.markdown("**Distribusi beban**")
        if len(series) > 0:
            fig = viz.hist_fig(series.dropna().values, PALETTE["accent"], nbins=64,
                               mean=float(series.mean()), median=float(series.median()),
                               x_title="Beban (MWh)")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            viz.empty_state("Belum tersedia", "Distribusi beban dihitung dari data aktual.", "results/data_with_regime.csv.gz", icon_name="chart")
    with c2:
        st.markdown("**Rata-rata beban per jam**")
        prof = None
        eda = hub.get("eda_profile")
        if len(series) > 0:
            hourly = series.groupby(series.index.hour).mean()
            prof = hourly
            src2 = f"aktual ({src_label.lower()})"
        elif eda is not None:
            piv = eda.pivot_table(index="hour", values="mean_load", aggfunc="mean")
            prof = piv["mean_load"]
            src2 = "pola EDA notebook (periode penuh)"
        if prof is not None:
            peak_h = int(prof.idxmax())
            colors = [PALETTE["accent"] if h == peak_h else "#3D5273" for h in prof.index]
            fig = viz.bar_fig([f"{h:02d}" for h in prof.index], prof.values, height=300,
                              x_title="Jam", y_title="Rata-rata (MWh)", color_list=colors, text_vals=False)
            fig.update_layout(title=f"Jam puncak: {peak_h:02d}:00")
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Sumber: {src2}")
        else:
            viz.empty_state("Belum tersedia", "Pola per jam dihitung dari data aktual atau tabel EDA.", "results/eda_profile_dow_hour.csv", icon_name="clock")
    with c3:
        st.markdown("**Rata-rata beban per hari (seminggu)**")
        if len(series) > 0:
            dow_mean = series.groupby(series.index.dayofweek).mean()
            dow_mean.index = [DAY_NAMES[i] for i in dow_mean.index]
            wk_max = dow_mean.idxmax()
            colors = [PALETTE["blue"] if d == wk_max else "#3D5273" for d in dow_mean.index]
            fig = viz.bar_fig(list(dow_mean.index), dow_mean.values, height=300,
                              x_title="Hari", y_title="Rata-rata (MWh)", color_list=colors, text_vals=False)
            fig.update_layout(title=f"Tertinggi: {wk_max}")
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Sumber: aktual ({src_label.lower()})")
        else:
            viz.empty_state("Belum tersedia", "Pola per hari dihitung dari data aktual.", "results/data_with_regime.csv.gz", icon_name="calendar")

    # ------------------------------------------- model terbaik / horizon ---
    viz.section("Model dengan RMSE Terbaik per Horizon", "regime test — dihitung dari metrics_all.csv")
    best = hub.best_per_horizon()
    if best is not None and len(best):
        h1, h2 = st.columns([1.35, 1])
        with h1:
            disp = best.copy()
            disp["fset"] = disp["fset"].astype(str)
            st.dataframe(
                disp,
                hide_index=True, use_container_width=True,
                column_config={
                    "horizon": st.column_config.NumberColumn("Horizon (jam)", format="%d"),
                    "RMSE": st.column_config.NumberColumn("RMSE (MWh)", format="%.2f"),
                    "MAE": st.column_config.NumberColumn("MAE (MWh)", format="%.2f"),
                    "MAPE": st.column_config.NumberColumn("MAPE (%)", format="%.2f"),
                    "sMAPE": st.column_config.NumberColumn("sMAPE (%)", format="%.2f"),
                },
            )
        with h2:
            fig = go.Figure(go.Bar(
                x=best["horizon"].astype(str), y=best["RMSE"],
                marker=dict(color=PALETTE["accent"], opacity=0.85),
                text=[f"{v:.1f}" for v in best["RMSE"]], textposition="outside",
                textfont=dict(size=10.5, color="#B9C6D8"),
                hovertemplate="h=%{x} jam<br>RMSE %{y:.2f} MWh<extra>" + "%{customdata}</extra>",
                customdata=best["model_id"].astype(str),
            ))
            fig.update_layout(height=290, xaxis_title="Horizon (jam)", yaxis_title="RMSE (MWh)")
            st.plotly_chart(fig, use_container_width=True)
    else:
        viz.empty_state(
            "Peringkat model belum tersedia",
            "Ringkasan model terbaik per horizon akan muncul setelah file metrik hasil eksekusi notebook tersedia.",
            "outputs/results/metrics_all.csv",
            ["Jalankan notebook utama hingga selesai (Run All, FAST_DEBUG=False)",
             "Salin results/metrics_all.csv ke outputs/results/",
             "Halaman Benchmark juga menampilkan peringkat lengkap"],
            icon_name="trophy",
        )

    # ------------------------------------------------------------ insight ---
    viz.section("Insight Otomatis", "dihitung dari data hasil yang benar-benar tersedia")
    insights: list[str] = []
    if len(series) > 0:
        peak_h = int(series.groupby(series.index.hour).mean().idxmax())
        low_h = int(series.groupby(series.index.hour).mean().idxmin())
        insights.append((
            "clock", "Pola harian",
            f"Rata-rata beban mencapai puncak pada jam <b>{peak_h:02d}:00</b> dan terendah pada jam "
            f"<b>{low_h:02d}:00</b> (berdasarkan aktual: {src_label.lower()})."))
        wk = series[series.index.dayofweek < 5].mean()
        we = series[series.index.dayofweek >= 5].mean()
        d_we = (we - wk) / wk * 100
        insights.append((
            "calendar", "Akhir pekan vs hari kerja",
            f"Rata-rata akhir pekan <b>{d_we:+.1f}%</b> relatif terhadap hari kerja "
            f"({viz.fmt_num(we)} vs {viz.fmt_num(wk)} MWh)."))
    adet = hub.get("anomalies_detected")
    if adet is not None:
        flagged = adet[adet["diberi_flag"] == True] if "diberi_flag" in adet else adet
        n_h = int(pd.to_numeric(flagged.get("jam_terdeteksi", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        dates = ", ".join(str(d) for d in flagged.get("tanggal", pd.Series(dtype=str)).tolist())
        insights.append((
            "alert", "Anomali / pemadaman",
            f"<b>{len(flagged)} tanggal</b> terdeteksi dengan total ≈{n_h} jam terdampak (pemadaman berflag: "
            f"<b>{dates}</b>). Jam anomali diberi flag dan digunakan sebagai fitur pada set F1 dst."))
    pca = hub.get("pca_loadings")
    if pca is not None:
        pc1 = pca.set_index("variabel")["wpc1"].astype(float)
        top = pc1.abs().sort_values(ascending=False).head(3)
        insights.append((
            "sun", "Cuaca — komponen PCA utama",
            f"Komponen <b>wpc1</b> paling ditentukan oleh {', '.join(f'<b>{v}</b>' for v in top.index)} "
            f"(loading {', '.join(f'{x:+.2f}' for x in pc1[top.index])})."))
    if reg is not None:
        rows = reg.set_index("regime")
        try:
            tr = rows.loc["train"]; te = rows.loc["test"]; va = rows.loc["val"]
            insights.append((
                "layers", "Split kronologis",
                f"Train <b>{int(tr['jumlah_jam']):,}</b> jam ({tr['mulai']:%d %b %Y} – {tr['akhir']:%d %b %Y}) · "
                f"Val <b>{int(va['jumlah_jam']):,}</b> jam · Test <b>{int(te['jumlah_jam']):,}</b> jam "
                f"({te['mulai']:%d %b %Y} – {te['akhir']:%d %b %Y}); proporsi 70/10/20 tanpa shuffle."))
        except Exception:
            pass
    n_avail, n_ess, n_tot = hub.counts()
    from core.data_loader import REGISTRY as _REG
    insights.append((
        "database", "Status pipeline hasil",
        f"<b>{n_avail} dari {len(_REG)}</b> file hasil terbaca (file utama: {n_ess}/{n_tot}). "
        f"Chronos-2: {viz.badge('Not Available / Failed Execution', 'warn')} — eksekusi gagal (kendala CUDA + torchao); "
        "benchmark berfokus pada kelompok Konvensional, ML, dan DL."))
    cols = st.columns(3)
    items = [(i, t, m) for i, t, m in insights]
    for j, (ic, t, m) in enumerate(items):
        with cols[j % 3]:
            st.markdown(
                f'<div class="kpi-card" style="min-height:118px">'
                f'<div class="kpi-top"><div class="kpi-label">{esc_title(t)}</div>'
                f'<div class="kpi-icon {tone_map(ic)}">{viz.icon(ic, 17)}</div></div>'
                f'<div class="kpi-sub" style="font-size:.82rem; color:#C9D5E5; margin-top:2px">{m}</div></div>',
                unsafe_allow_html=True,
            )

    # ----------------------------------------------------- data status ----
    viz.section("Status Sumber Data", "sistem pengecekan file outputs — dashboard tetap berjalan meski ada file yang belum ada")
    stat = hub.status_table()
    show = stat.copy()
    show["Status"] = show["Status"].map(
        lambda s: f'<span class="badge ok">● {s}</span>' if s == "Tersedia" else '<span class="badge muted">○ Belum ada</span>'
    )
    st.markdown(
        show.to_html(escape=False, index=False, classes="pei-table")
        .replace('<table class="dataframe pei-table"', '<table class="pei-table"'),
        unsafe_allow_html=True,
    )
    st.markdown("""<style>
.pei-table {width:100%; border-collapse:collapse; font-size:.8rem;}
.pei-table th {text-align:left; padding:8px 12px; color:#8CA0B8; font-weight:600; border-bottom:1px solid rgba(148,163,184,.18); text-transform:uppercase; font-size:.68rem; letter-spacing:.08em;}
.pei-table td {padding:8px 12px; color:#C9D5E5; border-bottom:1px solid rgba(148,163,184,.08);}
.pei-table tr:hover td {background:rgba(32,211,194,.03);}
</style>""", unsafe_allow_html=True)

    # ------------------------------------------------------- ringkasan ---
    rk = hub.get("ringkasan_txt")
    if rk:
        with st.expander("Ringkasan eksekusi notebook (results/ringkasan.txt)"):
            st.code(rk, language="text")


def esc_title(t: str) -> str:
    import html as _h
    return _h.escape(t)


def tone_map(ic: str) -> str:
    return {"clock": "blue", "calendar": "", "alert": "red", "sun": "amber",
            "layers": "purple", "database": "green", "chart": "blue"}.get(ic, "")
