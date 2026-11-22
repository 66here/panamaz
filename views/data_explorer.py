"""Halaman Data Explorer — eksplorasi dataset, cuaca 3 kota, anomali, PCA."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import viz
from core.data_loader import get_hub
from core.settings import CITIES, PALETTE, TARGET_DESC, WEATHER_VARS

PALETTE_VAR = {
    "nat_demand": "#20D3C2", "T2M": "#FFB65C", "QV2M": "#4C8DFF",
    "TQL": "#3DDC97", "W2M": "#A78BFA", "holiday": "#FF6B6B", "school": "#8CA0B8",
}


def _var_catalog() -> pd.DataFrame:
    rows = [
        {"Variabel": "nat_demand", "Kelompok": "Target", "Deskripsi": TARGET_DESC, "Kota": "—"},
        {"Variabel": "Holiday_ID", "Kelompok": "Kalender", "Deskripsi": "ID libur nasional (kategori hari libur)", "Kota": "—"},
        {"Variabel": "holiday", "Kelompok": "Kalender", "Deskripsi": "Flag hari libur (1 = libur)", "Kota": "—"},
        {"Variabel": "school", "Kelompok": "Kalender", "Deskripsi": "Flag hari sekolah (1 = sekolah)", "Kota": "—"},
    ]
    for pref, desc in WEATHER_VARS.items():
        for suf, city in CITIES.items():
            rows.append({
                "Variabel": f"{pref}_{suf}", "Kelompok": "Cuaca",
                "Deskripsi": f"{desc} di {city}", "Kota": city,
            })
    return pd.DataFrame(rows)


def render():
    viz.set_current("data-explorer")
    hub = get_hub()
    viz.page_header(
        "Data Explorer",
        "Eksplorasi dataset beban listrik dan kovariatenya: pemilihan rentang tanggal, "
        "statistik ringkas, distribusi, korelasi antar variabel, analisis missing value, "
        "serta informasi cuaca tiga kota (Tocumen, Santiago, David) dan anomali berflag.",
        "Data Source & Exploration",
        chips=["nat_demand (MWh)", "Cuaca MERRA-2 · 3 kota", "Kalender: holiday · Holiday_ID · school"],
    )

    series, src_label, src_file = hub.actuals()
    raw = hub.get("data_with_regime")   # dataset lengkap (jika ada)

    # ------------------------------------------------- ketersediaan data ---
    if raw is not None:
        viz.note(
            f"Dataset lengkap aktif: <b>{raw.index.min():%d %b %Y} – {raw.index.max():%d %b %Y}</b> "
            f"({viz.fmt_num(len(raw))} jam, {raw.shape[1]} kolom) dari <code>results/data_with_regime.csv.gz</code>. "
            "Seluruh filter dan analisis di halaman ini berjalan penuh.", "ok", "check", "Mode lengkap")
    elif len(series) > 0:
        viz.note(
            f"Mode terbatas: dataset lengkap belum dimuat. Analisis interaktif memakai aktual periode evaluasi "
            f"(<b>{series.index.min():%d %b %Y} – {series.index.max():%d %b %Y}</b>) dari <code>{src_file}</code>. "
            "Analisis korelasi & missing value memakai gambar dokumentasi notebook.", "warn", "alert", "Mode terbatas")
    else:
        viz.note(
            "Dataset belum tersedia di folder outputs. Katalog variabel, informasi cuaca tiga kota, "
            "dan gambar EDA notebook tetap dapat dilihat di bawah.", "danger", "alert", "Data belum dimuat")

    # --------------------------------------------------- filter & control ---
    has_full = raw is not None
    st.markdown('<div class="pei-sec"><div class="bar"></div><h3>Filter Data</h3></div>', unsafe_allow_html=True)
    fcol1, fcol2, fcol3, fcol4 = st.columns([1.2, 1, 0.8, 0.8])
    with fcol1:
        st.caption("Rentang tanggal")
        if has_full:
            dmin, dmax = raw.index.min(), raw.index.max()
            drange = st.date_input("Rentang tanggal", value=(dmin.date(), dmax.date()),
                                   min_value=dmin.date(), max_value=dmax.date(),
                                   label_visibility="collapsed")
        elif len(series) > 0:
            dmin, dmax = series.index.min(), series.index.max()
            drange = st.date_input("Rentang tanggal", value=(dmin.date(), dmax.date()),
                                   min_value=dmin.date(), max_value=dmax.date(),
                                   label_visibility="collapsed")
        else:
            drange = None
            st.info("Butuh data untuk filter tanggal.")
    with fcol2:
        st.caption("Resampling")
        resample = st.selectbox("Resampling", ["Per jam", "Harian", "Mingguan"], label_visibility="collapsed")
    with fcol3:
        st.caption("Normalisasi (multi-variabel)")
        normalize = st.selectbox("Normalisasi", ["Off", "Min-Max", "Z-score"], label_visibility="collapsed")
    with fcol4:
        st.caption("Unduh")
        dl_placeholder = st.empty()  # diisi setelah dataframe terfilter tersedia

    df_view: pd.DataFrame | None = None
    if has_full:
        if isinstance(drange, tuple) and len(drange) == 2:
            d0, d1 = pd.Timestamp(drange[0]), pd.Timestamp(drange[1]) + pd.Timedelta(hours=23)
            df_view = raw.loc[(raw.index >= d0) & (raw.index <= d1)].copy()
        else:
            df_view = raw.copy()
    elif len(series) > 0 and isinstance(drange, tuple) and len(drange) == 2:
        d0, d1 = pd.Timestamp(drange[0]), pd.Timestamp(drange[1]) + pd.Timedelta(hours=23)
        df_view = series.loc[(series.index >= d0) & (series.index <= d1)].to_frame("nat_demand")

    # ------------------------------------------------------- time series ---
    viz.section("Seri Waktu Variabel", "aktif mengikuti filter rentang tanggal dan resampling")
    if df_view is not None and len(df_view) > 0:
        if has_full:
            num_cols = [c for c in ["nat_demand", "T2M_toc", "QV2M_toc", "TQL_toc", "W2M_toc",
                                    "T2M_san", "QV2M_san", "TQL_san", "W2M_san",
                                    "T2M_dav", "QV2M_dav", "TQL_dav", "W2M_dav"] if c in df_view.columns]
            defaults = [c for c in num_cols if c in ("nat_demand", "T2M_toc", "QV2M_toc")]
            sel = st.multiselect("Variabel yang ditampilkan", num_cols, default=defaults[:3])
        else:
            sel = ["nat_demand"]
        plot_df = df_view[sel] if sel else df_view[["nat_demand"]]
        if resample == "Harian":
            plot_df = plot_df.resample("D").mean()
        elif resample == "Mingguan":
            plot_df = plot_df.resample("W").mean()
        if normalize == "Min-Max" and plot_df.shape[1] > 1:
            plot_df = (plot_df - plot_df.min()) / (plot_df.max() - plot_df.min())
        elif normalize == "Z-score" and plot_df.shape[1] > 1:
            plot_df = (plot_df - plot_df.mean()) / plot_df.std()
        fig = go.Figure()
        for c in plot_df.columns:
            base_col = c.split("_")[0]
            fig.add_trace(viz.line(plot_df.index, plot_df[c], c, PALETTE_VAR.get(base_col, "#4C8DFF"), width=1.4))
        fig.update_layout(height=360, yaxis_title="Nilai" if normalize != "Off" else "Beban (MWh)")
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        st.plotly_chart(fig, use_container_width=True)
        with dl_placeholder:
            viz.download_csv(df_view.reset_index().rename(columns={"index": "datetime"}),
                             "panama_data_explorer.csv", key="dl_de")
    else:
        viz.empty_state(
            "Seri waktu belum tersedia",
            "Visualisasi interaktif membutuhkan file dataset dari notebook. Dashboard tidak menampilkan grafik kosong.",
            "outputs/results/data_with_regime.csv.gz",
            ["Jalankan notebook utama (Run All) — file diekspor otomatis ke results/",
             "Atau salin file prediksi (predictions_eval.csv.gz / predictions_chronos.csv.gz) untuk mode terbatas"],
        )

    # -------------------------------------------- statistik & distribusi ---
    viz.section("Statistik Ringkas & Distribusi", "statistik deskriptif variabel terpilih")
    if df_view is not None and len(df_view) > 0 and has_full:
        num_cols = [c for c in df_view.select_dtypes("number").columns]
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("**Statistik deskriptif**")
            desc = df_view[num_cols].describe().T
            desc["missing"] = df_view[num_cols].isna().sum()
            desc["missing_%"] = df_view[num_cols].isna().mean() * 100
            st.dataframe(desc.round(2), use_container_width=True,
                         column_config={"mean": st.column_config.NumberColumn("mean", format="%.2f"),
                                        "std": st.column_config.NumberColumn("std", format="%.2f"),
                                        "min": st.column_config.NumberColumn("min", format="%.2f"),
                                        "50%": st.column_config.NumberColumn("median", format="%.2f"),
                                        "max": st.column_config.NumberColumn("max", format="%.2f"),
                                        "missing_%": st.column_config.NumberColumn("missing %", format="%.2f")})
        with c2:
            st.markdown("**Distribusi variabel terpilih**")
            var_sel = st.selectbox("Variabel", num_cols, index=0, label_visibility="collapsed")
            vals = df_view[var_sel].dropna()
            if len(vals):
                base_col = var_sel.split("_")[0]
                fig = viz.hist_fig(vals.values, PALETTE_VAR.get(base_col, "#20D3C2"), nbins=60,
                                   mean=float(vals.mean()), median=float(vals.median()),
                                   x_title=var_sel)
                fig.update_layout(height=340)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("Tidak ada nilai untuk variabel ini pada rentang terpilih.")
        viz.download_csv(desc.reset_index().rename(columns={"index": "variabel"}),
                         "panama_statistik_ringkas.csv", label="Unduh statistik ringkas (CSV)", key="dl_stats")
    else:
        c1, c2 = st.columns(2)
        with c1:
            viz.empty_state("Statistik ringkas belum tersedia", "Butuh dataset lengkap (data_with_regime.csv.gz).", "outputs/results/data_with_regime.csv.gz", icon_name="table")
        with c2:
            viz.empty_state("Distribusi belum tersedia", "Butuh dataset lengkap (data_with_regime.csv.gz).", "outputs/results/data_with_regime.csv.gz", icon_name="chart")

    # ------------------------------------------------------ korelasi -------
    viz.section("Korelasi Antar Variabel", "beban vs cuaca tiga kota")
    if has_full:
        corr_cols = [c for c in ["nat_demand"] + [f"{w}_{s}" for s in CITIES for w in WEATHER_VARS] if c in raw.columns]
        corr = raw[corr_cols].corr()
        fig = viz.heatmap_fig(corr, height=430, colorscale="RdBu", fmt=".2f",
                              x_title="Variabel", y_title="Variabel", reversescale=True)
        fig.update_traces(zmin=-1, zmax=1)
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Tabel korelasi"):
            st.dataframe(corr.round(3), use_container_width=True)
    else:
        st.caption("Korelasi interaktif aktif otomatis ketika dataset lengkap tersedia. Di bawah ini gambar korelasi dari notebook (periode penuh).")
        f = hub.figure("eda04_korelasi")
        if f:
            viz.fig_card(f, "<b>eda04_korelasi.png</b> — Matriks korelasi beban dan 12 variabel cuaca (Tocumen, Santiago, David) dari EDA notebook.")

    # --------------------------------------------------- missing values ----
    viz.section("Analisis Missing Value", "kelengkapan data per kolom")
    if has_full:
        miss = raw.isna().mean().mul(100).sort_values(ascending=False)
        miss = miss[miss > 0] if (miss > 0).any() else miss
        if len(miss) == 0 or miss.max() == 0:
            viz.note("Tidak ditemukan missing value pada dataset — seluruh jam lengkap (validasi notebook: index unik & selisih waktu 1 jam).", "ok", "check", "Data lengkap")
        else:
            fig = viz.bar_fig(miss.index, miss.values, PALETTE["red"], height=320,
                              x_title="Kolom", y_title="% missing", hover_fmt="%{y:.2f}%")
            st.plotly_chart(fig, use_container_width=True)
    else:
        viz.empty_state("Analisis missing value belum tersedia", "Dihitung dari dataset lengkap saat file tersedia.", "outputs/results/data_with_regime.csv.gz", icon_name="search")

    # ------------------------------------------------------- anomalii ------
    viz.section("Anomali & Pemadaman Berflag", "deteksi beban rendah terhadap median jam-minggu")
    adet = hub.get("anomalies_detected")
    anom = hub.get("anomalies")
    if adet is not None:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("**Rekap deteksi per tanggal**")
            show = adet.copy()
            show["diberi_flag"] = show["diberi_flag"].map(lambda b: "Berflag (masuk fitur)" if b else "Terdeteksi (tidak berflag)")
            st.dataframe(show, hide_index=True, use_container_width=True,
                         column_config={"beban_min": st.column_config.NumberColumn("Beban minimum (MWh)", format="%.1f"),
                                        "rasio_min": st.column_config.NumberColumn("Rasio vs median", format="%.2f")})
        with c2:
            st.markdown("**Jam anomali berflag (nilai aktual)**")
            if anom is not None:
                st.dataframe(anom, hide_index=True, use_container_width=True,
                             column_config={"nat_demand": st.column_config.NumberColumn("nat_demand (MWh)", format="%.1f"),
                                            "median_referensi": st.column_config.NumberColumn("Median referensi (MWh)", format="%.1f")})
            else:
                st.caption("File detail jam anomali tidak tersedia.")
        viz.note(
            "Dua tanggal diputuskan sebagai anomali berflag (pemadaman): <b>2017-06-27</b> dan <b>2019-01-20</b> "
            "(ambang rasio 0,70–1,30 terhadap median jam-minggu di luar hari libur). Perlakuan tetap: "
            "<b>flag</b> — beban asli dipakai, flag anomali menjadi fitur pada set F1 dst.", "info", "info", "Protokol anomali")
        f = hub.figure("eda06_cek_anomali")
        if f:
            viz.fig_card(f, "<b>eda06_cek_anomali.png</b> — Pengecekan profil harian pada tanggal terdeteksi vs median hari-yang-sama (notebook).")
    else:
        viz.empty_state("Tabel anomali belum tersedia", "File hasil deteksi anomali dari notebook belum ada di outputs.", "outputs/results/anomalies_detected.csv", icon_name="alert")

    # -------------------------------------------------- PCA & katalog ------
    viz.section("Komponen PCA Cuaca", "kompresi 12 variabel cuaca menjadi 7 komponen (95% varians)")
    pca = hub.get("pca_loadings")
    if pca is not None:
        pmat = pca.set_index("variabel").astype(float)
        fig = viz.heatmap_fig(pmat, height=430, colorscale="RdBu", fmt=".2f",
                              x_title="Komponen PCA", y_title="Variabel cuaca", reversescale=True)
        fig.update_traces(zmin=-1, zmax=1)
        st.plotly_chart(fig, use_container_width=True)
        pc1 = pmat["wpc1"]
        top1 = pc1.abs().sort_values(ascending=False).head(3)
        share = (pmat.abs().sum(axis=1) / pmat.abs().values.sum() * 100)
        st.markdown(
            f'<div class="pei-note teal"><div class="n-icon">{viz.icon("spark", 16)}</div>'
            f"<div>Interpretasi otomatis: <b>wpc1</b> (komponen dengan varians terbesar) didominasi oleh "
            f"{', '.join(f'<b>{v}</b> ({pc1[v]:+.2f})' for v in top1.index)} — komponen kelembapan/suhu. "
            f"Set fitur <b>F4</b> mengganti 12 kolom cuaca dengan wpc1–wpc7 (95% varians) untuk model "
            f"ML/DL dan mengurangi multikolinearitas.</div></div>",
            unsafe_allow_html=True,
        )
    else:
        viz.empty_state("Loading PCA belum tersedia", "File loading komponen cuaca belum ada di outputs.", "outputs/results/pca_weather_loadings.csv", icon_name="layers")

    viz.section("Katalog Variabel", "target, kalender, dan cuaca tiga kota")
    st.dataframe(_var_catalog(), hide_index=True, use_container_width=True,
                 column_config={"Kota": st.column_config.Column(width="small")})
    viz.note(
        "Tiga titik pengukuran cuaca mewakili karakter geografis Panama: <b>Tocumen</b> (wilayah metro/ketengah), "
        "<b>Santiago</b> (tengah-barat) dan <b>David</b> (barat). Variabel cuaca per kota: T2M (temperatur), "
        "QV2M (kelembapan spesifik), TQL (presipitasi cair kolom), W2M (kecepatan angin 2 m).", "teal", "sun", "Wilayah cuaca")

    # ------------------------------------------- dokumentasi EDA notebook ---
    remaining = [n for n, _ in hub.figures(["eda03_suhu_beban", "eda05_acf"])]
    if remaining:
        with st.expander("Gambar dokumentasi dari notebook (statis)"):
            if "eda03_suhu_beban" in remaining:
                f = hub.figure("eda03_suhu_beban")
                viz.fig_card(f, "<b>eda03_suhu_beban.png</b> — Hexbin suhu (T2M) tiap kota vs beban; korelasi positif terlihat pada rentang suhu tinggi.")
            if "eda05_acf" in remaining:
                f = hub.figure("eda05_acf")
                viz.fig_card(f, "<b>eda05_acf.png</b> — Autokorelasi beban hingga lag 336 jam; puncak periodik harian (24) dan mingguan (168).")
