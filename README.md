# PANAMA ENERGY INTELLIGENCE

Dashboard web interaktif untuk hasil penelitian **forecasting beban listrik nasional Panama 2015–2020** — benchmark konvensional (SARIMA, ETS, ARIMAX), machine learning (Random Forest, XGBoost, LightGBM), deep learning (LSTM, GRU, CNN-1D dilated), dan model fondasi Chronos-2, lengkap dengan uji Diebold-Mariano, analisis faktor penggerak (SHAP/PDP), efisiensi data (cold start), dan biaya komputasi.

Dibangun dengan **Python + Streamlit + Plotly + Pandas/NumPy** dan CSS kustom bertema *dark navy* (#0B1220) dengan aksen *teal* (#20D3C2).

---

## Cara Menjalankan (lokal)

```bash
cd panama-energy-intelligence
pip install -r requirements.txt
streamlit run app.py
```

Dashboard terbuka di `http://localhost:8501`.

## Deploy ke Streamlit Cloud

1. Push folder ini ke repository GitHub (termasuk folder `outputs/`).
2. Di [share.streamlit.io](https://share.streamlit.io) → *New app* → pilih repo/branch.
3. Main file path: `app.py`. Selesai — tidak ada variabel lingkungan khusus.

## Struktur Kode (modular)

```
panama-energy-intelligence/
├── app.py                  # entry point: navigasi, sidebar, error handling global
├── core/
│   ├── settings.py         # konfigurasi terpusat: palet, roster model, definisi fitur,
│   │                       #   kebijakan Chronos (CHRONOS_STATUS), path data
│   ├── theme.py            # design system: CSS kustom + template Plotly "pei"
│   ├── data_loader.py      # SATU sumber baca file hasil notebook (registry + cache
│   │                       #   + pengecekan ketersediaan + normalisasi skema)
│   └── viz.py              # komponen UI: ikon SVG, kartu KPI, badge, empty state,
│                           #   tooltip definisi metrik, helper chart
├── views/                  # 8 halaman (satu file per halaman)
│   ├── overview.py         # KPI, seri waktu, pola beban, insight otomatis, status data
│   ├── data_explorer.py    # filter, statistik, distribusi, korelasi, missing value,
│   │                       #   anomali, PCA cuaca, katalog variabel 3 kota
│   ├── forecasting.py      # aktual vs prediksi, residual, scatter, distribusi error,
│   │                       #   perbandingan model, KPI RMSE/MAE/MAPE/sMAPE, unduh CSV
│   ├── benchmark.py        # ranking, grouped bar MAE/RMSE, heatmap, ablasi F0–F4,
│   │                       #   banner status Chronos-2
│   ├── statistical.py      # uji Diebold-Mariano (H0/H1, alpha 0.05, warna signifikan)
│   ├── drivers.py          # SHAP per kelompok, PDP T2M, permutation importance DL,
│   │                       #   loading PCA cuaca
│   ├── efficiency.py       # cold start 1 bln / 3 bln / 1 thn / 3 thn / full
│   └── cost.py             # training time, fine-tuning, inference, GPU memory
├── outputs/
│   ├── figures/            # gambar hasil notebook (dipakai sebagai dokumentasi statis)
│   └── results/            # file hasil eksperimen (CSV/CSV.GZ) — sumber data dashboard
├── assets/logo.png
├── .streamlit/config.toml  # tema dasar Streamlit
├── requirements.txt
└── README.md
```

## Integrasi Data — sistem pengecekan file

Semua data dibaca dari folder `outputs/results/` (bisa dioverride dengan env var `PANAMA_OUTPUTS_DIR`). Loader terpusat (`core/data_loader.py`) memindai file berikut:

**File utama (9):** `metrics_all.csv` · `predictions_eval.csv.gz` · `dm_tests.csv` · `chronos_vs_trained.csv` · `cold_start_rmse.csv` · `prob_metrics.csv` · `importance_by_group_pct.csv` · `cost_table.csv` · `ringkasan.txt`

**File pelengkap:** `ablation_fset_rmse.csv`, `resid_by_*.csv`, `data_with_regime.csv.gz`, `eda_profile_dow_hour.csv`, `eda_monthly_mean.csv`, `eda_daily.csv`, `dl_permutation_importance.csv`, `lgbm_feature_importance.csv`, `pca_weather_loadings.csv`, `anomalies*.csv`, `regime_summary.csv`, `predictions_quantiles.csv.gz`, `config.json`.

Aturan penting:
- **File ada** → halaman menampilkan data (grafik interaktif Plotly + tabel + tombol unduh CSV).
- **File tidak ada** → halaman menampilkan *empty state* informatif (nama file yang ditunggu, skema, langkah menghasilkannya) dan gambar dokumentasi statis dari notebook bila ada. **Dashboard tidak pernah mengarang angka.**
- Setelah Anda menjalankan notebook (`Run All`, `FAST_DEBUG=False`) dan menyalin folder `results/` ke `outputs/`, seluruh halaman **aktif otomatis** — tanpa perubahan kode (cache disegarkan berdasarkan waktu modifikasi file).

## Status Chronos-2 (PENTING)

Sesuai hasil eksperimen saat ini, **Chronos-2 Small & Base belum menghasilkan prediksi** (kendala CUDA dan dependency `torchao`). Dashboard menampilkan statusnya apa adanya: **"Not Available / Failed Execution"** pada halaman Model Benchmark, Data Efficiency, dan Computational Cost — tanpa nilai prediksi/metrik fiktif, dan Chronos dikecualikan dari seluruh ranking.

Bila nanti eksekusi Chronos berhasil, cukup ubah `CHRONOS_STATUS = "available"` di `core/settings.py` — baris Chronos pada `metrics_all.csv` / `chronos_vs_trained.csv` / cold start akan terbaca otomatis ke semua perbandingan.

## Catatan sumber data saat ini

- Folder `outputs/results/` di repo ini berisi artefak dataset dari run notebook yang tersedia (rekap split, anomali berflag, loading PCA cuaca) + aktual periode test diekstrak dari file prediksi run notebook untuk menghidupkan grafik Overview/Data Explorer.
- Grafik interaktif menampilkan **cakupan aktual yang benar-benar tersedia** (Mei 2019 – Jun 2020) dan selalu diberi catatan sumber. Muat `data_with_regime.csv.gz` untuk periode penuh 2015–2020.
- Gambar statis (eda01–eda06, eval01–eval08, drv01–drv05) adalah output asli notebook dari run sebelumnya — tanpa Chronos.
