"""PANAMA ENERGY INTELLIGENCE — konfigurasi terpusat.

Semua konstanta aplikasi (palet, roster model, definisi fitur, status Chronos,
path data) dikelola di sini agar perubahan struktur file / kebijakan tampilan
cukup dilakukan di satu tempat.
"""
from __future__ import annotations

import os
from pathlib import Path

# ------------------------------------------------------------------ paths ---
BASE_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = BASE_DIR / "assets"
# Folder data dapat dioverride lewat env var (mis. saat deploy dengan folder lain)
DATA_DIR = Path(os.environ.get("PANAMA_OUTPUTS_DIR", str(BASE_DIR / "outputs")))
RESULTS_DIR = DATA_DIR / "results"
FIGURES_DIR = DATA_DIR / "figures"

APP_NAME = "Panama Energy Intelligence"
APP_VERSION = "1.0.0"
APP_TAGLINE = "Load forecasting analytics · Panama 2015–2020"

# ------------------------------------------------------------------ palet ---
PALETTE = {
    "bg": "#0B1220",          # background utama
    "surface": "#111C2E",     # kartu / panel
    "surface_hi": "#16233A",  # kartu hover / input
    "accent": "#20D3C2",      # teal/cyan
    "text": "#F2F5FA",
    "muted": "#8CA0B8",
    "border": "rgba(148,163,184,0.14)",
    "blue": "#4C8DFF",
    "amber": "#FFB65C",
    "red": "#FF6B6B",
    "green": "#3DDC97",
    "purple": "#A78BFA",
    "pink": "#F472B6",
    "gray": "#8CA0B8",
}

# Urutan warna seri untuk chart (konsisten antar halaman)
SERIES_COLORS = [
    "#20D3C2", "#4C8DFF", "#FFB65C", "#A78BFA",
    "#FF6B6B", "#3DDC97", "#F472B6", "#8CA0B8",
]

# Warna kelompok model (dipakai di semua chart perbandingan)
FAMILY_COLORS = {
    "Konvensional": "#FFB65C",
    "ML": "#4C8DFF",
    "DL": "#A78BFA",
    "Foundation": "#20D3C2",
    "Baseline": "#8CA0B8",
    "Cold-ML": "#3DDC97",
    "Cold-DL": "#F472B6",
    "Chronos-ctx": "#20D3C2",
}

# ------------------------------------------------------- parameter studi ---
HORIZONS = [1, 6, 24, 72]
H_REF = 24
SPLIT_FRAC = (0.70, 0.10, 0.20)
EVAL_STRIDE = 3                 # evaluasi pada origin tiap 3 jam
ALPHA_DM = 0.05                 # taraf signifikansi Diebold-Mariano
ANOM_DATES = ["2017-06-27", "2019-01-20"]
TOTAL_OBS = 48_048              # 33.623 train + 4.800 val + 9.625 test (jam)
TARGET_LABEL = "nat_demand"
TARGET_DESC = "Beban listrik nasional Panama (MWh)"
DATA_PERIOD = "January 2015 – June 2020"

FEATURE_SETS = {
    "F0": "Riwayat beban — lag 0/1/24/168 jam, rolling mean & std (24, 168), beban jam yang sama hari/minggu sebelumnya",
    "F1": "F0 + kalender — jam, hari, bulan, akhir pekan, holiday, Holiday_ID, school, flag anomali",
    "F2": "F1 + cuaca Tocumen (T2M, QV2M, TQL, W2M)",
    "F3": "F1 + cuaca tiga kota (Tocumen, Santiago, David — 12 variabel)",
    "F4": "F1 + komponen PCA cuaca (95% varians, 7 komponen wpc1–wpc7)",
}

WEATHER_VARS = {
    "T2M": "Temperatur udara 2 m (°C)",
    "QV2M": "Kelembapan spesifik 2 m (kg/kg)",
    "TQL": "Presipitasi cair total kolom (kg/m²)",
    "W2M": "Kecepatan angin 2 m (m/s)",
}
CITIES = {"toc": "Tocumen", "san": "Santiago", "dav": "David"}

# Roster model (sesuai notebook; model_id di metrics_all = "<nama>|<fset>")
MODEL_ROSTER = [
    {"family": "Konvensional", "label": "SARIMA", "match": "SARIMA",
     "desc": "Seasonal ARIMA (1,0,1)(0,1,1)[24], fit 120 hari, window 336 jam"},
    {"family": "Konvensional", "label": "ETS", "match": "ETS",
     "desc": "Exponential smoothing periode mingguan (168 jam)"},
    {"family": "Konvensional", "label": "ARIMAX", "match": "ARIMAX",
     "desc": "ARIMAX dengan kovariat kalender/cuaca (set fitur F1, F3)"},
    {"family": "ML", "label": "Random Forest", "match": "RandomForest",
     "desc": "150 pohon, hyperparameter ringan via validasi"},
    {"family": "ML", "label": "XGBoost", "match": "XGBoost",
     "desc": "2.000 boosting rounds + early stopping"},
    {"family": "ML", "label": "LightGBM", "match": "LightGBM",
     "desc": "2.000 boosting rounds + early stopping"},
    {"family": "DL", "label": "LSTM", "match": "DL-LSTM",
     "desc": "2 lapis × 64 unit, sekuens 96 jam"},
    {"family": "DL", "label": "GRU", "match": "DL-GRU",
     "desc": "2 lapis × 64 unit, sekuens 96 jam"},
    {"family": "DL", "label": "CNN-1D Dilated", "match": "DL-CNN",
     "desc": "CNN 1D dilated, sekuens 96 jam"},
    {"family": "Foundation", "label": "Chronos-2 Small", "match": "Chronos2-small",
     "desc": "Zero-shot + fine-tuning LoRA (±28M parameter)"},
    {"family": "Foundation", "label": "Chronos-2 Base", "match": "Chronos2-base",
     "desc": "Zero-shot + fine-tuning LoRA (±120M parameter)"},
]

MODEL_FAMILIES = ["Konvensional", "ML", "DL", "Foundation"]

# ------------------------------------------------------- kebijakan Chronos ---
# PENTING: sesuai hasil eksperimen saat ini, Chronos-2 belum menghasilkan
# prediksi yang valid (kendala CUDA + dependency torchao). Nilai prediksi /
# metrik Chronos TIDAK boleh ditampilkan dan TIDAK boleh diimprovisasi.
# Set ke "available" hanya setelah run berhasil dan file hasil tersedia.
CHRONOS_STATUS = "not_available"   # "not_available" | "available"
CHRONOS_NOTE = (
    "Eksekusi Chronos-2 gagal karena kendala CUDA dan dependency torchao, "
    "sehingga prediksi dan metrik Chronos-2 Small / Base belum tersedia dan "
    "belum dapat dibandingkan dengan model lain."
)
CHRONOS_PROTOCOL = {
    "Model": "Chronos-2 Small (autogluon/chronos-2-small, ±28M params) · Chronos-2 Base (amazon/chronos-2, ±120M params)",
    "Zero-shot": "Konteks utama 512 jam; studi panjang konteks 168 / 720 / 4.320 jam",
    "Kovariat": "Eksperimen covariate-support pada set fitur F1 dan F3 (ukuran base)",
    "Fine-tuning": "LoRA — grid (num_steps × lr): 250×1e-5 dan 1.000×1e-5, dipilih berdasar RMSE validasi",
    "Evaluasi": "Origin tiap 3 jam pada regime validasi & test, horizon 1/6/24/72 jam",
}

# --------------------------------------------------------- definisi metrik ---
METRIC_DEFS = {
    "RMSE": "Root Mean Squared Error = √(mean (y − ŷ)²). Satuan MWh; sensitif terhadap kesalahan besar.",
    "MAE": "Mean Absolute Error = mean |y − ŷ| (MWh). Lebih tahan terhadap outlier dibanding RMSE.",
    "MAPE": "Mean Absolute Percentage Error = mean (|y − ŷ| / |y| × 100)%. Tidak simetris dan sensitif saat beban kecil.",
    "sMAPE": "Symmetric MAPE = mean (2|y − ŷ| / (|y| + |ŷ|) × 100)%. Versi simetris dari MAPE.",
    "DM": "Uji Diebold-Mariano membandingkan akurasi dua model forecast menggunakan loss kuadrat dengan varians HAC (lag truncation K = max(⌈(h−1)/3⌉, ⌈24/3⌉)).",
    "eval_note": "Semua metrik dihitung pada origin evaluasi tiap 3 jam (EVAL_STRIDE = 3) untuk regime validasi dan test.",
}
