"""Pusat pemuatan data — semua file hasil notebook dibaca lewat modul ini.

Prinsip:
1. Satu registry untuk semua file yang diharapkan (nama, deskripsi, halaman).
2. Scan otomatis di dalam folder outputs (mendukung struktur nested).
3. Jika file ada -> dimuat dan dinormalisasi; jika tidak -> None (halaman
   menampilkan empty state informatif, TIDAK ada angka/improvisasi).
4. Cache berbasis path + mtime, sehingga file baru otomatis terbaca.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from . import settings

# ------------------------------------------------------------ registry ----
@dataclass
class FileSpec:
    key: str
    filename: str
    desc: str
    pages: str
    essential: bool = False   # essential = ada di daftar 9 file utama spesifikasi


REGISTRY: list[FileSpec] = [
    FileSpec("metrics_all", "metrics_all.csv", "Metrik MAE/RMSE/MAPE/sMAPE per model × horizon (validasi & test)", "Benchmark · Overview · Forecasting", True),
    FileSpec("predictions_eval", "predictions_eval.csv.gz", "Prediksi semua model pada origin evaluasi (y_true, y_pred)", "Forecasting", True),
    FileSpec("dm_tests", "dm_tests.csv", "Uji Diebold-Mariano model terbaik vs pembanding", "Statistical Testing", True),
    FileSpec("chronos_vs_trained", "chronos_vs_trained.csv", "Perbandingan Chronos vs model dilatih sendiri (RMSE test)", "Benchmark", True),
    FileSpec("cold_start_rmse", "cold_start_rmse.csv", "RMSE cold start per jumlah data training", "Data Efficiency", True),
    FileSpec("prob_metrics", "prob_metrics.csv", "Metrik probabilistik (pinball, CRPS, cakupan 80%)", "Benchmark", True),
    FileSpec("importance_by_group", "importance_by_group_pct.csv", "Kontribusi SHAP per kelompok fitur (%)", "Driver Analysis", True),
    FileSpec("cost_table", "cost_table.csv", "Biaya komputasi (waktu latih, inferensi, GPU)", "Computational Cost", True),
    FileSpec("ringkasan_txt", "ringkasan.txt", "Ringkasan hasil eksekusi notebook (teks)", "Overview", True),
    # file pelengkap yang juga diekspor notebook utama
    FileSpec("ablation_fset", "ablation_fset_rmse.csv", "Ablasi RMSE per set fitur (model ML)", "Benchmark"),
    FileSpec("resid_hour", "resid_by_hour.csv", "Residual per jam (model terbaik)", "Forecasting"),
    FileSpec("resid_daytype", "resid_by_tipe_hari.csv", "Residual per tipe hari", "Forecasting"),
    FileSpec("resid_season", "resid_by_musim.csv", "Residual per musim", "Forecasting"),
    FileSpec("resid_month", "resid_by_bulan.csv", "Residual per bulan", "Forecasting"),
    FileSpec("data_with_regime", "data_with_regime.csv.gz", "Dataset mentah lengkap + label regime (train/val/test)", "Data Explorer · Overview"),
    FileSpec("eda_profile", "eda_profile_dow_hour.csv", "Pola rata-rata beban hari × jam (EDA)", "Overview"),
    FileSpec("eda_monthly", "eda_monthly_mean.csv", "Rata-rata beban bulanan (EDA)", "Overview"),
    FileSpec("eda_daily", "eda_daily.csv", "Agregat harian beban (EDA)", "Data Explorer"),
    FileSpec("dl_perm_importance", "dl_permutation_importance.csv", "Permutation importance model Deep Learning", "Driver Analysis"),
    FileSpec("lgbm_importance", "lgbm_feature_importance.csv", "Feature importance LightGBM", "Driver Analysis"),
    FileSpec("dl_history", "dl_training_history.csv", "Riwayat training model DL", "Driver Analysis"),
    FileSpec("pca_loadings", "pca_weather_loadings.csv", "Loading PCA variabel cuaca (komponen wpc1–wpc7)", "Driver Analysis · Data Explorer"),
    FileSpec("anomalies", "anomalies.csv", "Jam anomali (pemadaman) dengan nilai aktual", "Data Explorer"),
    FileSpec("anomalies_detected", "anomalies_detected.csv", "Rekap deteksi anomali per tanggal", "Data Explorer"),
    FileSpec("regime_summary", "regime_summary.csv", "Rekap split kronologis train/val/test", "Overview"),
    FileSpec("predictions_quantiles", "predictions_quantiles.csv.gz", "Prediksi kuantil (pita interval 80%)", "Forecasting"),
    FileSpec("config_json", "config.json", "Konfigurasi eksekusi notebook", "Overview"),
]

ESSENTIAL_KEYS = [s.key for s in REGISTRY if s.essential]


# --------------------------------------------------------------- scan -----
@st.cache_resource(ttl=30)
def scan_files() -> dict[str, Path]:
    """Cari setiap file registry di dalam folder outputs (pencarian nested)."""
    found: dict[str, Path] = {}
    if not settings.DATA_DIR.exists():
        return found
    all_files = [p for p in settings.DATA_DIR.rglob("*") if p.is_file()]
    for spec in REGISTRY:
        candidates = [p for p in all_files if p.name == spec.filename]
        if candidates:
            found[spec.key] = sorted(candidates, key=lambda p: (len(p.parts), str(p)))[0]
    return found


@st.cache_data(ttl=30)
def _mtime(path_str: str) -> float:
    try:
        return Path(path_str).stat().st_mtime
    except OSError:
        return 0.0


# -------------------------------------------------------------- loaders ---
def _read_csv(path: Path, **kw) -> pd.DataFrame:
    if path.name.endswith(".csv.gz"):
        kw.setdefault("compression", "gzip")
    return pd.read_csv(path, **kw)


def _parse_dt(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    return df


@st.cache_data(ttl=30)
def load_file(key: str, path_str: str, mtime: float) -> pd.DataFrame | None:
    """Muat + normalisasi satu file hasil. Aman terhadap skema drift ringan."""
    path = Path(path_str)
    try:
        if key == "metrics_all":
            df = _read_csv(path)
            df["horizon"] = df.get("horizon", pd.Series(dtype=float)).astype(int)
            df["model_id"] = df["model_id"].astype(str)
            return df
        if key == "predictions_eval":
            df = _read_csv(path)
            df = _parse_dt(df, ["origin"])
            if "target_time" not in df.columns:
                df["target_time"] = df["origin"] + pd.to_timedelta(df["horizon"].astype(int), unit="h")
            else:
                df = _parse_dt(df, ["target_time"])
            df["model_id"] = df["model_id"].astype(str)
            return df
        if key == "dm_tests":
            df = _read_csv(path)
            df["horizon"] = df["horizon"].astype(int)
            for c in ("DM_stat", "p_value", "RMSE_terbaik", "RMSE_pembanding"):
                if c in df:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            df["signifikan"] = df["p_value"] < settings.ALPHA_DM
            return df
        if key == "cold_start_rmse":
            df = _read_csv(path)
            df["horizon"] = df["horizon"].astype(int)
            df["hari"] = pd.to_numeric(df["hari"], errors="coerce")
            return df
        if key in ("prob_metrics", "resid_hour", "resid_daytype", "resid_season", "resid_month"):
            df = _read_csv(path)
            if "horizon" in df:
                df["horizon"] = df["horizon"].astype(int)
            return df
        if key == "chronos_vs_trained":
            return _read_csv(path)
        if key == "ablation_fset":
            df = _read_csv(path)
            return df
        if key == "importance_by_group":
            df = _read_csv(path)
            df = df.rename(columns={df.columns[0]: "kelompok"})
            return df
        if key == "cost_table":
            df = _read_csv(path)
            for c in ("train_sec", "infer_sec", "n_fc", "ms_per_prakiraan", "gpu_mb_infer", "gpu_mb_train"):
                if c in df:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            df["model_id"] = df["model_id"].astype(str)
            return df
        if key == "data_with_regime":
            df = _read_csv(path)
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
                df = df.sort_values("datetime").set_index("datetime")
            return df
        if key in ("eda_profile", "eda_monthly", "eda_daily"):
            return _read_csv(path)
        if key in ("dl_perm_importance", "lgbm_importance"):
            df = _read_csv(path)
            return df.rename(columns={df.columns[0]: "fitur"})
        if key in ("pca_loadings",):
            df = _read_csv(path)
            return df.rename(columns={df.columns[0]: "variabel"})
        if key in ("anomalies",):
            return _parse_dt(_read_csv(path), ["datetime"])
        if key in ("anomalies_detected",):
            return _read_csv(path)
        if key == "regime_summary":
            df = _read_csv(path)
            df = df.rename(columns={df.columns[0]: "regime"})
            df = _parse_dt(df, ["mulai", "akhir"])
            return df
        if key == "predictions_quantiles":
            df = _read_csv(path)
            df = _parse_dt(df, ["origin"])
            df["model_id"] = df["model_id"].astype(str)
            return df
        if key == "ringkasan_txt":
            return path.read_text(encoding="utf-8", errors="replace")  # type: ignore[return-value]
        if key == "config_json":
            return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[return-value]
        return _read_csv(path)
    except Exception:
        # file korup / skema tak terduga -> anggap tidak tersedia (tanpa crash)
        return None


@st.cache_data(ttl=30)
def load_actuals(path_str: str, mtime: float, source: str) -> tuple[pd.Series, str, str]:
    """Seri aktual beban (y_true asli dari notebook, BUKAN hasil model).

    Prioritas sumber:
    1. data_with_regime.csv.gz  -> dataset lengkap 2015–2020.
    2. predictions_eval.csv.gz  -> aktual pada origin evaluasi (val+test).
    3. predictions_chronos.csv.gz -> aktual pada periode test (kolom y_true).
    """
    path = Path(path_str)
    try:
        if source == "data_with_regime":
            df = _read_csv(path)
            df["datetime"] = pd.to_datetime(df["datetime"])
            s = df.set_index("datetime")["nat_demand"].astype(float).sort_index()
            return s, "Dataset lengkap", "data_with_regime.csv.gz"
        if source == "predictions_eval":
            df = _read_csv(path)
            df["target_time"] = pd.to_datetime(df["target_time"]) if "target_time" in df.columns else pd.to_datetime(df["origin"]) + pd.to_timedelta(df["horizon"], unit="h")
            s = df.groupby("target_time")["y_true"].first().astype(float).sort_index()
            return s, "Aktual pada origin evaluasi (validasi + test)", "predictions_eval.csv.gz"
        # fallback: predictions_chronos (hanya periode test)
        df = _read_csv(path, compression="gzip")
        df["target_time"] = pd.to_datetime(df["target_time"])
        s = df.groupby("target_time")["y_true"].first().astype(float).sort_index()
        return s, "Aktual periode test (diekstrak dari run notebook)", "predictions_chronos.csv.gz"
    except Exception:
        return pd.Series(dtype=float), "", ""


# ------------------------------------------------------------- facade -----
class DataHub:
    """Facade ringan untuk halaman-halaman Streamlit."""

    def __init__(self):
        self.paths = scan_files()

    def has(self, key: str) -> bool:
        return key in self.paths

    def get(self, key: str):
        """DataFrame / teks / dict hasil, atau None bila file tidak ada."""
        if key not in self.paths:
            return None
        return load_file(key, str(self.paths[key]), _mtime(str(self.paths[key])))

    def figure(self, name: str) -> Path | None:
        """Path gambar statis hasil notebook (mis. 'eda01_series')."""
        for ext in (".png", ".jpg"):
            p = settings.FIGURES_DIR / f"{name}{ext}"
            if p.exists():
                return p
        return None

    def figures(self, names: list[str]) -> list[tuple[str, Path]]:
        out = []
        for n in names:
            p = self.figure(n)
            if p:
                out.append((n, p))
        return out

    def status_table(self) -> pd.DataFrame:
        rows = []
        for spec in REGISTRY:
            path = self.paths.get(spec.key)
            rows.append({
                "File": spec.filename,
                "Deskripsi": spec.desc,
                "Dipakai di": spec.pages,
                "Utama": "Ya" if spec.essential else "Pelengkap",
                "Status": "Tersedia" if path else "Belum tersedia",
                "Lokasi": str(path.relative_to(settings.DATA_DIR)) if path else "—",
            })
        return pd.DataFrame(rows)

    def counts(self) -> tuple[int, int, int]:
        """(jumlah tersedia total, jumlah file utama tersedia, 9)."""
        avail = len(self.paths)
        ess = sum(1 for k in ESSENTIAL_KEYS if k in self.paths)
        return avail, ess, len(ESSENTIAL_KEYS)

    # ------------------------------------------------ sering dipakai -----
    def actuals(self) -> tuple[pd.Series, str, str]:
        """Seri aktual beban + label sumber + nama file sumber."""
        if self.has("data_with_regime"):
            return load_actuals(str(self.paths["data_with_regime"]), _mtime(str(self.paths["data_with_regime"])), "data_with_regime")
        if self.has("predictions_eval"):
            return load_actuals(str(self.paths["predictions_eval"]), _mtime(str(self.paths["predictions_eval"])), "predictions_eval")
        if self.has("_fallback_actuals"):
            pass
        # fallback terakhir: aktual periode test dari file prediksi run notebook
        fb = settings.DATA_DIR / "results" / "predictions_chronos.csv.gz"
        if fb.exists():
            return load_actuals(str(fb), _mtime(str(fb)), "predictions_chronos")
        return pd.Series(dtype=float), "", ""

    def best_per_horizon(self) -> pd.DataFrame | None:
        """Model terbaik per horizon (test) — Chronos dikecualikan sesuai status."""
        m = self.get("metrics_all")
        if m is None:
            return None
        lb = m[(m.regime_group == "test") & m.model_id.astype(str).str.contains("|", regex=False)]
        lb = lb[~lb.model_id.astype(str).str.startswith("Chronos")]
        if lb.empty:
            return None
        idx = lb.groupby("horizon")["RMSE"].idxmin()
        cols = ["horizon", "model_id", "family", "model", "fset", "MAE", "RMSE", "MAPE", "sMAPE"]
        return lb.loc[idx, [c for c in cols if c in lb.columns]].sort_values("horizon").reset_index(drop=True)

    def chronos_on(self) -> bool:
        return settings.CHRONOS_STATUS == "available"


@st.cache_resource(ttl=60)
def get_hub() -> DataHub:
    return DataHub()


def chronos_family_mask(df: pd.DataFrame, col: str = "family") -> pd.Series:
    """True untuk baris keluarga model Foundation/Chronos (dikecualikan bila off)."""
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    fam = df[col].astype(str)
    return fam.str.contains("Chronos|Foundation", case=False, regex=True)
