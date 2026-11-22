"""Halaman Computational Cost — waktu latih/inferensi dan pemakaian GPU."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import viz
from core.data_loader import get_hub
from core.settings import PALETTE

METRICS_INFO = [
    ("train_sec", "Training Time", "Total waktu fitting model (detik) — sekali per kombinasi (set fitur × horizon).", "clock", "blue"),
    ("ft_sec", "Fine-tuning Time", "Waktu fine-tuning (mis. LoRA Chronos) — hanya kelompok yang melakukan fine-tuning.", "clock", "amber"),
    ("infer_sec", "Inference Time", "Total waktu inferensi seluruh origin evaluasi (detik).", "cpu", "purple"),
    ("ms_per_prakiraan", "Inference per Forecast", "ms per prakiraan (satu origin × satu horizon) — metrik latensi deploy.", "activity", "green"),
    ("gpu_mb", "GPU Memory Usage", "Memori GPU maksimum saat training/inferensi (MB).", "database", "red"),
]


def render():
    viz.set_current("computational-cost")
    hub = get_hub()
    viz.page_header(
        "Computational Cost",
        "Biaya komputasi setiap model: waktu training, fine-tuning, inferensi total, latensi per prakiraan, "
        "dan pemakaian memori GPU. Data diukur langsung oleh notebook saat eksperimen berjalan.",
        "Compute & Efficiency",
        chips=["Training time", "Fine-tuning", "Inference latency", "GPU memory"],
    )

    cost = hub.get("cost_table")

    # definisi metrik selalu tampil
    viz.section("Definisi Metrik", "arahkan kursor ke ikon untuk penjelasan")
    cols = st.columns(len(METRICS_INFO))
    for i, (k, lab, desc, ic, tone) in enumerate(METRICS_INFO):
        with cols[i]:
            st.markdown(
                f'<div class="kpi-card" style="min-height:118px"><div class="kpi-top">'
                f'<div class="kpi-label">{viz.esc(lab)} <span class="tip">{viz.icon("info", 12)}'
                f'<span class="tip-body">{desc}</span></span></div>'
                f'<div class="kpi-icon {tone}">{viz.icon(ic, 17)}</div></div>'
                f'<div class="kpi-sub" style="font-size:.78rem">{viz.esc(desc.split(" — ")[0])}.</div></div>',
                unsafe_allow_html=True,
            )

    if cost is None:
        viz.section("Tabel & Grafik Komparasi", "aktif setelah file biaya komputasi tersedia")
        viz.empty_state(
            "Tabel biaya komputasi belum tersedia",
            "Nilai biaya diukur notebook selama eksekusi (sel 15: biaya komputasi dan ekspor). "
            "Dashboard menampilkan definisi metrik di atas, namun tidak mengarang angka biaya. "
            "Baris Chronos juga hanya dimuat setelah eksperimen Chronos berhasil (status saat ini: gagal dieksekusi).",
            "outputs/results/cost_table.csv",
            ["Jalankan notebook utama (Run All) — waktu latih/inferensi dicatat otomatis",
             "Salin results/cost_table.csv ke outputs/results/",
             "Skema: model_id · family · train_sec · infer_sec · n_fc · ms_per_prakiraan · (gpu_mb_*)"],
            icon_name="cpu",
        )
        viz.note(
            "Chronos-2 Small & Base: status <b>Not Available / Failed Execution</b> — kendala CUDA dan dependency "
            "torchao sehingga biaya komputasi Chronos juga belum tersedia untuk dibandingkan.", "warn", "alert", "Status Chronos-2",
        )
        return

    # ------------------------------------------------------------- aktif ---
    if not hub.chronos_on() and "model_id" in cost.columns:
        cost = cost[~cost["model_id"].astype(str).str.startswith("Chronos")]
    cost = cost.sort_values(["family", "ms_per_prakiraan"], ascending=[True, False])

    viz.kpi_row([
        viz.kpi_card("Model Terukur", str(cost["model_id"].nunique()), "kombinasi model × set fitur", "cpu", "blue"),
        viz.kpi_card("Training Terlama", f"{viz.fmt_num(pd.to_numeric(cost.get('train_sec'), errors='coerce').max(), 0)} dtk",
                     "satu kombinasi terlama", "clock", "amber"),
        viz.kpi_card("Latensi Terbaik", f"{pd.to_numeric(cost['ms_per_prakiraan'], errors='coerce').min():.1f} ms",
                     "per prakiraan (satu origin × horizon)", "activity", "green"),
        viz.kpi_card("GPU Terberat", f"{viz.fmt_num(pd.to_numeric(cost[[c for c in cost.columns if c.startswith('gpu_mb')]].max(axis=1), errors='coerce').max(), 0)} MB" if any(c.startswith('gpu_mb') for c in cost.columns) else "—",
                     "memori GPU maksimum tercatat", "database", "red"),
    ])

    viz.section("Latensi per Prakiraan", "horizontal bar — semakin pendek semakin cepat")
    d1 = cost.dropna(subset=["ms_per_prakiraan"]).sort_values("ms_per_prakiraan")
    fig = go.Figure(go.Bar(
        y=d1["model_id"], x=d1["ms_per_prakiraan"], orientation="h",
        marker=dict(color=[PALETTE["accent"] if v == d1["ms_per_prakiraan"].min() else "#3D5273" for v in d1["ms_per_prakiraan"]], opacity=0.9),
        text=[f"{v:.1f} ms" for v in d1["ms_per_prakiraan"]], textposition="outside",
        textfont=dict(size=10.5, color="#B9C6D8"),
        hovertemplate="%{y}<br>%{x:.2f} ms/prakiraan<extra></extra>",
    ))
    fig.update_layout(height=max(300, 26 * len(d1)), xaxis_title="ms per prakiraan (log scale)",
                      xaxis=dict(type="log"))
    st.plotly_chart(fig, use_container_width=True)

    if "train_sec" in cost.columns and cost["train_sec"].notna().any():
        viz.section("Waktu Training", "detik — sekali fit per kombinasi")
        d2 = cost.dropna(subset=["train_sec"]).sort_values("train_sec")
        fig2 = go.Figure(go.Bar(
            y=d2["model_id"], x=d2["train_sec"], orientation="h",
            marker=dict(color=PALETTE["blue"], opacity=0.85),
            text=[f"{v:,.0f} dtk" for v in d2["train_sec"]], textposition="outside",
            textfont=dict(size=10.5, color="#B9C6D8"),
            hovertemplate="%{y}<br>%{x:,.0f} detik<extra></extra>",
        ))
        fig2.update_layout(height=max(300, 26 * len(d2)), xaxis_title="detik (log scale)", xaxis=dict(type="log"))
        st.plotly_chart(fig2, use_container_width=True)

    viz.section("Tabel Komparasi Lengkap", "dapat diurutkan per kolom")
    disp = cost.copy()
    st.dataframe(disp, hide_index=True, use_container_width=True, height=380,
                 column_config={
                     "train_sec": st.column_config.NumberColumn("train (dtk)", format="%.1f"),
                     "infer_sec": st.column_config.NumberColumn("infer (dtk)", format="%.1f"),
                     "n_fc": st.column_config.NumberColumn("n prakiraan", format="%d"),
                     "ms_per_prakiraan": st.column_config.NumberColumn("ms/prakiraan", format="%.2f"),
                     "gpu_mb_infer": st.column_config.NumberColumn("GPU infer (MB)", format="%.0f"),
                     "gpu_mb_train": st.column_config.NumberColumn("GPU train (MB)", format="%.0f"),
                 })
    viz.download_csv(cost, "cost_table.csv", key="dl_cost")
