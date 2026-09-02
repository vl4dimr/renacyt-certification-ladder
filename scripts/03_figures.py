# -*- coding: utf-8 -*-
"""Figuras del manuscrito (Research Evaluation, 600 dpi, minimalistas).
Población principal: Reglamento 2021 (activos). También escribe
outputs/tabla_nivel_2021.csv para la Tabla 1 del docx."""
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

AZUL = "#1c5cab"
AZUL2 = "#86b6ef"
GRIS1, GRIS2, GRIS3 = "#55534e", "#9a9a94", "#d4d2cb"
INK, INK2, GRID = "#1a1a19", "#6b6a64", "#eceae5"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "font.size": 9, "axes.labelsize": 9,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.edgecolor": "#c9c7c0", "axes.linewidth": 0.7, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 110, "savefig.dpi": 600, "savefig.bbox": "tight",
    "legend.frameon": False, "legend.fontsize": 8.5,
})

FIG = "outputs/figures/"

aut = pd.read_csv("data/autores_pe_enlazados.csv", encoding="utf-8-sig")
ren = pd.read_csv("data/renacyt_limpio.csv", encoding="utf-8-sig")
cert = aut[aut.certificado == 1]
per = cert.groupby("CODIGO_RENACYT").agg(
    n_obras=("n_obras", "sum"), citas=("citas", "sum"),
    primer_anio=("primer_anio", "min"),
    field_dom=("field_dom", "first")).reset_index()
per = per.merge(ren[["CODIGO_RENACYT", "sexo", "REGLAMENTO", "nivel",
                     "nivel_orden", "rango_edad"]],
                on="CODIGO_RENACYT", how="left")
per = per[per.nivel_orden.notna()].copy()
per["citas_por_obra"] = per.citas / per.n_obras

p21 = per[per.REGLAMENTO == "Reglamento_2021"].copy()
NIV = ["VII", "VI", "V", "IV", "III", "II", "I", "Distinguido"]  # ascendente
LBL = ["VII", "VI", "V", "IV", "III", "II", "I", "Dist."]

# ---- tabla 1 (2021) ---------------------------------------------------------
rows = []
for n in NIV:
    g = p21[p21.nivel == n]
    rows.append({
        "nivel": n, "n": len(g),
        "obras_mediana": g.n_obras.median(),
        "obras_iqr": f"{g.n_obras.quantile(.25):.0f}-{g.n_obras.quantile(.75):.0f}",
        "citas_mediana": g.citas.median(),
        "citas_obra_mediana": round(g.citas_por_obra.median(), 1),
        "pct_mujeres": round(100 * (g.sexo == "Femenino").mean(), 1),
    })
pd.DataFrame(rows[::-1]).to_csv("outputs/tabla_nivel_2021.csv", index=False,
                                encoding="utf-8-sig")

# ---- fig 1: la escalera (medianas + IQR, obras y citas/obra) ---------------
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
for ax, col, lab in [(axes[0], "n_obras", "Works 2015–2026 (median, IQR)"),
                     (axes[1], "citas_por_obra",
                      "Citations per work (median, IQR)")]:
    med = [p21[p21.nivel == n][col].median() for n in NIV]
    q1 = [p21[p21.nivel == n][col].quantile(.25) for n in NIV]
    q3 = [p21[p21.nivel == n][col].quantile(.75) for n in NIV]
    x = np.arange(len(NIV))
    ax.errorbar(x, med, yerr=[np.array(med) - q1, np.array(q3) - med],
                fmt="o", color=AZUL, ms=6, capsize=3, lw=1.2,
                ecolor=GRIS2, mec="white", mew=0.8)
    ax.set_xticks(x, LBL)
    ax.set_xlabel("Certification level (ascending rank)")
    ax.set_ylabel(lab)
    ax.grid(axis="x", visible=False)
axes[0].text(0.02, 0.96, "a", transform=axes[0].transAxes, fontsize=11,
             fontweight="bold", va="top", color=INK)
axes[1].text(0.02, 0.96, "b", transform=axes[1].transAxes, fontsize=11,
             fontweight="bold", va="top", color=INK)
fig.tight_layout()
fig.savefig(FIG + "fig1_ladder.png"); plt.close(fig)
print("fig1 ok")

# ---- fig 2: AUC de pares adyacentes ----------------------------------------
pares, aucs, sigs = [], [], []
for a, b in zip(NIV[:-1], NIV[1:]):
    xa = p21[p21.nivel == b].n_obras.values   # nivel alto
    xb = p21[p21.nivel == a].n_obras.values   # nivel bajo
    u, p = stats.mannwhitneyu(xa, xb)
    pares.append(f"{LBL[NIV.index(b)]} vs {LBL[NIV.index(a)]}")
    aucs.append(u / (len(xa) * len(xb)))
    sigs.append(p < 0.01)
# extremos
xa = p21[p21.nivel == "Distinguido"].n_obras.values
xb = p21[p21.nivel == "VII"].n_obras.values
u, _ = stats.mannwhitneyu(xa, xb)
auc_ext = u / (len(xa) * len(xb))

fig, ax = plt.subplots(figsize=(6.4, 3.6))
y = np.arange(len(pares))
ax.axvline(0.5, color=GRIS2, lw=1, ls=(0, (3, 2)))
ax.axvline(auc_ext, color=AZUL2, lw=1, ls=(0, (3, 2)))
ax.text(0.502, len(pares) - 0.4, "coin flip", fontsize=7.5, color=INK2)
ax.text(auc_ext + 0.002, 0.1, f"top vs entry ({auc_ext:.2f})", fontsize=7.5,
        color=AZUL, rotation=90, va="bottom")
for yi, (auc, sig) in enumerate(zip(aucs, sigs)):
    ax.plot([0.5, auc], [yi, yi], color=GRIS3, lw=2, zorder=1)
    ax.scatter(auc, yi, s=70, color=AZUL if sig else GRIS2, zorder=3,
               edgecolors="white", linewidth=1)
    ax.annotate(f"{auc:.2f}" + ("" if sig else " (n.s.)"),
                (auc, yi), textcoords="offset points", xytext=(8, 0),
                fontsize=8, color=INK, va="center")
ax.set_yticks(y, pares)
ax.set_xlabel("Probability that the higher level out-produces the lower "
              "(AUC, works)")
ax.set_xlim(0.45, 0.9)
ax.grid(axis="y", visible=False)
fig.savefig(FIG + "fig2_rungs.png"); plt.close(fig)
print("fig2 ok")

# ---- fig 3: escalera de género ---------------------------------------------
pctf = [100 * (p21[p21.nivel == n].sexo == "Femenino").mean() for n in NIV]
glob = 100 * (p21.sexo == "Femenino").mean()
fig, ax = plt.subplots(figsize=(6.4, 3.4))
x = np.arange(len(NIV))
ax.bar(x, pctf, color=[AZUL if v < glob else GRIS3 for v in pctf], width=0.62)
for xi, v in zip(x, pctf):
    ax.annotate(f"{v:.0f}", (xi, v), ha="center", va="bottom", fontsize=7.5,
                color=INK2)
ax.axhline(glob, color=GRIS1, lw=1, ls=(0, (3, 2)))
ax.text(len(NIV) - 0.45, glob + 0.6, f"all certified ({glob:.0f}%)",
        fontsize=7.5, color=GRIS1, ha="right")
ax.set_xticks(x, LBL)
ax.set_xlabel("Certification level (ascending rank)")
ax.set_ylabel("Women (%)")
ax.set_ylim(0, 44)
ax.grid(axis="x", visible=False)
fig.savefig(FIG + "fig3_gender.png"); plt.close(fig)
print("fig3 ok")

# ---- fig 4: cohorte reciente vs todos --------------------------------------
joven = p21[p21.primer_anio >= 2018]
fig, ax = plt.subplots(figsize=(6.4, 3.6))
x = np.arange(len(NIV))
for d, col, lab, dx in [(p21, AZUL, "All certified (Reg. 2021)", -0.12),
                        (joven, GRIS2, "First publication ≥ 2018", 0.12)]:
    med = [d[d.nivel == n].n_obras.median() for n in NIV]
    ax.plot(x + dx, med, "o-", color=col, ms=5.5, lw=1.3, label=lab,
            mec="white", mew=0.8)
ax.set_xticks(x, LBL)
ax.set_xlabel("Certification level (ascending rank)")
ax.set_ylabel("Works 2015–2026 (median)")
ax.legend(loc="upper left")
ax.grid(axis="x", visible=False)
fig.savefig(FIG + "fig4_recency.png"); plt.close(fig)
print("fig4 ok")

# resumen para el texto
res = {"n_2021": int(len(p21)), "pct_mujeres_2021": round(float(glob), 1),
       "auc_extremos_2021": round(float(auc_ext), 3),
       "aucs_adyacentes": {p: round(float(a), 3)
                           for p, a in zip(pares, aucs)},
       "medianas_joven": {n: float(joven[joven.nivel == n].n_obras.median())
                          for n in NIV},
       "n_joven": int(len(joven))}
with open("outputs/resumen_figuras.json", "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print(json.dumps(res, ensure_ascii=False, indent=2))
