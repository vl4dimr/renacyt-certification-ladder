# -*- coding: utf-8 -*-
"""Sensibilidades del análisis de niveles:
(a) solo condicion == Activo; (b) solo Reglamento 2021; (c) cohorte joven
(primer_anio >= 2018, carrera casi contenida en la ventana 2015-2026);
(d) producción normalizada por campo (percentil dentro del campo).
Salida: outputs/sensibilidad.json"""
import json

import numpy as np
import pandas as pd
from scipy import stats

aut = pd.read_csv("data/autores_pe_enlazados.csv", encoding="utf-8-sig")
ren = pd.read_csv("data/renacyt_limpio.csv", encoding="utf-8-sig")

cert = aut[aut.certificado == 1]
per = cert.groupby("CODIGO_RENACYT").agg(
    n_obras=("n_obras", "sum"), citas=("citas", "sum"),
    primer_anio=("primer_anio", "min"), ultimo_anio=("ultimo_anio", "max"),
    field_dom=("field_dom", "first")).reset_index()
per = per.merge(
    ren[["CODIGO_RENACYT", "sexo", "REGLAMENTO", "condicion", "nivel",
         "nivel_orden", "anio_calificacion"]],
    on="CODIGO_RENACYT", how="left")
per = per[per.nivel_orden.notna()].copy()
per["citas_por_obra"] = per.citas / per.n_obras
per["edad_carrera"] = 2026 - per.primer_anio

# percentil de obras dentro del campo dominante
per["pct_obras_campo"] = per.groupby("field_dom").n_obras.rank(pct=True)


def resumen(d, etiqueta):
    r1, p1 = stats.spearmanr(d.nivel_orden, d.n_obras)
    r2, p2 = stats.spearmanr(d.nivel_orden, d.citas)
    r3, p3 = stats.spearmanr(d.nivel_orden, d.citas_por_obra)
    m4 = d.pct_obras_campo.notna()
    r4, p4 = stats.spearmanr(d.nivel_orden[m4], d.pct_obras_campo[m4])
    # AUC extremos y % techo
    top = d[d.nivel == "Distinguido"].n_obras
    low = d[d.nivel == "VII"].n_obras
    auc = None
    pct_top_bajo = None
    if len(top) >= 10 and len(low) >= 10:
        u, _ = stats.mannwhitneyu(top, low)
        auc = round(float(u / (len(top) * len(low))), 3)
        pct_top_bajo = round(100 * (top < low.median()).mean(), 1)
    # AUC adyacentes promedio
    orden = (d.groupby("nivel").nivel_orden.first().sort_values().index.tolist())
    aucs = []
    for a, b in zip(orden[:-1], orden[1:]):
        xa, xb = d[d.nivel == a].n_obras, d[d.nivel == b].n_obras
        if len(xa) >= 10 and len(xb) >= 10:
            u, _ = stats.mannwhitneyu(xa, xb)
            aucs.append(u / (len(xa) * len(xb)))
    # gradiente de género
    gen = {niv: round(100 * (g.sexo == "Femenino").mean(), 1)
           for niv, g in d.groupby("nivel") if len(g) >= 30}
    return {
        "etiqueta": etiqueta, "n": int(len(d)),
        "rho_nivel_obras": round(float(r1), 3),
        "rho_nivel_citas": round(float(r2), 3),
        "rho_nivel_citas_obra": round(float(r3), 3),
        "rho_nivel_pct_obras_campo": round(float(r4), 3),
        "auc_extremos": auc,
        "pct_distinguido_bajo_mediana_vii": pct_top_bajo,
        "auc_adyacente_media": round(float(np.mean(aucs)), 3) if aucs else None,
        "auc_adyacente_max": round(float(np.max(aucs)), 3) if aucs else None,
        "pct_mujeres_por_nivel": gen,
    }


res = {
    "base": resumen(per, "todos (n=10,366)"),
    "activos": resumen(per[per.condicion == "Activo"], "solo Activo"),
    "reglamento_2021": resumen(per[per.REGLAMENTO == "Reglamento_2021"],
                               "solo Reglamento 2021"),
    "cohorte_joven": resumen(per[per.primer_anio >= 2018],
                             "primer_anio >= 2018 (carrera en ventana)"),
}

# lag: nivel vs años entre 1a publicación (censurada) y calificación
d = per[per.anio_calificacion.notna()].copy()
d["lag"] = d.anio_calificacion - d.primer_anio
r, p = stats.spearmanr(d.nivel_orden, d.lag)
res["lag_calificacion"] = {"rho_nivel_lag": round(float(r), 3),
                           "lag_mediana_global": float(d.lag.median())}

with open("outputs/sensibilidad.json", "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print(json.dumps(res, ensure_ascii=False, indent=2))
