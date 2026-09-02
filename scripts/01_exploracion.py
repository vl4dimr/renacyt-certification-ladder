# -*- coding: utf-8 -*-
"""Exploración: ¿qué mide el nivel de certificación RENACYT?
Une los certificados enlazados (producción OpenAlex 2015-2026) con los
atributos del registro (nivel, edad, reglamento, fecha de calificación)
y contrasta nivel vs producción, impacto y antigüedad.
Salida: outputs/exploracion.json (+ tablas CSV)."""
import json

import numpy as np
import pandas as pd
from scipy import stats

aut = pd.read_csv("data/autores_pe_enlazados.csv", encoding="utf-8-sig")
ren = pd.read_csv("data/renacyt_limpio.csv", encoding="utf-8-sig")

cert = aut[aut.certificado == 1].copy()

# persona única: agregar perfiles fragmentados por codigo RENACYT
per = cert.groupby("CODIGO_RENACYT").agg(
    n_obras=("n_obras", "sum"),
    citas=("citas", "sum"),
    primer_anio=("primer_anio", "min"),
    ultimo_anio=("ultimo_anio", "max"),
    perfiles=("author_id", "count"),
    field_dom=("field_dom", "first"),
    subfield_dom=("subfield_dom", "first"),
).reset_index()

per = per.merge(
    ren[["CODIGO_RENACYT", "sexo", "rango_edad", "REGLAMENTO", "condicion",
         "nivel", "nivel_orden", "anio_calificacion", "region", "macrozona"]],
    on="CODIGO_RENACYT", how="left")

per["citas_por_obra"] = per.citas / per.n_obras
per["anios_activo"] = per.ultimo_anio - per.primer_anio + 1
per["edad_carrera_2015"] = 2026 - per.primer_anio  # censurada en 2015
per["lag_calificacion"] = per.anio_calificacion - per.primer_anio

res = {"n_personas": int(len(per)),
       "n_con_nivel": int(per.nivel.notna().sum()),
       "reglamentos": per.REGLAMENTO.value_counts().to_dict(),
       "condicion": per.condicion.value_counts().to_dict(),
       "niveles_disponibles": sorted(per.nivel.dropna().unique().tolist())}

# ---- por nivel: producción, impacto, antigüedad, demografía ----------------
per_n = per[per.nivel.notna() & per.nivel_orden.notna()].copy()
tabla = []
for nivel, g in per_n.groupby("nivel"):
    tabla.append({
        "nivel": nivel,
        "orden": float(g.nivel_orden.iloc[0]),
        "n": int(len(g)),
        "obras_mediana": float(g.n_obras.median()),
        "obras_p90": float(g.n_obras.quantile(0.9)),
        "citas_mediana": float(g.citas.median()),
        "citas_obra_mediana": float(g.citas_por_obra.median()),
        "edad_carrera_mediana": float(g.edad_carrera_2015.median()),
        "lag_calif_mediana": float(g.lag_calificacion.median()),
        "pct_mujeres": round(100 * (g.sexo == "Femenino").mean(), 1),
        "edad_modal": g.rango_edad.mode().iloc[0] if len(g.rango_edad.mode()) else None,
        "anio_calif_mediana": float(g.anio_calificacion.median()),
    })
tabla = sorted(tabla, key=lambda d: d["orden"])
res["por_nivel"] = tabla
pd.DataFrame(tabla).to_csv("outputs/tabla_por_nivel.csv", index=False,
                           encoding="utf-8-sig")

# ---- correlaciones globales nivel vs indicadores ---------------------------
# nivel_orden: verificar dirección con los datos (¿1 = top?)
def spear(a, b):
    m = per_n[a].notna() & per_n[b].notna()
    r, p = stats.spearmanr(per_n.loc[m, a], per_n.loc[m, b])
    return {"rho": round(float(r), 3), "p": float(p), "n": int(m.sum())}

res["correlaciones_nivel_orden"] = {
    "n_obras": spear("nivel_orden", "n_obras"),
    "citas": spear("nivel_orden", "citas"),
    "citas_por_obra": spear("nivel_orden", "citas_por_obra"),
    "edad_carrera": spear("nivel_orden", "edad_carrera_2015"),
    "lag_calificacion": spear("nivel_orden", "lag_calificacion"),
}

# correlaciones parciales aproximadas: nivel~produccion controlando carrera
# (Spearman sobre residuos de rangos)
def spearman_parcial(x, y, z):
    d = per_n[[x, y, z]].dropna()
    rx = d[x].rank(); ry = d[y].rank(); rz = d[z].rank()
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    r, p = stats.spearmanr(ex, ey)
    return {"rho": round(float(r), 3), "p": float(p), "n": int(len(d))}

res["parciales"] = {
    "nivel_vs_obras_ctrl_carrera":
        spearman_parcial("nivel_orden", "n_obras", "edad_carrera_2015"),
    "nivel_vs_citasobra_ctrl_carrera":
        spearman_parcial("nivel_orden", "citas_por_obra", "edad_carrera_2015"),
    "nivel_vs_carrera_ctrl_obras":
        spearman_parcial("nivel_orden", "edad_carrera_2015", "n_obras"),
}

# ---- solapamiento entre niveles: ¿separan la produccion? -------------------
# prob. de que un nivel alto supere a uno bajo en obras (AUC por pares adyacentes)
niveles_ord = [t["nivel"] for t in tabla]
solap = []
for a, b in zip(niveles_ord[:-1], niveles_ord[1:]):
    xa = per_n[per_n.nivel == a].n_obras.values
    xb = per_n[per_n.nivel == b].n_obras.values
    if len(xa) < 10 or len(xb) < 10:
        continue
    u, p = stats.mannwhitneyu(xa, xb, alternative="two-sided")
    auc = u / (len(xa) * len(xb))
    solap.append({"nivel_alto": a, "nivel_bajo": b,
                  "auc_obras": round(float(auc), 3), "p": float(p),
                  "n_alto": int(len(xa)), "n_bajo": int(len(xb))})
res["solapamiento_adyacente"] = solap

# extremos: top nivel vs nivel de entrada
top, low = niveles_ord[0], niveles_ord[-1]
xa = per_n[per_n.nivel == top].n_obras
xb = per_n[per_n.nivel == low].n_obras
u, _ = stats.mannwhitneyu(xa, xb, alternative="two-sided")
res["auc_extremos_obras"] = {
    "niveles": [top, low], "auc": round(float(u / (len(xa) * len(xb))), 3)}

# ---- dispersión dentro de nivel: % de nivel bajo que supera mediana top ----
med_top = per_n[per_n.nivel == top].n_obras.median()
res["pct_entrada_supera_mediana_top"] = round(
    100 * (per_n[per_n.nivel == low].n_obras > med_top).mean(), 1)
med_low = per_n[per_n.nivel == low].n_obras.median()
res["pct_top_bajo_mediana_entrada"] = round(
    100 * (per_n[per_n.nivel == top].n_obras < med_low).mean(), 1)

# ---- por campo: correlacion nivel-obras dentro de cada campo ---------------
por_campo = []
for campo, g in per_n.groupby("field_dom"):
    if len(g) < 100:
        continue
    r, p = stats.spearmanr(g.nivel_orden, g.n_obras)
    por_campo.append({"campo": campo, "n": int(len(g)),
                      "rho_nivel_obras": round(float(r), 3), "p": float(p)})
res["por_campo"] = sorted(por_campo, key=lambda d: d["rho_nivel_obras"])

# ---- género por nivel (piramide) -------------------------------------------
res["pct_mujeres_global"] = round(
    100 * (per_n.sexo == "Femenino").mean(), 1)

with open("outputs/exploracion.json", "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2, default=str)
print(json.dumps(res, ensure_ascii=False, indent=2, default=str)[:4500])
