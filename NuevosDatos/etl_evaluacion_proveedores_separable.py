"""
ETL Sintético Separable — fact_evaluacion_proveedores
Origen: fact_evaluacion_proveedores.json (1500 registros reales)
Target: clase de desempeño proveedor → 0=Mal proveedor  1=Regular  2=Excelente
Features reales: tasa_rechazo_pct, tiempo_entrega_dias, entregado_a_tiempo,
                 calificacion_entrega, costo_adquisicion_total,
                 calificacion_proveedor, tipo_contrato, region_natural,
                 margen_ganancia_producto
"""

import json
import numpy as np
import pandas as pd

np.random.seed(42)

# ──────────────────────────────────────────────
# 1. CARGA Y APLANADO
# ──────────────────────────────────────────────
with open("fact_evaluacion_proveedores.json") as f:
    raw = json.load(f)

records = []
for r in raw:
    records.append({
        "tasa_rechazo_pct":        r.get("tasa_rechazo_pct", 0),
        "tiempo_entrega_dias":     r.get("tiempo_entrega_dias", 0),
        "entregado_a_tiempo":      r.get("entregado_a_tiempo", 0),
        "calificacion_entrega":    r.get("calificacion_entrega", 1),
        "costo_adquisicion_total": r.get("costo_adquisicion_total", 0),
        "calificacion_proveedor":  r["proveedor"].get("calificacion", "B"),
        "tipo_contrato":           r["proveedor"].get("tipo_contrato", "MENSUAL"),
        "region_natural":          r["sucursal"].get("region_natural", "SIERRA"),
        "margen_ganancia":         r["producto"].get("margen_ganancia", 0),
        "estado_producto":         r["producto"].get("estado", "ACTIVO"),
    })

df_real = pd.DataFrame(records)

# ──────────────────────────────────────────────
# 2. SCORE DE DESEMPEÑO DEL PROVEEDOR
# ──────────────────────────────────────────────
score = pd.Series(np.zeros(len(df_real)))

score -= df_real["tasa_rechazo_pct"] * 5.0        # Rechazo es lo peor
score -= df_real["tiempo_entrega_dias"] * 0.15    # Más días = peor
score += df_real["entregado_a_tiempo"] * 2.0
score += df_real["calificacion_entrega"] * 1.5

cal_map = {"A+": 3.0, "A": 2.5, "B+": 1.5, "B": 1.0, "C": -2.0}
score += df_real["calificacion_proveedor"].map(cal_map).fillna(0)

contrato_map = {"CONTRATO ANUAL": 2.0, "SEMESTRAL": 1.0,
                "MENSUAL": 0.0, "ORDEN DE COMPRA": -1.5}
score += df_real["tipo_contrato"].map(contrato_map).fillna(0)

region_map = {"COSTA": 1.0, "SIERRA": 0.5, "ORIENTE": -1.0, "INSULAR": -1.5}
score += df_real["region_natural"].map(region_map).fillna(0)

score += df_real["margen_ganancia"] * 1.5

p33 = np.percentile(score, 33)
p66 = np.percentile(score, 66)
df_real["clase"] = 1
df_real.loc[score < p33, "clase"] = 0
df_real.loc[score >= p66, "clase"] = 2

# ──────────────────────────────────────────────
# 3. ENCODING
# ──────────────────────────────────────────────
ENCODINGS = {
    "calificacion_proveedor": {"A+": 4, "A": 3, "B+": 2, "B": 1, "C": 0},
    "tipo_contrato":          {"CONTRATO ANUAL": 3, "SEMESTRAL": 2, "MENSUAL": 1, "ORDEN DE COMPRA": 0},
    "region_natural":         {"COSTA": 3, "SIERRA": 2, "ORIENTE": 1, "INSULAR": 0},
    "estado_producto":        {"ACTIVO": 1, "DESCONTINUADO": 0},
}
for col, mapping in ENCODINGS.items():
    df_real[col] = df_real[col].map(mapping).fillna(1).astype(int)

FEATURES = ["tasa_rechazo_pct", "tiempo_entrega_dias", "entregado_a_tiempo",
            "calificacion_entrega", "costo_adquisicion_total",
            "calificacion_proveedor", "tipo_contrato", "region_natural",
            "margen_ganancia", "estado_producto"]

# ──────────────────────────────────────────────
# 4. SINTÉTICOS CON GAPS AGRESIVOS
# ──────────────────────────────────────────────
CLASS_PARAMS = {
    # Clase 0: proveedor malo — muchos rechazos, entrega tardía, calificación C
    0: {
        "tasa_rechazo_pct":        (0.12, 0.30),
        "tiempo_entrega_dias":     (18, 30),
        "entregado_a_tiempo":      0,
        "calificacion_entrega":    (1, 2),
        "costo_adquisicion_total": (25000, 60000),
        "calificacion_proveedor":  0,   # C
        "tipo_contrato":           0,   # ORDEN DE COMPRA
        "region_natural":          0,   # INSULAR
        "margen_ganancia":         (0.02, 0.12),
        "estado_producto":         0,
    },
    # Clase 1: proveedor regular
    1: {
        "tasa_rechazo_pct":        (0.04, 0.09),
        "tiempo_entrega_dias":     (8, 14),
        "entregado_a_tiempo":      1,
        "calificacion_entrega":    (2, 3),
        "costo_adquisicion_total": (8000, 20000),
        "calificacion_proveedor":  2,   # B+
        "tipo_contrato":           1,   # MENSUAL
        "region_natural":          2,   # SIERRA
        "margen_ganancia":         (0.22, 0.38),
        "estado_producto":         1,
    },
    # Clase 2: proveedor excelente — bajo rechazo, entrega puntual, A+
    2: {
        "tasa_rechazo_pct":        (0.00, 0.015),
        "tiempo_entrega_dias":     (1, 4),
        "entregado_a_tiempo":      1,
        "calificacion_entrega":    (4, 5),
        "costo_adquisicion_total": (500, 5000),
        "calificacion_proveedor":  4,   # A+
        "tipo_contrato":           3,   # CONTRATO ANUAL
        "region_natural":          3,   # COSTA
        "margen_ganancia":         (0.45, 0.72),
        "estado_producto":         1,
    },
}

N_SYN_PER_CLASS = 800
synthetic_rows = []

for clase, params in CLASS_PARAMS.items():
    n = N_SYN_PER_CLASS
    std = 0.08

    def gen(key):
        lo, hi = params[key]
        return np.clip(np.random.normal(np.mean([lo, hi]), (hi - lo) * std, n), lo, hi)

    rows = {
        "tasa_rechazo_pct":        np.clip(gen("tasa_rechazo_pct"), 0, 1),
        "tiempo_entrega_dias":     gen("tiempo_entrega_dias").astype(int),
        "entregado_a_tiempo":      np.full(n, params["entregado_a_tiempo"]),
        "calificacion_entrega":    gen("calificacion_entrega").astype(int),
        "costo_adquisicion_total": gen("costo_adquisicion_total"),
        "calificacion_proveedor":  np.full(n, params["calificacion_proveedor"]),
        "tipo_contrato":           np.full(n, params["tipo_contrato"]),
        "region_natural":          np.full(n, params["region_natural"]),
        "margen_ganancia":         gen("margen_ganancia"),
        "estado_producto":         np.full(n, params["estado_producto"]),
        "clase":                   np.full(n, clase),
        "origen":                  np.full(n, "SINTETICO", dtype=object),
    }
    synthetic_rows.append(pd.DataFrame(rows))

df_syn = pd.concat(synthetic_rows, ignore_index=True)

# ──────────────────────────────────────────────
# 5. UNIÓN Y GUARDADO
# ──────────────────────────────────────────────
df_real["origen"] = "REAL"
df_final = pd.concat([df_real[FEATURES + ["clase", "origen"]], df_syn], ignore_index=True)

output = df_final.to_dict(orient="records")
with open("fact_evaluacion_proveedores_optimizado.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"✅ fact_evaluacion_proveedores_optimizado.json guardado")
print(f"   Total: {len(df_final)} | Real: {(df_final['origen']=='REAL').sum()} | Sintético: {(df_final['origen']=='SINTETICO').sum()}")
print(f"   Clases:\n{df_final['clase'].value_counts().sort_index()}")

# ──────────────────────────────────────────────
# 6. t-SNE PREVIEW
# ──────────────────────────────────────────────
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    X = df_final[FEATURES].values
    y = df_final["clase"].values
    origen = df_final["origen"].values

    X_sc = StandardScaler().fit_transform(X)
    X_2d = TSNE(n_components=2, perplexity=40, learning_rate=150,
                max_iter=1000, random_state=42).fit_transform(X_sc)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = {0: "#e74c3c", 1: "#f39c12", 2: "#27ae60"}
    labels = {0: "Clase 0 – Mal proveedor", 1: "Clase 1 – Regular", 2: "Clase 2 – Excelente"}

    for c in [0, 1, 2]:
        mask = y == c
        axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1],
                        c=colors[c], label=labels[c], alpha=0.4, s=8)
    axes[0].set_title("t-SNE por Clase — fact_evaluacion_proveedores")
    axes[0].legend(markerscale=3)

    for og, col in [("REAL", "#2980b9"), ("SINTETICO", "#8e44ad")]:
        mask = origen == og
        axes[1].scatter(X_2d[mask, 0], X_2d[mask, 1],
                        c=col, label=og, alpha=0.35, s=6)
    axes[1].set_title("t-SNE Real vs Sintético")
    axes[1].legend(markerscale=3)

    plt.tight_layout()
    plt.savefig("tsne_evaluacion_proveedores.png", dpi=150)
    print("📊 tsne_evaluacion_proveedores.png guardado")
except Exception as e:
    print(f"⚠️  t-SNE no generado: {e}")
