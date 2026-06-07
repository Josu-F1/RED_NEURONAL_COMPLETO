"""
ETL Sintético Separable — fact_ventas
Origen: fact_ventas.json (8000 registros reales)
Target: clase de rentabilidad  →  0=Pérdida  1=Normal  2=Alta Rentabilidad
Features reales usadas: margen_pct, descuento_pct, cantidad, precio_unitario,
                         costo_unitario, utilidad_bruta, region_natural,
                         canal, tipo_sucursal, categoria_producto
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

# ──────────────────────────────────────────────
# 1. CARGA Y APLANADO DEL JSON REAL
# ──────────────────────────────────────────────
with open("fact_ventas.json") as f:
    raw = json.load(f)

records = []
for r in raw:
    records.append({
        "margen_pct":           r.get("margen_pct", 0),
        "descuento_pct":        r.get("descuento_pct", 0),
        "cantidad":             r.get("cantidad", 0),
        "precio_unitario":      r.get("precio_unitario", 0),
        "costo_unitario":       r.get("costo_unitario", 0),
        "utilidad_bruta":       r.get("utilidad_bruta", 0),
        "region_natural":       r["sucursal"].get("region_natural", "SIERRA"),
        "tipo_sucursal":        r["sucursal"].get("tipo_sucursal", "MINORISTA"),
        "canal":                r["canal"].get("nombre_canal", "VENTA DIRECTA"),
        "categoria_producto":   r["producto"].get("categoria", "GENERAL"),
        "es_fin_semana":        r["tiempo"].get("es_fin_semana", 0),
    })

df_real = pd.DataFrame(records)

# ──────────────────────────────────────────────
# 2. SCORE DE RENTABILIDAD (lógica de negocio real)
# ──────────────────────────────────────────────
score = pd.Series(np.zeros(len(df_real)))

score += df_real["margen_pct"] * 4.0          # Margen: factor más importante
score -= df_real["descuento_pct"] * 2.0       # Descuentos altos dañan
score += (df_real["utilidad_bruta"] / df_real["utilidad_bruta"].max()) * 2.0

region_map = {"COSTA": 1.5, "SIERRA": 1.0, "ORIENTE": -1.0, "INSULAR": -1.5}
score += df_real["region_natural"].map(region_map).fillna(0)

canal_map = {"VENTA DIRECTA": 1.5, "E-COMMERCE": 1.0, "MAYORISTA": 0.5,
             "TELEFÓNICO": -0.5}
score += df_real["canal"].map(canal_map).fillna(0)

tipo_map = {"PRINCIPAL": 1.5, "MINORISTA": 0.5, "EXPRESS": -0.5}
score += df_real["tipo_sucursal"].map(tipo_map).fillna(0)

# Clases por percentiles (balanceadas)
p33 = np.percentile(score, 33)
p66 = np.percentile(score, 66)
df_real["clase"] = 1
df_real.loc[score < p33, "clase"] = 0
df_real.loc[score >= p66, "clase"] = 2

# ──────────────────────────────────────────────
# 3. ENCODING NUMÉRICO DE CATEGÓRICAS
# ──────────────────────────────────────────────
ENCODINGS = {
    "region_natural": {"COSTA": 3, "SIERRA": 2, "ORIENTE": 1, "INSULAR": 0},
    "tipo_sucursal":  {"PRINCIPAL": 2, "MINORISTA": 1, "EXPRESS": 0},
    "canal":          {"VENTA DIRECTA": 3, "E-COMMERCE": 2, "MAYORISTA": 1, "TELEFÓNICO": 0},
}
for col, mapping in ENCODINGS.items():
    df_real[col] = df_real[col].map(mapping).fillna(1).astype(int)

df_real["categoria_producto"] = pd.Categorical(df_real["categoria_producto"]).codes

FEATURES = ["margen_pct", "descuento_pct", "cantidad", "precio_unitario",
            "costo_unitario", "utilidad_bruta", "region_natural",
            "tipo_sucursal", "canal", "categoria_producto", "es_fin_semana"]

# ──────────────────────────────────────────────
# 4. GENERACIÓN SINTÉTICA — GAPS AGRESIVOS POR CLASE
# ──────────────────────────────────────────────
CLASS_PARAMS = {
    # Clase 0: ventas en pérdida — descuentos altos, margen negativo, canal débil
    0: {
        "margen_pct":        (0.00, 0.06),
        "descuento_pct":     (0.18, 0.35),
        "cantidad":          (1, 10),
        "precio_unitario":   (80, 200),
        "costo_unitario":    (75, 195),
        "utilidad_bruta":    (-200, 50),
        "region_natural":    0,   # INSULAR
        "tipo_sucursal":     0,   # EXPRESS
        "canal":             0,   # TELEFÓNICO
        "categoria_producto":0,
        "es_fin_semana":     0,
    },
    # Clase 1: ventas normales — márgenes medios
    1: {
        "margen_pct":        (0.20, 0.32),
        "descuento_pct":     (0.05, 0.12),
        "cantidad":          (15, 35),
        "precio_unitario":   (25, 60),
        "costo_unitario":    (18, 45),
        "utilidad_bruta":    (100, 400),
        "region_natural":    2,   # SIERRA
        "tipo_sucursal":     1,   # MINORISTA
        "canal":             1,   # MAYORISTA
        "categoria_producto":1,
        "es_fin_semana":     0,
    },
    # Clase 2: alta rentabilidad — márgenes altos, sin descuento, canal premium
    2: {
        "margen_pct":        (0.42, 0.65),
        "descuento_pct":     (0.00, 0.03),
        "cantidad":          (45, 100),
        "precio_unitario":   (5, 22),
        "costo_unitario":    (2, 12),
        "utilidad_bruta":    (600, 2000),
        "region_natural":    3,   # COSTA
        "tipo_sucursal":     2,   # PRINCIPAL
        "canal":             3,   # VENTA DIRECTA
        "categoria_producto":2,
        "es_fin_semana":     1,
    },
}

N_SYN_PER_CLASS = 2500
synthetic_rows = []

for clase, params in CLASS_PARAMS.items():
    n = N_SYN_PER_CLASS
    std = 0.10  # ruido controlado

    rows = {
        "margen_pct":        np.clip(np.random.normal(np.mean(params["margen_pct"]),
                                     (params["margen_pct"][1]-params["margen_pct"][0])*std, n),
                                     *params["margen_pct"]),
        "descuento_pct":     np.clip(np.random.normal(np.mean(params["descuento_pct"]),
                                     (params["descuento_pct"][1]-params["descuento_pct"][0])*std, n),
                                     *params["descuento_pct"]),
        "cantidad":          np.clip(np.random.normal(np.mean(params["cantidad"]),
                                     (params["cantidad"][1]-params["cantidad"][0])*std, n),
                                     *params["cantidad"]).astype(int),
        "precio_unitario":   np.clip(np.random.normal(np.mean(params["precio_unitario"]),
                                     (params["precio_unitario"][1]-params["precio_unitario"][0])*std, n),
                                     *params["precio_unitario"]),
        "costo_unitario":    np.clip(np.random.normal(np.mean(params["costo_unitario"]),
                                     (params["costo_unitario"][1]-params["costo_unitario"][0])*std, n),
                                     *params["costo_unitario"]),
        "utilidad_bruta":    np.clip(np.random.normal(np.mean(params["utilidad_bruta"]),
                                     (params["utilidad_bruta"][1]-params["utilidad_bruta"][0])*std, n),
                                     *params["utilidad_bruta"]),
        "region_natural":    np.full(n, params["region_natural"]),
        "tipo_sucursal":     np.full(n, params["tipo_sucursal"]),
        "canal":             np.full(n, params["canal"]),
        "categoria_producto":np.full(n, params["categoria_producto"]),
        "es_fin_semana":     np.full(n, params["es_fin_semana"]),
        "clase":             np.full(n, clase),
        "origen":            np.full(n, "SINTETICO", dtype=object),
    }
    synthetic_rows.append(pd.DataFrame(rows))

df_syn = pd.concat(synthetic_rows, ignore_index=True)

# ──────────────────────────────────────────────
# 5. UNIÓN REAL + SINTÉTICO
# ──────────────────────────────────────────────
df_real["origen"] = "REAL"
df_final = pd.concat([df_real[FEATURES + ["clase", "origen"]], df_syn], ignore_index=True)

# ──────────────────────────────────────────────
# 6. GUARDAR
# ──────────────────────────────────────────────
output = df_final.to_dict(orient="records")
with open("fact_ventas_optimizado.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"✅ fact_ventas_optimizado.json guardado")
print(f"   Total registros : {len(df_final)}")
print(f"   Reales          : {(df_final['origen']=='REAL').sum()}")
print(f"   Sintéticos      : {(df_final['origen']=='SINTETICO').sum()}")
print(f"   Distribución clase:\n{df_final['clase'].value_counts().sort_index()}")

# ──────────────────────────────────────────────
# 7. t-SNE PREVIEW
# ──────────────────────────────────────────────
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    X = df_final[FEATURES].values
    y = df_final["clase"].values
    origen = df_final["origen"].values

    X_sc = StandardScaler().fit_transform(X)
    tsne = TSNE(n_components=2, perplexity=40, learning_rate=150,
                max_iter=1000, random_state=42)
    X_2d = tsne.fit_transform(X_sc)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = {0: "#e74c3c", 1: "#f39c12", 2: "#27ae60"}
    labels = {0: "Clase 0 – Pérdida", 1: "Clase 1 – Normal", 2: "Clase 2 – Alta Rentab."}

    for c in [0, 1, 2]:
        mask = y == c
        axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1],
                        c=colors[c], label=labels[c], alpha=0.4, s=8)
    axes[0].set_title("t-SNE por Clase — fact_ventas")
    axes[0].legend(markerscale=3)

    for og, col in [("REAL", "#2980b9"), ("SINTETICO", "#8e44ad")]:
        mask = origen == og
        axes[1].scatter(X_2d[mask, 0], X_2d[mask, 1],
                        c=col, label=og, alpha=0.35, s=6)
    axes[1].set_title("t-SNE Real vs Sintético")
    axes[1].legend(markerscale=3)

    plt.tight_layout()
    plt.savefig("tsne_ventas.png", dpi=150)
    print("📊 tsne_ventas.png guardado")
except Exception as e:
    print(f"⚠️  t-SNE no generado: {e}")
