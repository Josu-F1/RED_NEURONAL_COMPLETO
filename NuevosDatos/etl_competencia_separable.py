"""
ETL Sintético Separable — fact_competencia
Origen: fact_competencia.json (1200 registros reales)
Target: clase de posición competitiva → 0=Desventaja  1=Paridad  2=Ventaja
Features reales: precio_nuestro, precio_competidor, diferencia_precio,
                 pct_diferencia, margen_ganancia, region, categoria_producto,
                 nombre_competidor
"""

import json
import numpy as np
import pandas as pd

np.random.seed(42)

# ──────────────────────────────────────────────
# 1. CARGA Y APLANADO
# ──────────────────────────────────────────────
with open("fact_competencia.json") as f:
    raw = json.load(f)

records = []
for r in raw:
    records.append({
        "precio_nuestro":       r.get("precio_nuestro", 0),
        "precio_competidor":    r.get("precio_competidor", 0),
        "diferencia_precio":    r.get("diferencia_precio", 0),
        "pct_diferencia":       r.get("pct_diferencia", 0),
        "margen_ganancia":      r["producto"].get("margen_ganancia", 0),
        "precio_costo":         r["producto"].get("precio_costo", 0),
        "region":               r.get("region", "Sierra"),
        "categoria_producto":   r["producto"].get("categoria", "GENERAL"),
        "nombre_competidor":    r["competidor"].get("nombre_competidor", "OTRO"),
        "anio_ingreso":         r["producto"].get("anio_ingreso", 2020),
    })

df_real = pd.DataFrame(records)

# ──────────────────────────────────────────────
# 2. SCORE DE POSICIÓN COMPETITIVA
# ──────────────────────────────────────────────
score = pd.Series(np.zeros(len(df_real)))

# pct_diferencia negativo = somos más baratos = ventaja
score -= df_real["pct_diferencia"] * 4.0
# Margen alto = podemos bajar precio y aun ganar = ventaja estratégica
score += df_real["margen_ganancia"] * 3.0
# Producto más nuevo = más ventaja (características)
score += (df_real["anio_ingreso"] - 2018) * 0.3

region_map = {"Costa": 1.0, "Sierra": 0.5, "Oriente": -0.5,
              "Insular": -1.0, "Costa ": 1.0}
score += df_real["region"].map(region_map).fillna(0)

# Competidores fuertes restan puntos
comp_map = {"MEGAMAXI": -1.5, "SUPERMAXI": -1.5, "MI COMISARIATO": -1.0,
            "AKI": -0.5, "TIA": -0.5}
score += df_real["nombre_competidor"].map(comp_map).fillna(0)

p33 = np.percentile(score, 33)
p66 = np.percentile(score, 66)
df_real["clase"] = 1
df_real.loc[score < p33, "clase"] = 0
df_real.loc[score >= p66, "clase"] = 2

# ──────────────────────────────────────────────
# 3. ENCODING
# ──────────────────────────────────────────────
ENCODINGS = {
    "region": {"Costa": 3, "Sierra": 2, "Oriente": 1, "Insular": 0, "Costa ": 3},
    "nombre_competidor": {"MEGAMAXI": 0, "SUPERMAXI": 1, "MI COMISARIATO": 2,
                          "AKI": 3, "TIA": 4},
}
for col, mapping in ENCODINGS.items():
    df_real[col] = df_real[col].map(mapping).fillna(2).astype(int)
df_real["categoria_producto"] = pd.Categorical(df_real["categoria_producto"]).codes

FEATURES = ["precio_nuestro", "precio_competidor", "diferencia_precio",
            "pct_diferencia", "margen_ganancia", "precio_costo",
            "region", "categoria_producto", "nombre_competidor", "anio_ingreso"]

# ──────────────────────────────────────────────
# 4. SINTÉTICOS CON GAPS AGRESIVOS
# ──────────────────────────────────────────────
CLASS_PARAMS = {
    # Clase 0: en desventaja — somos más caros, margen bajo, competidor fuerte
    0: {
        "precio_nuestro":    (40, 100),
        "precio_competidor": (25, 70),
        "diferencia_precio": (8, 30),    # nosotros más caros
        "pct_diferencia":    (0.20, 0.50),
        "margen_ganancia":   (0.02, 0.12),
        "precio_costo":      (35, 95),
        "region":            0,          # Insular
        "categoria_producto":0,
        "nombre_competidor": 0,          # MEGAMAXI (el más duro)
        "anio_ingreso":      2018,
    },
    # Clase 1: paridad competitiva
    1: {
        "precio_nuestro":    (20, 45),
        "precio_competidor": (18, 42),
        "diferencia_precio": (-2, 4),
        "pct_diferencia":    (-0.05, 0.08),
        "margen_ganancia":   (0.25, 0.38),
        "precio_costo":      (13, 35),
        "region":            2,          # Sierra
        "categoria_producto":1,
        "nombre_competidor": 2,          # MI COMISARIATO
        "anio_ingreso":      2021,
    },
    # Clase 2: ventaja competitiva — somos más baratos con buen margen
    2: {
        "precio_nuestro":    (5, 18),
        "precio_competidor": (8, 25),
        "diferencia_precio": (-10, -2),  # somos más baratos
        "pct_diferencia":    (-0.40, -0.15),
        "margen_ganancia":   (0.50, 0.75),
        "precio_costo":      (2, 10),
        "region":            3,          # Costa
        "categoria_producto":2,
        "nombre_competidor": 4,          # TIA (competidor débil)
        "anio_ingreso":      2023,
    },
}

N_SYN_PER_CLASS = 700
synthetic_rows = []

for clase, params in CLASS_PARAMS.items():
    n = N_SYN_PER_CLASS
    std = 0.08

    def gen(key):
        lo, hi = params[key]
        return np.clip(np.random.normal(np.mean([lo, hi]), abs(hi - lo) * std, n), min(lo, hi), max(lo, hi))

    rows = {
        "precio_nuestro":    gen("precio_nuestro"),
        "precio_competidor": gen("precio_competidor"),
        "diferencia_precio": gen("diferencia_precio"),
        "pct_diferencia":    gen("pct_diferencia"),
        "margen_ganancia":   gen("margen_ganancia"),
        "precio_costo":      gen("precio_costo"),
        "region":            np.full(n, params["region"]),
        "categoria_producto":np.full(n, params["categoria_producto"]),
        "nombre_competidor": np.full(n, params["nombre_competidor"]),
        "anio_ingreso":      np.full(n, params["anio_ingreso"]),
        "clase":             np.full(n, clase),
        "origen":            np.full(n, "SINTETICO", dtype=object),
    }
    synthetic_rows.append(pd.DataFrame(rows))

df_syn = pd.concat(synthetic_rows, ignore_index=True)

# ──────────────────────────────────────────────
# 5. UNIÓN Y GUARDADO
# ──────────────────────────────────────────────
df_real["origen"] = "REAL"
df_final = pd.concat([df_real[FEATURES + ["clase", "origen"]], df_syn], ignore_index=True)

output = df_final.to_dict(orient="records")
with open("fact_competencia_optimizado.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"✅ fact_competencia_optimizado.json guardado")
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
    labels = {0: "Clase 0 – Desventaja", 1: "Clase 1 – Paridad", 2: "Clase 2 – Ventaja"}

    for c in [0, 1, 2]:
        mask = y == c
        axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1],
                        c=colors[c], label=labels[c], alpha=0.4, s=8)
    axes[0].set_title("t-SNE por Clase — fact_competencia")
    axes[0].legend(markerscale=3)

    for og, col in [("REAL", "#2980b9"), ("SINTETICO", "#8e44ad")]:
        mask = origen == og
        axes[1].scatter(X_2d[mask, 0], X_2d[mask, 1],
                        c=col, label=og, alpha=0.35, s=6)
    axes[1].set_title("t-SNE Real vs Sintético")
    axes[1].legend(markerscale=3)

    plt.tight_layout()
    plt.savefig("tsne_competencia.png", dpi=150)
    print("📊 tsne_competencia.png guardado")
except Exception as e:
    print(f"⚠️  t-SNE no generado: {e}")
