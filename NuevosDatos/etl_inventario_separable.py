"""
ETL Sintético Separable — fact_inventario
Origen: fact_inventario.json (3000 registros reales)
Target: clase de riesgo de inventario → 0=Crítico  1=Normal  2=Saludable
Features reales: stock_actual, stock_minimo, dias_sin_movimiento,
                 valor_unitario, valor_total_inventario,
                 region_natural, tipo_sucursal, categoria_producto,
                 ratio_stock (stock_actual/stock_minimo)
"""

import json
import numpy as np
import pandas as pd

np.random.seed(42)

# ──────────────────────────────────────────────
# 1. CARGA Y APLANADO
# ──────────────────────────────────────────────
with open("fact_inventario.json") as f:
    raw = json.load(f)

records = []
for r in raw:
    stock_actual = r.get("stock_actual", 1)
    stock_minimo = r.get("stock_minimo", 1) or 1
    records.append({
        "stock_actual":            stock_actual,
        "stock_minimo":            stock_minimo,
        "ratio_stock":             stock_actual / stock_minimo,
        "dias_sin_movimiento":     r.get("dias_sin_movimiento", 0),
        "valor_unitario":          r.get("valor_unitario", 0),
        "valor_total_inventario":  r.get("valor_total_inventario", 0),
        "margen_ganancia":         r["producto"].get("margen_ganancia", 0),
        "region_natural":          r["sucursal"].get("region_natural", "SIERRA"),
        "tipo_sucursal":           r["sucursal"].get("tipo_sucursal", "MINORISTA"),
        "categoria_producto":      r["producto"].get("categoria", "GENERAL"),
        "origen_producto":         r["producto"].get("origen", "NACIONAL"),
    })

df_real = pd.DataFrame(records)

# ──────────────────────────────────────────────
# 2. SCORE DE RIESGO DE INVENTARIO
# ──────────────────────────────────────────────
score = pd.Series(np.zeros(len(df_real)))

# Ratio alto = saludable; bajo = crítico
score += df_real["ratio_stock"].clip(0, 5) * 2.0
# Muchos días sin movimiento = riesgo alto (resta)
score -= (df_real["dias_sin_movimiento"] / df_real["dias_sin_movimiento"].max()) * 3.0
# Margen alto = producto valioso a mantener
score += df_real["margen_ganancia"] * 2.0

region_map = {"COSTA": 1.0, "SIERRA": 0.5, "ORIENTE": -1.0, "INSULAR": -1.5}
score += df_real["region_natural"].map(region_map).fillna(0)

tipo_map = {"PRINCIPAL": 1.0, "MINORISTA": 0.5, "EXPRESS": -1.0}
score += df_real["tipo_sucursal"].map(tipo_map).fillna(0)

origen_map = {"NACIONAL": 0.5, "IMPORTADO": -0.5}
score += df_real["origen_producto"].map(origen_map).fillna(0)

p33 = np.percentile(score, 33)
p66 = np.percentile(score, 66)
df_real["clase"] = 1
df_real.loc[score < p33, "clase"] = 0
df_real.loc[score >= p66, "clase"] = 2

# ──────────────────────────────────────────────
# 3. ENCODING
# ──────────────────────────────────────────────
ENCODINGS = {
    "region_natural":   {"COSTA": 3, "SIERRA": 2, "ORIENTE": 1, "INSULAR": 0},
    "tipo_sucursal":    {"PRINCIPAL": 2, "MINORISTA": 1, "EXPRESS": 0},
    "origen_producto":  {"NACIONAL": 1, "IMPORTADO": 0},
}
for col, mapping in ENCODINGS.items():
    df_real[col] = df_real[col].map(mapping).fillna(1).astype(int)
df_real["categoria_producto"] = pd.Categorical(df_real["categoria_producto"]).codes

FEATURES = ["stock_actual", "stock_minimo", "ratio_stock", "dias_sin_movimiento",
            "valor_unitario", "valor_total_inventario", "margen_ganancia",
            "region_natural", "tipo_sucursal", "categoria_producto", "origen_producto"]

# ──────────────────────────────────────────────
# 4. SINTÉTICOS CON GAPS AGRESIVOS
# ──────────────────────────────────────────────
CLASS_PARAMS = {
    # Clase 0: inventario crítico — bajo stock, mucho tiempo parado
    0: {
        "stock_actual":           (1, 30),
        "stock_minimo":           (40, 80),
        "ratio_stock":            (0.05, 0.45),
        "dias_sin_movimiento":    (90, 180),
        "valor_unitario":         (60, 150),
        "valor_total_inventario": (200, 2000),
        "margen_ganancia":        (0.02, 0.12),
        "region_natural":         0,
        "tipo_sucursal":          0,
        "categoria_producto":     0,
        "origen_producto":        0,
    },
    # Clase 1: inventario normal
    1: {
        "stock_actual":           (50, 150),
        "stock_minimo":           (30, 60),
        "ratio_stock":            (1.2, 2.5),
        "dias_sin_movimiento":    (20, 50),
        "valor_unitario":         (20, 50),
        "valor_total_inventario": (3000, 8000),
        "margen_ganancia":        (0.25, 0.40),
        "region_natural":         2,
        "tipo_sucursal":          1,
        "categoria_producto":     1,
        "origen_producto":        1,
    },
    # Clase 2: inventario saludable — alto stock relativo, movimiento frecuente
    2: {
        "stock_actual":           (250, 600),
        "stock_minimo":           (10, 30),
        "ratio_stock":            (8.0, 20.0),
        "dias_sin_movimiento":    (0, 8),
        "valor_unitario":         (5, 18),
        "valor_total_inventario": (12000, 30000),
        "margen_ganancia":        (0.50, 0.75),
        "region_natural":         3,
        "tipo_sucursal":          2,
        "categoria_producto":     2,
        "origen_producto":        1,
    },
}

N_SYN_PER_CLASS = 1500
synthetic_rows = []

for clase, params in CLASS_PARAMS.items():
    n = N_SYN_PER_CLASS
    std = 0.08

    def gen(key):
        lo, hi = params[key]
        return np.clip(np.random.normal(np.mean([lo, hi]), (hi - lo) * std, n), lo, hi)

    rows = {
        "stock_actual":           gen("stock_actual").astype(int),
        "stock_minimo":           gen("stock_minimo").astype(int),
        "ratio_stock":            gen("ratio_stock"),
        "dias_sin_movimiento":    gen("dias_sin_movimiento").astype(int),
        "valor_unitario":         gen("valor_unitario"),
        "valor_total_inventario": gen("valor_total_inventario"),
        "margen_ganancia":        gen("margen_ganancia"),
        "region_natural":         np.full(n, params["region_natural"]),
        "tipo_sucursal":          np.full(n, params["tipo_sucursal"]),
        "categoria_producto":     np.full(n, params["categoria_producto"]),
        "origen_producto":        np.full(n, params["origen_producto"]),
        "clase":                  np.full(n, clase),
        "origen":                 np.full(n, "SINTETICO", dtype=object),
    }
    synthetic_rows.append(pd.DataFrame(rows))

df_syn = pd.concat(synthetic_rows, ignore_index=True)

# ──────────────────────────────────────────────
# 5. UNIÓN Y GUARDADO
# ──────────────────────────────────────────────
df_real["origen"] = "REAL"
df_final = pd.concat([df_real[FEATURES + ["clase", "origen"]], df_syn], ignore_index=True)

output = df_final.to_dict(orient="records")
with open("fact_inventario_optimizado.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"✅ fact_inventario_optimizado.json guardado")
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
    labels = {0: "Clase 0 – Crítico", 1: "Clase 1 – Normal", 2: "Clase 2 – Saludable"}

    for c in [0, 1, 2]:
        mask = y == c
        axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1],
                        c=colors[c], label=labels[c], alpha=0.4, s=8)
    axes[0].set_title("t-SNE por Clase — fact_inventario")
    axes[0].legend(markerscale=3)

    for og, col in [("REAL", "#2980b9"), ("SINTETICO", "#8e44ad")]:
        mask = origen == og
        axes[1].scatter(X_2d[mask, 0], X_2d[mask, 1],
                        c=col, label=og, alpha=0.35, s=6)
    axes[1].set_title("t-SNE Real vs Sintético")
    axes[1].legend(markerscale=3)

    plt.tight_layout()
    plt.savefig("tsne_inventario.png", dpi=150)
    print("📊 tsne_inventario.png guardado")
except Exception as e:
    print(f"⚠️  t-SNE no generado: {e}")
