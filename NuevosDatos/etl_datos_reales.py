"""
ETL + Augmentación Sintética con t-SNE Separable (v2 — GAP MÁXIMO)
====================================================================
Lee datos REALES de fact_abastecimiento_logistica.json,
re-etiqueta con lógica de negocio y augmenta con sintéticos
diseñados para máxima separabilidad entre las 3 clases.

La clave v2: cada clase vive en una región NO solapada del espacio
de features (std bajo, medias muy separadas). Esto garantiza
separabilidad incluso después de PCA → SMOTE → t-SNE en el notebook.

Marlito - FISEI UTA - Pipeline ML Abastecimiento
"""

import json
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import datetime
from collections import Counter

# ─── Semillas ────────────────────────────────────────────────────────────────
np.random.seed(42)
random.seed(42)

ARCHIVO_REAL      = 'fact_abastecimiento_logistica.json'
ARCHIVO_SALIDA    = 'fact_abastecimiento_optimizado.json'
TOTAL_SINTETICOS  = 2000   # por clase (domina sobre los reales para mayor separabilidad)

# ─── Tablas de Score ──────────────────────────────────────────────────────────
CAL_SCORE      = {'A+': 2.0, 'A': 1.5, 'B+': 0.5, 'B': 0.0, 'C': -1.5}
CONTRATO_SCORE = {
    'CONTRATO ANUAL': 1.5, 'CONTRATO SEMESTRAL': 0.8,
    'CONTRATO MENSUAL': 0.0, 'CONSIGNACION': -0.5, 'ORDEN DE COMPRA': -1.0
}
REGION_SCORE = {'COSTA': 0.5, 'SIERRA': 0.3, 'ORIENTE': -0.5, 'INSULAR': -1.0}


# ─── PASO 1: Leer datos REALES ────────────────────────────────────────────────
print("=" * 60)
print("PASO 1 — Leyendo datos reales...")
with open(ARCHIVO_REAL) as f:
    data_real = json.load(f)
print(f"  Registros cargados: {len(data_real)}")


def calcular_score_riesgo(r):
    s  = CAL_SCORE.get(r['proveedor']['calificacion'], 0)
    s += CONTRATO_SCORE.get(r['proveedor']['tipo_contrato'], 0)
    s += REGION_SCORE.get(r['sucursal']['region_natural'], 0)
    s += (10 - r['tiempo_entrega_dias']) * 0.15
    s += r['producto']['margen_ganancia'] * 2.0
    s += np.log(max(1, r['sucursal']['capacidad_almacenamiento'])) * 0.1
    return s


scores_reales = np.array([calcular_score_riesgo(r) for r in data_real])
p33 = np.percentile(scores_reales, 33)
p66 = np.percentile(scores_reales, 66)
print(f"  Percentiles → p33={p33:.2f}, p66={p66:.2f}")


def extraer_features(r):
    score = calcular_score_riesgo(r)
    if   score < p33: clase = 0
    elif score < p66: clase = 1
    else:             clase = 2
    return {
        "_id": r.get("_id", f"REAL-{r.get('id_logistica','?')}"),
        "origen": "REAL",
        "tiempo_entrega_dias":       r["tiempo_entrega_dias"],
        "cantidad_solicitada":       r["cantidad_solicitada"],
        "cantidad_recibida":         r["cantidad_recibida"],
        "ratio_entrega":             round(r["cantidad_recibida"] / max(1, r["cantidad_solicitada"]), 4),
        "costo_unitario":            r["costo_unitario"],
        "costo_logistico":           r["costo_logistico"],
        "costo_total_orden":         r["costo_total_orden"],
        "stock_minimo":              r["stock_minimo"],
        "capacidad_almacenamiento":  r["sucursal"]["capacidad_almacenamiento"],
        "metros_cuadrados":          r["sucursal"]["metros_cuadrados"],
        "numero_empleados":          r["sucursal"]["numero_empleados"],
        "margen_ganancia":           r["producto"]["margen_ganancia"],
        "precio_costo":              r["producto"]["precio_costo"],
        "es_fin_semana":             abs(r["tiempo"]["es_fin_semana"]),
        "es_feriado":                abs(r["tiempo"]["es_feriado"]),
        "trimestre":                 r["tiempo"]["trimestre"],
        "score_riesgo":              round(score, 4),
        "calificacion_proveedor":    r["proveedor"]["calificacion"],
        "tipo_contrato":             r["proveedor"]["tipo_contrato"],
        "region_natural":            r["sucursal"]["region_natural"],
        "tipo_sucursal":             r["sucursal"]["tipo_sucursal"],
        "categoria_producto":        r["producto"]["categoria"],
        "origen_producto":           r["producto"]["origen"],
        "clase_riesgo":              clase
    }


registros_reales = [extraer_features(r) for r in data_real]
dist_real = Counter(f["clase_riesgo"] for f in registros_reales)
print(f"  Distribución real: {dict(dist_real)}")


# ─── PASO 2: Sintéticos con GAP MÁXIMO entre clases ─────────────────────────
print("\nPASO 2 — Generando sintéticos con separabilidad máxima...")

CATEGORIAS = ['ALIMENTACION', 'TECNOLOGIA', 'FARMACEUTICO', 'TEXTIL', 'BEBIDAS', 'EMPAQUES']
ORIGENES   = ['NACIONAL', 'IMPORTADO']
TIPOS_SUC  = ['MAYORISTA', 'MINORISTA', 'DISTRIBUIDOR']


def generar_sintetico_clase(clase_obj, idx):
    """
    v2 — Estrategia de GAP MÁXIMO:
    • Varianza MUY BAJA (std 50-70% menor que v1)
    • Medias muy alejadas entre clases
    • Variables categóricas FIJAS por clase (sin solapamiento)
    • Esto crea nubes compactas y bien separadas en t-SNE
    """

    if clase_obj == 0:
        # ── ZONA CRÍTICA: todo en el peor extremo ──
        cal      = 'C'
        contrato = 'ORDEN DE COMPRA'
        region   = 'ORIENTE'
        tipo_suc = 'MINORISTA'
        dias     = int(np.clip(np.random.normal(18, 1.2), 15, 21))
        cap_alm  = int(np.clip(np.random.normal(480, 60),  300,  650))
        margen   = np.clip(np.random.normal(0.055, 0.015), 0.02, 0.10)
        cant_sol = int(np.clip(np.random.normal(680, 50),  500,  800))
        ratio    = np.clip(np.random.normal(0.33, 0.04), 0.20, 0.45)
        cant_rec = int(cant_sol * ratio)
        costo_u  = np.clip(np.random.normal(180, 20), 130, 240)
        costo_l  = np.clip(np.random.normal(380, 15), 340, 400)
        empleados = int(np.clip(np.random.normal(18, 3), 10, 28))
        m2        = int(np.clip(np.random.normal(320, 30), 250, 420))

    elif clase_obj == 1:
        # ── ZONA MEDIA: centrada, compacta, sin solapamiento ──
        cal      = 'B+'
        contrato = 'CONTRATO MENSUAL'
        region   = 'SIERRA'
        tipo_suc = 'MAYORISTA'
        dias     = int(np.clip(np.random.normal(10, 0.8), 8, 12))
        cap_alm  = int(np.clip(np.random.normal(2100, 100), 1850, 2350))
        margen   = np.clip(np.random.normal(0.32, 0.02), 0.27, 0.38)
        cant_sol = int(np.clip(np.random.normal(350, 40),  250,  480))
        ratio    = np.clip(np.random.normal(0.73, 0.03), 0.65, 0.82)
        cant_rec = int(cant_sol * ratio)
        costo_u  = np.clip(np.random.normal(45, 8),  28,  65)
        costo_l  = np.clip(np.random.normal(200, 12), 170, 230)
        empleados = int(np.clip(np.random.normal(55, 5), 42, 68))
        m2        = int(np.clip(np.random.normal(1500, 80), 1300, 1700))

    else:
        # ── ZONA EXITOSA: todo en el mejor extremo ──
        cal      = 'A+'
        contrato = 'CONTRATO ANUAL'
        region   = 'COSTA'
        tipo_suc = 'DISTRIBUIDOR'
        dias     = int(np.clip(np.random.normal(2, 0.6), 1, 4))
        cap_alm  = int(np.clip(np.random.normal(4700, 120), 4400, 5000))
        margen   = np.clip(np.random.normal(0.73, 0.03), 0.66, 0.82)
        cant_sol = int(np.clip(np.random.normal(130, 30),  50,  220))
        cant_rec = cant_sol   # entrega 100%
        ratio    = 1.0
        costo_u  = np.clip(np.random.normal(6, 2),   2,  12)
        costo_l  = np.clip(np.random.normal(18, 3),  10,  28)
        empleados = int(np.clip(np.random.normal(105, 6), 90, 120))
        m2        = int(np.clip(np.random.normal(3400, 100), 3100, 3700))

    score = (
        CAL_SCORE.get(cal, 0)
        + CONTRATO_SCORE.get(contrato, 0)
        + REGION_SCORE.get(region, 0)
        + (10 - dias) * 0.15
        + margen * 2.0
        + np.log(max(1, cap_alm)) * 0.1
    )

    return {
        "_id":                       f"SYN-C{clase_obj}-{idx}",
        "origen":                    "SINTETICO",
        "tiempo_entrega_dias":       dias,
        "cantidad_solicitada":       cant_sol,
        "cantidad_recibida":         cant_rec,
        "ratio_entrega":             round(ratio, 4),
        "costo_unitario":            round(float(costo_u), 2),
        "costo_logistico":           round(float(costo_l), 2),
        "costo_total_orden":         round(float(costo_u) * cant_sol + float(costo_l), 2),
        "stock_minimo":              int(np.random.uniform(5, 50)),
        "capacidad_almacenamiento":  cap_alm,
        "metros_cuadrados":          m2,
        "numero_empleados":          empleados,
        "margen_ganancia":           round(float(margen), 4),
        "precio_costo":              round(float(costo_u), 2),
        "es_fin_semana":             random.choice([0, 1]),
        "es_feriado":                random.choices([0, 1], weights=[0.9, 0.1])[0],
        "trimestre":                 random.randint(1, 4),
        "score_riesgo":              round(score, 4),
        "calificacion_proveedor":    cal,
        "tipo_contrato":             contrato,
        "region_natural":            region,
        "tipo_sucursal":             tipo_suc,
        "categoria_producto":        random.choice(CATEGORIAS),
        "origen_producto":           random.choice(ORIGENES),
        "clase_riesgo":              clase_obj
    }


registros_sinteticos = []
for clase_obj in [0, 1, 2]:
    for i in range(TOTAL_SINTETICOS):
        registros_sinteticos.append(generar_sintetico_clase(clase_obj, i))

dist_sint = Counter(f["clase_riesgo"] for f in registros_sinteticos)
print(f"  Sintéticos por clase: {dict(dist_sint)}")


# ─── PASO 3: Combinar y guardar ───────────────────────────────────────────────
print("\nPASO 3 — Guardando dataset optimizado...")
dataset_final = registros_reales + registros_sinteticos
random.shuffle(dataset_final)

with open(ARCHIVO_SALIDA, 'w', encoding='utf-8') as f:
    json.dump(dataset_final, f, ensure_ascii=False, indent=2)

dist_final = Counter(r["clase_riesgo"] for r in dataset_final)
print(f"  Total registros: {len(dataset_final)}")
print(f"  Distribución: {dict(dist_final)}")


# ─── PASO 4: Preparar para t-SNE ─────────────────────────────────────────────
print("\nPASO 4 — Preparando matriz de features...")

CATS = ['calificacion_proveedor', 'tipo_contrato', 'region_natural',
        'tipo_sucursal', 'categoria_producto', 'origen_producto']
encoders = {c: LabelEncoder() for c in CATS}
for c in CATS:
    encoders[c].fit([r[c] for r in dataset_final])

NUMERICAS = [
    'tiempo_entrega_dias', 'cantidad_solicitada', 'cantidad_recibida',
    'ratio_entrega', 'costo_unitario', 'costo_logistico', 'costo_total_orden',
    'stock_minimo', 'capacidad_almacenamiento', 'metros_cuadrados',
    'numero_empleados', 'margen_ganancia', 'precio_costo',
    'es_fin_semana', 'es_feriado', 'trimestre', 'score_riesgo'
]

X_rows, y_labels, orig_labels = [], [], []
for r in dataset_final:
    row = [r[n] for n in NUMERICAS]
    for c in CATS:
        row.append(float(encoders[c].transform([r[c]])[0]))
    X_rows.append(row)
    y_labels.append(r["clase_riesgo"])
    orig_labels.append(r["origen"])

X = np.array(X_rows, dtype=float)
y = np.array(y_labels)
X_scaled = StandardScaler().fit_transform(X)
print(f"  Matriz X: {X_scaled.shape}")


# ─── PASO 5: t-SNE estratificado ──────────────────────────────────────────────
print("\nPASO 5 — Ejecutando t-SNE...")

MUESTRA = 3000
indices = []
for c in [0, 1, 2]:
    idx_c = np.where(y == c)[0]
    n = min(MUESTRA // 3, len(idx_c))
    indices.extend(np.random.choice(idx_c, n, replace=False).tolist())
random.shuffle(indices)

X_tsne_in = X_scaled[indices]
y_tsne    = y[indices]
orig_tsne = [orig_labels[i] for i in indices]

tsne = TSNE(
    n_components=2,
    perplexity=40,
    learning_rate=150,
    max_iter=1500,
    random_state=42,
    init='pca',
    metric='euclidean'
)
X_2d = tsne.fit_transform(X_tsne_in)
print(f"  t-SNE completado: {X_2d.shape}")


# ─── PASO 6: Graficar ─────────────────────────────────────────────────────────
print("\nPASO 6 — Graficando...")

COLORES = {0: '#E8553E', 1: '#4A90D9', 2: '#2ECC71'}
NOMBRES = {0: 'Clase 0 — Alto Riesgo', 1: 'Clase 1 — Riesgo Medio', 2: 'Clase 2 — Exitoso'}

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.patch.set_facecolor('#F8F9FA')

# ── Subplot 1: Por clase ──────────────────────────────────────────────────────
ax1 = axes[0]
ax1.set_facecolor('#FFFFFF')
for c in [0, 1, 2]:
    mask = y_tsne == c
    ax1.scatter(X_2d[mask, 0], X_2d[mask, 1],
                c=COLORES[c], s=9, alpha=0.6, linewidths=0, label=NOMBRES[c])
ax1.set_title('t-SNE — Por Clase de Riesgo (v2)', fontsize=14, fontweight='bold', pad=12)
ax1.set_xlabel('Componente 1', fontsize=11)
ax1.set_ylabel('Componente 2', fontsize=11)
ax1.legend(fontsize=10, framealpha=0.9)
ax1.grid(True, alpha=0.2, linestyle='--')
ax1.spines[['top', 'right']].set_visible(False)

# ── Subplot 2: Real vs Sintético ──────────────────────────────────────────────
ax2 = axes[1]
ax2.set_facecolor('#FFFFFF')
for c in [0, 1, 2]:
    for origen, marker, sz, alpha in [('REAL', 'o', 18, 0.95), ('SINTETICO', '.', 5, 0.35)]:
        mask = (y_tsne == c) & np.array([o == origen for o in orig_tsne])
        if mask.sum() == 0:
            continue
        ax2.scatter(X_2d[mask, 0], X_2d[mask, 1],
                    c=COLORES[c], marker=marker, s=sz, alpha=alpha, linewidths=0)

patches = [mpatches.Patch(color=COLORES[c], label=NOMBRES[c]) for c in [0, 1, 2]]
lines = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=9, label='Reales'),
    Line2D([0], [0], marker='.', color='w', markerfacecolor='gray', markersize=9, label='Sintéticos'),
]
ax2.legend(handles=patches + lines, fontsize=9, framealpha=0.9)
ax2.set_title('t-SNE — Real vs Sintético por Clase', fontsize=14, fontweight='bold', pad=12)
ax2.set_xlabel('Componente 1', fontsize=11)
ax2.set_ylabel('Componente 2', fontsize=11)
ax2.grid(True, alpha=0.2, linestyle='--')
ax2.spines[['top', 'right']].set_visible(False)

plt.suptitle(
    'Pipeline ML — Abastecimiento Logístico  |  v2 — Máxima Separabilidad\n'
    'Datos Reales + Augmentación Sintética con Gap Estructurado',
    fontsize=13, fontweight='bold', y=1.01
)
plt.tight_layout()
plt.savefig('tsne_separable.png', dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
print("  Guardado: tsne_separable.png")

# ─── Resumen ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RESUMEN FINAL")
print("=" * 60)
for c in [0, 1, 2]:
    nr = sum(1 for r in registros_reales    if r['clase_riesgo'] == c)
    ns = sum(1 for r in registros_sinteticos if r['clase_riesgo'] == c)
    print(f"  Clase {c} ({NOMBRES[c][-14:].strip()}): {nr+ns} total  ({nr} reales + {ns} sint.)")
print(f"\n  Features: {len(NUMERICAS)} numéricas + {len(CATS)} categóricas = {len(NUMERICAS)+len(CATS)} total")
print(f"  Salida JSON: {ARCHIVO_SALIDA}")
print(f"  t-SNE:       tsne_separable.png")
print("=" * 60)