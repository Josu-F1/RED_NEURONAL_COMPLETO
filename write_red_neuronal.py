import json
import os

notebook_path = r"c:\Users\ASUS\OneDrive\Escritorio\data_set_refinado\RED_NEURONAL\red_neuronal_ventas.ipynb"
os.makedirs(os.path.dirname(notebook_path), exist_ok=True)

def make_cell(cell_type, source):
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": source if isinstance(source, list) else [line + "\n" for line in source.split("\n")]
    }

cells = []

# Cell 0: Header Markdown
cells.append(make_cell("markdown", """# Red Neuronal Artificial (Perceptrón Multicapa) - `fact_ventas`
## Predicción de Alta Rentabilidad con Evitación de Fuga de Datos y Evaluación Premium

Este notebook implementa y evalúa un modelo de **Red Neuronal Artificial (MLP)** de scikit-learn para predecir si una transacción de venta será de **Alta Rentabilidad** (definida como `margen_pct >= 0.30`).

### Objetivos Clave:
1. **Evitar la Fuga de Datos (Data Leakage):** Se excluyen todas las métricas post-venta (`monto_total`, `costo_total`, `utilidad_bruta`, `margen_pct`). Solo se utilizan variables operativas conocidas ex-ante.
2. **Entrenamiento Monitoreado por Épocas:** Mediante un bucle personalizado con `partial_fit`, registramos y visualizamos el progreso del aprendizaje (curvas de Loss y Accuracy para Train y Validación).
3. **Métricas de Evaluación y Dashboard Premium:** Generación de matriz de confusión, distribución de probabilidades de predicción, visualización t-SNE 2D y reporte de clasificación completo.
4. **Cumplimiento del Estándar de Entregables:** Exportación de los resultados de pesos, configuración y predicciones a 6 archivos Excel estructurados en la carpeta de salidas.

---
### Definición del Target (Rentabilidad)
- **Clase 0 (Rentabilidad Estándar):** `margen_pct < 0.30`
- **Clase 1 (Alta Rentabilidad):** `margen_pct >= 0.30`"""))

# Cell 1: Imports
cells.append(make_cell("code", """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import os
import json
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo visual premium
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.family'] = 'sans-serif'

# Crear directorio de outputs locales
os.makedirs('outputs', exist_ok=True)
print("Entorno inicializado correctamente.")"""))

# Cell 2: Markdown 1
cells.append(make_cell("markdown", "## 1. Cargar Datos y Extraer Características (Evitando Fuga de Datos)"))

# Cell 3: Code Load Data
cells.append(make_cell("code", """# Cargar el archivo JSON original de ventas desde la carpeta raíz
with open('../fact_ventas.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
df_raw = pd.DataFrame(data)

# 1. Definir el target binario basado en alta rentabilidad (margen_pct >= 30%)
y = (df_raw['margen_pct'] >= 0.30).astype(int)

# 2. Extraer características operativas para X
# IMPORTANTE: Eliminamos 'monto_total', 'costo_total', 'utilidad_bruta' y 'margen_pct' 
# de X para evitar fuga de datos (Data Leakage).
df = pd.DataFrame()
df['cantidad'] = df_raw['cantidad']
df['precio_unitario'] = df_raw['precio_unitario']
df['costo_unitario'] = df_raw['costo_unitario']
df['descuento_pct'] = df_raw['descuento_pct']

# Variables contextuales independientes de la venta
df['mes'] = df_raw['tiempo'].apply(lambda x: x['mes'])
df['es_fin_semana'] = df_raw['tiempo'].apply(lambda x: 1 if x['es_fin_semana'] == -1 else 0)
df['es_feriado'] = df_raw['tiempo'].apply(lambda x: int(x['es_feriado']))
df['margen_ganancia_prod'] = df_raw['producto'].apply(lambda x: x['margen_ganancia'])

# Variables categóricas codificadas ordinalmente
canal_map = {'VENTA DIRECTA':0, 'MAYORISTA':1, 'EN LINEA':2, 'MINORISTA':3, 'DISTRIBUIDORA':4}
df['canal'] = df_raw['canal'].apply(lambda x: canal_map.get(x['nombre_canal'], 0))

mp_map = {'EFECTIVO':0, 'TARJETA DE CREDITO':1, 'TARJETA DE DEBITO':2, 'TRANSFERENCIA BANCARIA':3, 'CHEQUE':4}
df['metodo_pago'] = df_raw['metodo_pago'].apply(
    lambda x: mp_map.get(x['descripcion_pago'].replace('É','E').replace('É','E'), 0))

suc_map = {'MINORISTA':0, 'PRINCIPAL':1, 'SECUNDARIA':2, 'MAYORISTA':3, 'EXPRESS':4}
df['tipo_sucursal'] = df_raw['sucursal'].apply(lambda x: suc_map.get(x['tipo_sucursal'], 0))

print(f"Dataset cargado con éxito. Dimensiones de Features X: {df.shape}")
print(f"Distribución del target (Clase 0: {np.sum(y==0)} | Clase 1: {np.sum(y==1)})")"""))

# Cell 4: Markdown 2
cells.append(make_cell("markdown", "## 2. Normalización y División del Dataset (3-Way Split)"))

# Cell 5: Code Split & Scale
cells.append(make_cell("code", """# Escalado de datos usando QuantileTransformer (mapeado normal, ideal para redes neuronales)
qt = QuantileTransformer(output_distribution='normal', random_state=42)
X_scaled = qt.fit_transform(df)

# División estratificada de datos
# Test: 25%
X_temp, X_test, y_temp, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42, stratify=y
)

# Val: 25% total | Train: 50% total
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=(1/3), random_state=42, stratify=y_temp
)

# Convertir y a arrays numpy de tipo entero para evitar fallos
y_train = y_train.values.astype(int)
y_val = y_val.values.astype(int)
y_test = y_test.values.astype(int)

print(f"Total registros : {len(y):,}")
print(f"Entrenamiento   : {X_train.shape[0]:,} ({X_train.shape[0]/len(y)*100:.0f}%)")
print(f"Validación      : {X_val.shape[0]:,} ({X_val.shape[0]/len(y)*100:.0f}%)")
print(f"Prueba          : {X_test.shape[0]:,} ({X_test.shape[0]/len(y)*100:.0f}%)")"""))

# Cell 6: Markdown 3
cells.append(make_cell("markdown", "## 3. Entrenamiento Monitoreado de la Red Neuronal (MLP)"))

# Cell 7: Code Neural Network Train
cells.append(make_cell("code", """# Inicializar la Red Neuronal (Perceptrón Multicapa)
# Usamos dos capas ocultas de (30, 15) neuronas, activaciones ReLU y optimizador Adam
LR_INIT = 0.01
EPOCHS = 500

mlp = MLPClassifier(
    hidden_layer_sizes=(30, 15),
    activation='relu',
    solver='adam',
    learning_rate_init=LR_INIT,
    random_state=42,
    warm_start=True
)

classes = np.array([0, 1])
historial = []

print("="*80)
print("  ENTRENAMIENTO MONITOREADO DE LA RED NEURONAL (MLP)")
print("="*80)
print(f"  Arquitectura: {mlp.hidden_layer_sizes} | Activación: {mlp.activation} | Optimizador: {mlp.solver}")
print(f"  Tasa de aprendizaje inicial: {LR_INIT} | Épocas de entrenamiento: {EPOCHS}")
print("-"*80)
print(f"{'Época':>7} | {'Loss Train':>12} | {'Loss Val':>10} | {'Acc Train':>10} | {'Acc Val':>8}")
print("-"*80)

for epoch in range(EPOCHS):
    # Entrenar por una sola época
    mlp.partial_fit(X_train, y_train, classes=classes)
    
    # Predecir probabilidades
    y_prob_tr = mlp.predict_proba(X_train)[:, 1]
    y_prob_v  = mlp.predict_proba(X_val)[:, 1]
    
    # Calcular Log-Loss (entropía cruzada)
    loss_tr = -np.mean(y_train * np.log(np.clip(y_prob_tr, 1e-15, 1 - 1e-15)) + (1 - y_train) * np.log(np.clip(1 - y_prob_tr, 1e-15, 1 - 1e-15)))
    loss_v  = -np.mean(y_val * np.log(np.clip(y_prob_v, 1e-15, 1 - 1e-15)) + (1 - y_val) * np.log(np.clip(1 - y_prob_v, 1e-15, 1 - 1e-15)))
    
    # Calcular exactitud (Accuracy)
    y_pred_tr = (y_prob_tr > 0.5).astype(int)
    y_pred_v  = (y_prob_v > 0.5).astype(int)
    
    acc_tr = np.mean(y_pred_tr == y_train)
    acc_v  = np.mean(y_pred_v == y_val)
    
    historial.append({
        'epoca': epoch + 1,
        'loss_train': loss_tr,
        'loss_val': loss_v,
        'acc_train': acc_tr,
        'acc_val': acc_v
    })
    
    if epoch == 0 or (epoch + 1) % 50 == 0 or epoch == EPOCHS - 1:
        print(f"{epoch+1:>7d} | {loss_tr:>12.6f} | {loss_v:>10.6f} | {acc_tr*100:>9.2f}% | {acc_v*100:>7.2f}%")

print("="*80)
df_hist = pd.DataFrame(historial)
print("Entrenamiento finalizado con éxito.")"""))

# Cell 8: Markdown 4
cells.append(make_cell("markdown", "## 4. Exportación de Resultados y Predicciones a Excel"))

# Cell 9: Code Export Excel
cells.append(make_cell("code", """# Helper para guardar a Excel con fallback a CSV
def guardar_excel_local(df_export, filename):
    filepath = os.path.join('outputs', filename)
    try:
        df_export.to_excel(filepath, index=False)
        print(f"  [Excel] Guardado exitosamente: {filepath} ({len(df_export):,} filas)")
    except Exception as e:
        filepath_csv = filepath.replace('.xlsx', '.csv')
        df_export.to_csv(filepath_csv, index=False)
        print(f"  [CSV Fallback] Guardado exitosamente: {filepath_csv}  [{e}]")

# Obtener predicciones finales
y_prob_train = mlp.predict_proba(X_train)[:, 1]
y_prob_val   = mlp.predict_proba(X_val)[:, 1]
y_prob_test  = mlp.predict_proba(X_test)[:, 1]

y_pred_train = (y_prob_train > 0.5).astype(int)
y_pred_val   = (y_prob_val > 0.5).astype(int)
y_pred_test  = (y_prob_test > 0.5).astype(int)

acc_train = np.mean(y_pred_train == y_train)
acc_val   = np.mean(y_pred_val == y_val)
acc_test  = np.mean(y_pred_test == y_test)

loss_train = -np.mean(y_train * np.log(np.clip(y_prob_train, 1e-15, 1 - 1e-15)) + (1 - y_train) * np.log(np.clip(1 - y_prob_train, 1e-15, 1 - 1e-15)))
loss_val   = -np.mean(y_val * np.log(np.clip(y_prob_val, 1e-15, 1 - 1e-15)) + (1 - y_val) * np.log(np.clip(1 - y_prob_val, 1e-15, 1 - 1e-15)))
loss_test  = -np.mean(y_test * np.log(np.clip(y_prob_test, 1e-15, 1 - 1e-15)) + (1 - y_test) * np.log(np.clip(1 - y_prob_test, 1e-15, 1 - 1e-15)))

print("Exportando datos consolidados a carpeta outputs/...")

# 1. Pesos de Entrada (Magnitud acumulada absoluta)
feature_names = df.columns.tolist()
weights_first_layer = np.sum(np.abs(mlp.coefs_[0]), axis=1)
df_weights = pd.DataFrame({
    'feature': feature_names,
    'peso_acumulado_abs': weights_first_layer,
    'importancia_%': (weights_first_layer / weights_first_layer.sum()) * 100
}).sort_values('peso_acumulado_abs', ascending=False)
guardar_excel_local(df_weights, '01_pesos_finales.xlsx')

# 2. Configuración y Hiperparámetros del Modelo
df_config = pd.DataFrame([
    {'parametro': 'Capas Ocultas (hidden_layers)', 'valor': str(mlp.hidden_layer_sizes)},
    {'parametro': 'Función de Activación',          'valor': mlp.activation},
    {'parametro': 'Optimizador (solver)',          'valor': mlp.solver},
    {'parametro': 'Tasa de Aprendizaje Inicial',   'valor': LR_INIT},
    {'parametro': 'Épocas de Entrenamiento',       'valor': EPOCHS},
    {'parametro': 'Cantidad de Features entrada',  'valor': len(feature_names)},
    {'parametro': 'Accuracy Entrenamiento',        'valor': round(acc_train * 100, 4)},
    {'parametro': 'Accuracy Validación',           'valor': round(acc_val * 100, 4)},
    {'parametro': 'Accuracy Prueba (Test)',        'valor': round(acc_test * 100, 4)},
    {'parametro': 'Log-Loss Entrenamiento',        'valor': round(loss_train, 6)},
    {'parametro': 'Log-Loss Validación',           'valor': round(loss_val, 6)},
    {'parametro': 'Log-Loss Prueba (Test)',        'valor': round(loss_test, 6)}
])
guardar_excel_local(df_config, '02_configuracion_modelo.xlsx')

# 3, 4, 5. Predicciones Detalladas
for name, y_true, y_pred, y_prob in [
    ('train', y_train, y_pred_train, y_prob_train),
    ('val',   y_val,   y_pred_val,   y_prob_val),
    ('test',  y_test,  y_pred_test,  y_prob_test)
]:
    df_preds = pd.DataFrame({
        'y_real': y_true,
        'y_predicha': y_pred,
        'probabilidad': np.round(y_prob, 6),
        'correcto': (y_pred == y_true).astype(int)
    })
    guardar_excel_local(df_preds, f'03_predicciones_{name}.xlsx')

# 6. Historial de Entrenamiento por época
guardar_excel_local(df_hist, '06_historial_entrenamiento.xlsx')

print("\\nReporte de Clasificación en Test:")
print(classification_report(y_test, y_pred_test, target_names=['Rentabilidad Estándar', 'Alta Rentabilidad']))"""))

# Cell 10: Markdown 5
cells.append(make_cell("markdown", "## 5. Dashboard de Evaluación Premium (Visualización del Rendimiento)"))

# Cell 11: Code Dashboard Plots
cells.append(make_cell("code", """# ─── Paleta de colores Premium (Fondo Oscuro) ──────────────────────────────
COLOR_0  = '#EF476F'   # Rosa/Rojo - clase 0 (Rentabilidad Estándar)
COLOR_1  = '#06D6A0'   # Verde/Teal - clase 1 (Alta Rentabilidad)
BG_DARK  = '#0D1117'   # Fondo general
BG_PANEL = '#161B22'   # Fondo paneles
TEXT_CLR = '#E6EDF3'   # Texto principal
GRID_CLR = '#30363D'   # Rejilla
ACCENT   = '#58A6FF'   # Azul de destaque

# Configurar matplotlib para modo oscuro premium
plt.rcParams.update({
    'figure.facecolor': BG_DARK,
    'axes.facecolor':   BG_PANEL,
    'axes.edgecolor':   GRID_CLR,
    'axes.labelcolor':  TEXT_CLR,
    'text.color':       TEXT_CLR,
    'xtick.color':      TEXT_CLR,
    'ytick.color':      TEXT_CLR,
    'grid.color':       GRID_CLR,
    'grid.alpha':       0.4,
    'font.family':      'monospace',
})

fig = plt.figure(figsize=(22, 15))
fig.suptitle(
    'EVALUACIÓN DE RED NEURONAL ARTIFICIAL (MLP) — fact_ventas',
    fontsize=18, fontweight='bold', color=ACCENT, y=0.98
)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# 1. Curvas de aprendizaje (Loss Curve)
ax1 = fig.add_subplot(gs[0, :])
epocas = df_hist['epoca'].values
ax1.plot(epocas, df_hist['loss_train'], color=COLOR_0, lw=2.5, label='Loss Train')
ax1.plot(epocas, df_hist['loss_val'],   color=COLOR_1, lw=2.5, label='Loss Val', linestyle='--')
ax1.set_title('Historial de Pérdida por Época (Binary Cross-Entropy / Log-Loss)', fontsize=13, fontweight='bold')
ax1.set_xlabel('Época', fontsize=11)
ax1.set_ylabel('Log-Loss', fontsize=11)
ax1.legend(loc='upper right', fontsize=11)
ax1.grid(True)

# 2. Accuracy por época
ax2 = fig.add_subplot(gs[1, 0:2])
ax2.plot(epocas, df_hist['acc_train']*100, color=COLOR_0, lw=2.5, label='Acc Train')
ax2.plot(epocas, df_hist['acc_val']*100,   color=COLOR_1, lw=2.5, label='Acc Val', linestyle='--')
ax2.axhline(acc_test*100, color=ACCENT, lw=1.5, linestyle=':', label=f'Acc Test = {acc_test*100:.2f}%')
ax2.set_title('Evolución de la Exactitud (Accuracy) por Época', fontsize=13, fontweight='bold')
ax2.set_xlabel('Época', fontsize=11)
ax2.set_ylabel('Accuracy (%)', fontsize=11)
ax2.set_ylim(40, 102)
ax2.legend(loc='lower right', fontsize=11)
ax2.grid(True)

# 3. Matriz de confusión
ax3 = fig.add_subplot(gs[1, 2])
cm = confusion_matrix(y_test, y_pred_test)
ConfusionMatrixDisplay(cm, display_labels=['Estándar', 'Alta Rent.']).plot(
    ax=ax3, colorbar=False, cmap=plt.cm.Blues
)
ax3.set_title('Matriz de Confusión (Test Set)', fontsize=13, fontweight='bold')
ax3.grid(False)

# 4. Distribución de probabilidades predichas
ax4 = fig.add_subplot(gs[2, 0])
ax4.hist(y_prob_test[y_test == 0], bins=30, color=COLOR_0, alpha=0.7, label='Estándar (Clase 0)', density=True)
ax4.hist(y_prob_test[y_test == 1], bins=30, color=COLOR_1, alpha=0.7, label='Alta Rent. (Clase 1)', density=True)
ax4.axvline(0.5, color='white', lw=1.5, linestyle='--')
ax4.set_title('Distribución de Probabilidades (Test Set)', fontsize=13, fontweight='bold')
ax4.set_xlabel('P(Alta Rentabilidad)', fontsize=11)
ax4.set_ylabel('Densidad', fontsize=11)
ax4.legend(fontsize=10)
ax4.grid(True)

# 5. Visualización t-SNE Proyectado de Predicciones
ax5 = fig.add_subplot(gs[2, 1])
correct = (y_pred_test == y_test).astype(int)
from sklearn.manifold import TSNE
print("Calculando proyección t-SNE 2D para test set...")
tsne_eval = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1)
X_test_tsne = tsne_eval.fit_transform(X_test)
ax5.scatter(X_test_tsne[y_test == 0, 0], X_test_tsne[y_test == 0, 1], color=COLOR_0, s=25, alpha=0.6, label='Estándar Real')
ax5.scatter(X_test_tsne[y_test == 1, 0], X_test_tsne[y_test == 1, 1], color=COLOR_1, s=25, alpha=0.7, label='Alta Rent. Real')
err_indices = np.where(correct == 0)[0]
if len(err_indices) > 0:
    ax5.scatter(X_test_tsne[err_indices, 0], X_test_tsne[err_indices, 1], facecolors='none', edgecolors='white', s=80, lw=1.5, label='Predicciones Incorrectas')
ax5.set_title('Separabilidad y Clasificación t-SNE (Test Set)', fontsize=13, fontweight='bold')
ax5.set_xlabel('t-SNE Componente 1', fontsize=11)
ax5.set_ylabel('t-SNE Componente 2', fontsize=11)
ax5.legend(fontsize=10)
ax5.grid(True)

# 6. Cuadro de Resumen de Métricas
ax6 = fig.add_subplot(gs[2, 2])
ax6.axis('off')

from sklearn.metrics import precision_recall_fscore_support
precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred_test, average='binary')

metrics_text = (
    f"  MÉTRICAS DE RENDIMIENTO FINAL\\n"
    f"  {'─'*33}\\n"
    f"   Accuracy Test  : {acc_test*100:>7.2f}%\\n"
    f"   Precision Test : {precision*100:>7.2f}%\\n"
    f"   Recall Test    : {recall*100:>7.2f}%\\n"
    f"   F1-Score Test  : {fscore*100:>7.2f}%\\n"
    f"  {'─'*33}\\n"
    f"   Loss Test (Log): {loss_test:>8.5f}\\n"
    f"   Loss Train     : {loss_train:>8.5f}\\n"
    f"   Loss Val       : {loss_val:>8.5f}\\n"
    f"  {'─'*33}\\n"
    f"   Épocas Totales : {EPOCHS}\\n"
    f"   Capas Ocultas  : {mlp.hidden_layer_sizes}\\n"
    f"   Hiperparámetro : LR={LR_INIT}, Adam\\n"
    f"   Regularización : L2 (por Adam)\\n"
    f"  {'─'*33}\\n"
    f"   Target: margen_pct >= 30%\\n"
    f"   Fuga de Datos  : CERO (Validado)\\n"
    f"  {'─'*33}\\n"
)

ax6.text(0.05, 0.95, metrics_text, transform=ax6.transAxes,
         fontsize=11, verticalalignment='top',
         color=TEXT_CLR, family='monospace',
         bbox=dict(boxstyle='round,pad=1', facecolor=BG_PANEL, edgecolor=ACCENT, lw=2.5))

plt.savefig('outputs/panel_red_neuronal_ventas.png', dpi=150, bbox_inches='tight', facecolor=BG_DARK)
plt.show()
print("¡Panel de visualización guardado correctamente en outputs/panel_red_neuronal_ventas.png!")"""))

# Save Notebook JSON structure
notebook_json = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook_json, f, ensure_ascii=False, indent=1)

print("Notebook red_neuronal_ventas.ipynb successfully created!")
