import json
import os

cells = []

def add_md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [text]
    })

def add_code(text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.split('\n')]
    })

# ============================================================
# CELDA 0: Título
# ============================================================
add_md("# Pipeline Integral - Abastecimiento y Logística\nEste notebook contiene el pipeline completo:\n\nLimpieza segura → Correlación → PCA → Balanceo → Red Neuronal → Exportación de Modelos (.pkl)")

# ============================================================
# CELDA 1: Imports
# ============================================================
add_code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymongo
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

os.makedirs('outputs', exist_ok=True)
os.makedirs('models', exist_ok=True)
print("✅ Librerías cargadas y carpetas creadas.")""")

# ============================================================
# CELDA 2: Carga MongoDB + Flattening
# ============================================================
add_md("## 1. Carga de Datos desde MongoDB y Flattening")

add_code("""client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["ProyectoBueno"]
collection = db["fact_abastecimiento"]
data = list(collection.find())
print(f"✅ Registros cargados: {len(data)}")

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            items.append((new_key, str(v)))
        else:
            items.append((new_key, v))
    return dict(items)

flat_data = [flatten_dict(record) for record in data]
df = pd.DataFrame(flat_data)
print(f"📊 Dimensiones iniciales aplanadas: {df.shape}")""")

# ============================================================
# CELDA 3: Creación del Y Objetivo
# ============================================================
add_md("## 2. Creación del 'Y' Objetivo")

add_code("""# Calculamos el ratio de entrega
if 'cantidad_recibida' in df.columns and 'cantidad_solicitada' in df.columns:
    df['cantidad_recibida'] = pd.to_numeric(df['cantidad_recibida'], errors='coerce').fillna(0)
    df['cantidad_solicitada'] = pd.to_numeric(df['cantidad_solicitada'], errors='coerce').fillna(1)
    df['cantidad_solicitada'] = df['cantidad_solicitada'].replace(0, 1)
    ratio = df['cantidad_recibida'] / df['cantidad_solicitada']

    # Terciles relativos: garantiza SIEMPRE 3 clases balanceadas
    try:
        df['y_target'] = pd.qcut(ratio, q=3, labels=[0, 1, 2], duplicates='drop').astype(int)
        n_clases = df['y_target'].nunique()
        if n_clases < 3:
            raise ValueError("qcut produjo menos de 3 clases")
        print("✅ Target 'y_target' creado con Terciles Relativos (3 clases).")
    except (ValueError, TypeError):
        # Fallback: usar percentiles manuales para forzar 3 clases
        p33 = ratio.quantile(0.33)
        p66 = ratio.quantile(0.66)
        df['y_target'] = np.where(ratio <= p33, 0, np.where(ratio <= p66, 1, 2))
        print("✅ Target 'y_target' creado con percentiles manuales (3 clases).")
else:
    numeric_cols_temp = df.select_dtypes(include=[np.number]).columns
    df['y_target'] = (df[numeric_cols_temp[0]] > df[numeric_cols_temp[0]].median()).astype(int)
    print("⚠️ Target binario creado como fallback.")

print(df['y_target'].value_counts().sort_index())""")

# ============================================================
# CELDA 4: Limpieza + Imputación
# ============================================================
add_md("## 3. Feature Engineering, Limpieza Segura e Imputación")

add_code("""# --- A. INGENIERÍA DE CARACTERÍSTICAS (Extrayendo oro de las fechas) ---
print("⚙️ Aplicando Feature Engineering...")
fechas_cols = [c for c in df.columns if 'fecha' in c.lower()]
for col in fechas_cols:
    try:
        # Convertimos a datetime
        df_dt = pd.to_datetime(df[col], errors='coerce')
        # Extraemos el mes y el día de la semana (0=Lunes, 6=Domingo)
        df[f'{col}_mes'] = df_dt.dt.month
        df[f'{col}_dia_semana'] = df_dt.dt.dayofweek
    except:
        pass

# --- B. FILTRO SEGURO Y PREVENCIÓN DE FUGA DE DATOS ---
# CRÍTICO: Eliminamos las variables con las que calculamos el Y
cols_to_drop = ['cantidad_recibida', 'cantidad_solicitada'] 

for col in df.columns:
    cl = col.lower()
    is_id = (cl == 'id' or cl == '_id' or cl == '_collection'
             or cl.startswith('id_') or cl.endswith('_id')
             or '_id_' in cl or 'id_venta_dw' in cl)
    # Agregamos las columnas de fecha originales para borrarlas, ya que salvamos el mes/día
    is_junk = ('telefono' in cl or 'correo' in cl or 'email' in cl
               or 'codigo_barras' in cl or 'codigo_producto' in cl
               or cl in fechas_cols)
    
    if (is_id or is_junk) and col != 'y_target' and col not in cols_to_drop:
        cols_to_drop.append(col)

df_clean = df.drop(columns=cols_to_drop, errors='ignore')
print(f"✅ Fuga de datos corregida y {len(cols_to_drop)} columnas basura eliminadas.")
print(f"📊 Dimensiones limpias (con features nuevas): {df_clean.shape}")

# Separar Target de Features
y = df_clean['y_target'].copy()
X_raw = df_clean.drop(columns=['y_target']).copy()

# Identificar tipos
numeric_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_raw.select_dtypes(exclude=[np.number]).columns.tolist()

# IMPUTACIÓN
for col in numeric_cols:
    X_raw[col] = X_raw[col].fillna(X_raw[col].median())
for col in categorical_cols:
    mode_val = X_raw[col].mode()
    fill_val = mode_val.iloc[0] if not mode_val.empty else 'UNKNOWN'
    X_raw[col] = X_raw[col].fillna(fill_val)

print("✅ Valores nulos imputados correctamente.")""")

# ============================================================
# CELDA 5: OneHotEncoder + Normalización
# ============================================================
add_md("## 4. Codificación (Frontend-Safe) y Normalización")

add_code("""# ONE-HOT ENCODER PREPARADO PARA PRODUCCIÓN
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_cat_encoded = encoder.fit_transform(X_raw[categorical_cols])

feature_names = encoder.get_feature_names_out(categorical_cols)
X_cat_df = pd.DataFrame(X_cat_encoded, columns=feature_names, index=X_raw.index)

X_numeric = X_raw[numeric_cols]
X_encoded = pd.concat([X_numeric, X_cat_df], axis=1)

print(f"📊 Dimensiones tras One-Hot: {X_encoded.shape}")

# NORMALIZACIÓN
scaler = QuantileTransformer(output_distribution='normal', random_state=42)
X_scaled_arr = scaler.fit_transform(X_encoded)
X_scaled = pd.DataFrame(X_scaled_arr, columns=X_encoded.columns, index=X_encoded.index)
print("✅ Datos normalizados.")""")

# ============================================================
# CELDA 6: Correlación
# ============================================================
add_md("## 5. Análisis de Correlación (Filtro Antirruido)")

add_code("""corr_matrix = X_scaled.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop_corr = [column for column in upper.columns if any(upper[column] > 0.90)]

X_uncorrelated = X_scaled.drop(columns=to_drop_corr)
print(f"✅ Columnas altamente correlacionadas eliminadas: {len(to_drop_corr)}")
print(f"📊 Dimensiones sin redundancia: {X_uncorrelated.shape}")""")

# ============================================================
# CELDA 7: PCA
# ============================================================
add_md("## 6. PCA: Reducción de Dimensionalidad para Entrenamiento")

add_code("""pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_uncorrelated)

print(f"✅ PCA Completado.")
print(f"📊 Dimensiones FINALES para la Red Neuronal: {X_pca.shape}")
print(f"📈 Con {X_pca.shape[1]} componentes explicamos el 95% de la varianza.")""")

# ============================================================
# CELDA 8: Balanceo
# ============================================================
add_md("## 7. Balanceo de Clases (SMOTE - Synthetic Minority Over-sampling)")

add_code("""# Importación necesaria (asegúrate de tener imblearn instalado)
from imblearn.over_sampling import SMOTE

print("⚖️ Balanceando clases con SMOTE...")

# Aplicamos SMOTE directamente sobre los datos reducidos por PCA
smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X_pca, y)

print("✅ Clases Balanceadas exitosamente (Sin duplicados exactos):")
for c, count in zip(*np.unique(y_bal, return_counts=True)):
    print(f"  Clase {c}: {count} muestras")""")

# ============================================================
# CELDA 9: Red Neuronal
# ============================================================
add_md("## 8. Entrenamiento de la Red Neuronal (El Cerebro)")

add_code("""X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal)

print("🧠 Entrenando Red Neuronal (MLPClassifier)...")
nn_model = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42,
    early_stopping=True)

nn_model.fit(X_train, y_train)
y_pred = nn_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"✅ Entrenamiento completado en {nn_model.n_iter_} épocas.")
print(f"🎯 Accuracy en Test: {acc*100:.2f}%")
print()
print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión - Red Neuronal (Test)')
plt.ylabel('Valor Real')
plt.xlabel('Predicción')
plt.tight_layout()
plt.savefig('outputs/01_confusion_matrix.png')
plt.show()""")

# ============================================================
# CELDA 10: Dashboard Visual LDA + t-SNE
# ============================================================
add_md("## 9. Dashboard Visual de Dispersión (t-SNE) [Solo Reporte]")

add_code("""print("🎨 Generando dashboard visual de separabilidad con t-SNE...")

# t-SNE se alimenta directamente de X_bal (Alta dimensionalidad proveniente de PCA+SMOTE)
# Ajustamos perplexity a un valor estándar seguro (ej. 30)
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne_visual = tsne.fit_transform(X_bal)

sil_score = silhouette_score(X_tsne_visual, y_bal)
print(f"✅ Silhouette Score de la dispersión: {sil_score:.4f}")

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne_visual[:, 0], X_tsne_visual[:, 1],
                      c=y_bal, cmap='viridis', alpha=0.7, edgecolors='k', s=15)

plt.title(f'Dispersión 2D (PCA -> SMOTE -> t-SNE)\\nSilhouette Score: {sil_score:.4f}',
          fontsize=14, fontweight='bold')
plt.xlabel('t-SNE Dim 1')
plt.ylabel('t-SNE Dim 2')
plt.colorbar(scatter, ticks=np.unique(y_bal), label='Clases (Target)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/02_dispersion_tsne_corregido.png', dpi=150)
plt.show()""")

# ============================================================
# CELDA 11: Exportación .pkl
# ============================================================
add_md("## 10. Guardado Final para el Frontend (Archivos .pkl)")

add_code("""print("💾 Exportando moldes y modelo para el Frontend...")

joblib.dump(encoder, 'models/01_onehot_encoder.pkl')
joblib.dump(scaler, 'models/02_scaler.pkl')
joblib.dump(to_drop_corr, 'models/03_cols_to_drop_corr.pkl')
joblib.dump(pca, 'models/04_pca.pkl')
joblib.dump(nn_model, 'models/05_neural_network.pkl')

print("✅ ¡Exportación exitosa! Archivos .pkl en la carpeta 'models/'.")
print("🚀 EL BACKEND ESTÁ LISTO PARA PRODUCCIÓN.")""")


# ============================================================
# Escribir notebook
# ============================================================
notebook_json = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.9.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_path = "fact_abastecimiento_logistica/00_pipeline_integral_abastecimiento.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook_json, f, indent=2)

print(f"Notebook creado exitosamente en {out_path}")
