import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
import os
import json
import pymongo
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo visual premium
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.family'] = 'sans-serif'

# Crear directorio de outputs
os.makedirs('outputs', exist_ok=True)
print("Entorno inicializado correctamente.")


# Cargar JSON crudo desde la raíz
# Cargar datos desde MongoDB
import pymongo
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["ProyectoBueno"]
collection = db["fact_evaluacion_proveedores"]
data = list(collection.find())

# Aplanar registros recursivamente
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
print(f"Dimensiones iniciales aplanadas: {df.shape}")

# Calcular columna target
df["y_desempeno"] = pd.cut(df["calificacion_entrega"], bins=[-np.inf, 2, 4, np.inf], labels=[0, 1, 2]).astype(int)
print(f"Target 'y_desempeno' agregado. Distribución original:")
print(df["y_desempeno"].value_counts(dropna=False))


# Paso 1: Limpieza de columnas e imputación de nulos
shape_inicial = df.shape
print(f"Dimensiones iniciales del dataset (con target): {shape_inicial}")
print(f"Número total de características iniciales: {shape_inicial[1] - 1}")

cols_to_drop = []
for col in df.columns:
    col_lower = col.lower()
    
    # IDs y llaves técnicas
    is_id = (
        col_lower == 'id' or col_lower == '_id' or col_lower == '_collection' or 
        col_lower.startswith('id_') or col_lower.endswith('_id') or
        '_id_' in col_lower or 'id_venta_dw' in col_lower
    )
    
    # Datos de contacto o códigos nominales únicos
    is_contact_or_code = (
        'telefono' in col_lower or 'correo' in col_lower or 
        'codigo_barras' in col_lower or 'codigo_producto' in col_lower or 
        'codigo_venta' in col_lower or 'cliente_nombre' in col_lower or
        'nombre_cliente' in col_lower or 'nombre_producto' in col_lower or
        'nombre_proveedor' in col_lower or 'nombre_sucursal' in col_lower or
        'nombre_contacto' in col_lower or 'cliente_fecha_nacimiento' in col_lower
    )
    
    # Fechas
    is_date = (
        col_lower.endswith('_fecha') or col_lower.endswith('_hora') or 
        col_lower == 'fecha' or col_lower == 'hora' or col_lower == 'tiempo_fecha'
    )
    
    # Evitar eliminar el target
    if (is_id or is_contact_or_code or is_date) and col != "y_desempeno":
        cols_to_drop.append(col)

# Droppear columnas identificadas
df = df.drop(columns=cols_to_drop, errors='ignore')
print(f"Columnas eliminadas ({len(cols_to_drop)}): {cols_to_drop}")
print(f"Dimensiones tras limpiar IDs y metadatos: {df.shape}")
print(f"Número de características tras limpieza: {df.shape[1] - 1}")

# Imputación de nulos
nulos_detectados = df.isnull().sum().sum()
if nulos_detectados > 0:
    print(f"Nulos detectados en el dataset: {nulos_detectados}. Imputando...")
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                df[col] = df[col].fillna(df[col].median())
            else:
                mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 'UNKNOWN'
                df[col] = df[col].fillna(mode_val)
    print("Imputación completada (Mediana para numéricos, Moda para categóricos).")
else:
    print("No se detectaron valores nulos en el dataset.")

# Paso 2: One-Hot Encoding
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
if "y_desempeno" in categorical_cols:
    categorical_cols.remove("y_desempeno")

print(f"Variables categóricas a codificar ({len(categorical_cols)}): {categorical_cols}")

df = pd.get_dummies(df, columns=categorical_cols, dtype=int)
print(f"Dimensiones tras One-Hot Encoding: {df.shape}")
print(f"Número total de características finales: {df.shape[1] - 1}")

# Paso 3: Separación y normalización (Split-First)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
import numpy as np
import pandas as pd
import joblib
import os

y = df["y_desempeno"].values.astype(int)
X = df.drop(columns=["y_desempeno"]).values.astype(float)
feature_names = df.drop(columns=["y_desempeno"]).columns.tolist()

# Split-First Protocol
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

qt = QuantileTransformer(output_distribution='normal', random_state=42)
X_train_scaled = qt.fit_transform(X_train)
X_test_scaled = qt.transform(X_test)

os.makedirs("outputs", exist_ok=True)
joblib.dump(qt, "outputs/quantile_transformer.joblib")

print(f"Split completo:")
print(f"  Train: {X_train.shape} | Clases: {np.bincount(y_train)}")
print(f"  Test:  {X_test.shape}  | Clases: {np.bincount(y_test)}")
# Paso 4: t-SNE de varianza de datos completos (antes de balancear)
from sklearn.preprocessing import QuantileTransformer
X_scaled = QuantileTransformer(output_distribution='normal', random_state=42).fit_transform(X)
# Paso 4: t-SNE de varianza de datos completos (antes de balancear)
print("Calculando proyección t-SNE 2D para el dataset completo...")
tsne = TSNE(n_components=2, perplexity=min(30, len(y)-1), random_state=42, n_jobs=-1)
X_tsne = tsne.fit_transform(X_scaled)

# Configuración de gráfico premium en modo oscuro
plt.figure(figsize=(10, 8))
BG_DARK  = '#0D1117'
BG_PANEL = '#161B22'
TEXT_CLR = '#E6EDF3'
GRID_CLR = '#30363D'
ACCENT   = '#58A6FF'

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
    'font.family':      'sans-serif',
})

unique_classes = np.unique(y)
colors = ['#EF476F', '#06D6A0', '#FFD166']
class_names_map = {0: 'Bajo Desempeño (0)', 1: 'Desempeño Medio (1)', 2: 'Alto Desempeño (2)'}

for idx_c, val_c in enumerate(unique_classes):
    color = colors[idx_c % len(colors)]
    name = class_names_map.get(val_c, f"Clase {val_c}")
    plt.scatter(
        X_tsne[y == val_c, 0], X_tsne[y == val_c, 1],
        color=color, label=name, alpha=0.7, s=25
    )

plt.title('Separabilidad y Varianza del Dataset completo — t-SNE 2D', fontsize=14, fontweight='bold', color=ACCENT, pad=15)
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.legend(frameon=True, facecolor=BG_PANEL, edgecolor=GRID_CLR)
plt.grid(True)

tsne_plot_path = 'outputs/tsne_varianza_datos.png'
plt.savefig(tsne_plot_path, dpi=150, bbox_inches='tight', facecolor=BG_DARK)
plt.show()
print(f"¡Gráfico de varianza t-SNE guardado exitosamente en: {tsne_plot_path}!")

# Sobremuestreo aleatorio (Random Oversampling) manual en TRAIN
unique_classes, class_counts = np.unique(y_train, return_counts=True)
max_class_count = np.max(class_counts)

X_balanced = []
y_balanced = []

for c in unique_classes:
    idx = np.where(y_train == c)[0]
    if len(idx) < max_class_count:
        # Muestrear con reemplazo
        np.random.seed(42)
        resampled_idx = np.random.choice(idx, size=max_class_count, replace=True)
        X_balanced.append(X_train_scaled[resampled_idx])
        y_balanced.append(y_train[resampled_idx])
    else:
        X_balanced.append(X_train_scaled[idx])
        y_balanced.append(y_train[idx])

X_balanced = np.vstack(X_balanced)
y_balanced = np.concatenate(y_balanced)

# Mezclar aleatoriamente el dataset balanceado
np.random.seed(42)
shuffle_idx = np.random.permutation(len(y_balanced))
X_train_bal = X_balanced[shuffle_idx]
y_train_bal = y_balanced[shuffle_idx]

print(f"Distribución del target DESPUÉS del balanceo en Train:")
classes_bal, counts_bal = np.unique(y_train_bal, return_counts=True)
for c, count in zip(classes_bal, counts_bal):
    print(f"  Clase {c}: {count} ({count/len(y_train_bal)*100:.1f}%)")

# Selección de Características (SelectKBest ANOVA F-value)
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(score_func=f_classif, k=30)
X_train_bal_sel = selector.fit_transform(X_train_bal, y_train_bal)
X_test_sel = selector.transform(X_test_scaled)

selected_feature_indices = selector.get_support(indices=True)
selected_feature_names = [feature_names[i] for i in selected_feature_indices]

print(f"\nSelección de características completada:")
print(f"  Train original: {X_train_bal.shape} -> Seleccionado: {X_train_bal_sel.shape}")
print(f"  Test original:  {X_test_scaled.shape} -> Seleccionado: {X_test_sel.shape}")

import json
import numpy as np

# Load existing meta
meta_path = r"c:\Users\Marlon\Documents\Sexto\BI\MiProyecto\RED_NEURONAL_COMPLETO\fact_evaluacion_proveedores\outputs\inference_model.json"
with open(meta_path, 'r', encoding='utf-8') as f:
    meta = json.load(f)

meta['original_feature_names'] = feature_names
meta['selected_feature_indices'] = selected_feature_indices.tolist()

with open(meta_path, 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
print("Updated meta in " + meta_path)
