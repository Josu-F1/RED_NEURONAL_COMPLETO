import json
import os
import subprocess

def create_notebook(filename, cells_content):
    cells = []
    for cell_type, content in cells_content:
        if cell_type == 'code':
            cell = {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + '\n' for line in content.split('\n')]
            }
        else:
            cell = {
                "cell_type": "markdown",
                "metadata": {},
                "source": [line + '\n' for line in content.split('\n')]
            }
        if cell["source"]:
            cell["source"][-1] = cell["source"][-1].rstrip('\n')
        cells.append(cell)

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)


datasets = [
    'fact_abastecimiento_logistica',
    'fact_competencia',
    'fact_evaluacion_proveedores'
]

for ds in datasets:
    cells = [
        ('markdown', '# Construcción de Red Neuronal (MLP)'),
        ('code', '''import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import warnings
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", palette="deep")'''),
        ('markdown', '## 1. Carga de Datos Preprocesados'),
        ('code', '''import os
# Cargamos directamente los datos procesados del notebook 01
data_path = 'outputs/01_processed_data.npz'
if not os.path.exists(data_path):
    raise FileNotFoundError("¡No se encontró el archivo de datos! Asegúrate de ejecutar el Notebook 01 primero.")

data = np.load(data_path, allow_pickle=True)
X = data['X_scaled']
y = data['y']

print(f"Dimensiones de X: {X.shape}")
print(f"Dimensiones de y: {y.shape}")'''),
        ('markdown', '## 2. División de Datos (Train / Test)'),
        ('code', '''X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Set de Entrenamiento: {X_train.shape[0]} muestras")
print(f"Set de Prueba (Test): {X_test.shape[0]} muestras")'''),
        ('markdown', '## 3. Entrenamiento de la Red Neuronal'),
        ('code', '''# Arquitectura de 2 capas ocultas (100 neuronas, 50 neuronas)
print("Iniciando entrenamiento de la Red Neuronal...")
nn = MLPClassifier(hidden_layer_sizes=(100, 50), 
                   max_iter=500, 
                   activation='relu', 
                   solver='adam', 
                   random_state=42, 
                   early_stopping=True, 
                   n_iter_no_change=10)

nn.fit(X_train, y_train)
print(f"Entrenamiento completado en {nn.n_iter_} iteraciones.")'''),
        ('markdown', '## 4. Evaluación de Precisión'),
        ('code', '''# Predicciones
y_pred_train = nn.predict(X_train)
y_pred_test = nn.predict(X_test)

# Cálculo de precisión (Accuracy)
acc_train = accuracy_score(y_train, y_pred_train)
acc_test = accuracy_score(y_test, y_pred_test)

print(f"==========================================")
print(f"Precisión en Entrenamiento: {acc_train*100:.2f}%")
print(f"Precisión en Prueba (Test): {acc_test*100:.2f}%")
print(f"==========================================\\n")

print("Reporte de Clasificación (Test):")
print(classification_report(y_test, y_pred_test))'''),
        ('markdown', '## 5. Matriz de Confusión'),
        ('code', '''plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión - Red Neuronal')
plt.ylabel('Valor Real')
plt.xlabel('Predicción')
plt.show()''')
    ]
    
    out_path = f"{ds}/03_red_neuronal.ipynb"
    create_notebook(out_path, cells)

print("Notebooks de Red Neuronal creados exitosamente.")
