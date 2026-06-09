import json

path = r"c:\Users\Marlon\Documents\Sexto\BI\MiProyecto\RED_NEURONAL_COMPLETO\fact_abastecimiento_logistica\red_neuronal_abastecimiento.ipynb"

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

markdown_source = [
    "## 🎓 Apunte para la Defensa: Entrenamiento y Búsqueda de Hiperparámetros (Grid Search y 5-Fold CV)\n",
    "\n",
    "El jurado querrá saber cómo elegimos el tamaño de la red. ¡La respuesta es que no lo adivinamos, sino que hicimos que el código lo descubriera empíricamente probando distintas combinaciones!\n",
    "\n",
    "* Aplicamos **Búsqueda en Cuadrícula (Grid Search)**: Le dimos al código opciones de \"Hiperparámetros\". Por ejemplo, 3 arquitecturas de neuronas distintas (ej. 64-32, 32-20, y 16-10 neuronas) y 2 Tasas de Aprendizaje o *Learning Rates* (0.1 y 0.05). En total, el modelo evaluó matemáticamente cuál de estas 6 combinaciones era la mejor.\n",
    "* Aplicamos **Validación Cruzada de 5 Folds (Stratified K-Fold CV)**: Para asegurarnos de que el resultado no fuera \"suerte\", por CADA combinación de hiperparámetros, dividimos nuestros datos de entrenamiento en 5 bloques (Folds).\n",
    "  * **¿Qué significa probar 5 veces?** El modelo se entrena con 4 bloques de datos y se examina con el bloque sobrante. Luego, repite el proceso 5 veces, rotando el bloque de examen cada vez para que toda la información sirva como examen al menos una vez. \n",
    "  * Por lo tanto, con 6 combinaciones de hiperparámetros y 5 Folds, el código **entrenó 30 mini-redes neuronales distintas** internamente. Al final, se sacó un promedio de la precisión de esas 5 iteraciones.\n",
    "* Nuestro código escogió de manera automática la combinación \"Campeona\" (la que sacó el mejor promedio en las 5 pruebas de validación cruzada). Finalmente, con esos hiperparámetros óptimos, entrenamos nuestro \"Modelo Final\" definitivo a gran profundidad (por 1000 épocas)."
]

new_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": markdown_source
}

nb['cells'].append(new_cell)

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Apended markdown successfully.")
