import json

filepath = 'fact_abastecimiento_logistica/01_analisis_correlacion_separabilidad.ipynb'
with open(filepath, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "tsne = TSNE(" in source and "X_tsne = tsne.fit_transform" in source:
            cell['source'] = [
                "# PCA previo a t-SNE para reducir dimensionalidad (Soluciona el problema de las 1300+ columnas)\n",
                "print(\"⏳ Reduciendo dimensionalidad con PCA antes de t-SNE...\")\n",
                "pca_tsne = PCA(n_components=min(50, X_scaled.shape[1]))\n",
                "X_pca_for_tsne = pca_tsne.fit_transform(X_scaled)\n",
                "\n",
                "# t-SNE\n",
                "print(\"⏳ Calculando t-SNE 2D...\")\n",
                "tsne = TSNE(n_components=2, perplexity=min(30, len(X_scaled)//3), random_state=42, n_iter=1000, n_jobs=-1)\n",
                "X_tsne = tsne.fit_transform(X_pca_for_tsne)\n",
                "print(f\"✅ t-SNE completado\")\n"
            ]

with open(filepath, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("t-SNE cell fixed")
