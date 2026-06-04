import json

datasets = [
    'fact_abastecimiento_logistica',
    'fact_competencia',
    'fact_evaluacion_proveedores'
]

for ds in datasets:
    nb_path = f"{ds}/02_pca_tsne_pipeline.ipynb"
    try:
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                new_source = []
                for line in cell['source']:
                    if "tsne = TSNE(" in line:
                        line = line.replace("max_iter=1000", "max_iter=300, init='pca', learning_rate='auto'")
                    new_source.append(line)
                cell['source'] = new_source
                
        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            
    except Exception as e:
        print(f"Error {ds}: {e}")

print("Optimizacion TSNE completada.")
