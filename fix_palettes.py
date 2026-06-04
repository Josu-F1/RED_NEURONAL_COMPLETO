import json
import os

datasets = [
    'fact_abastecimiento_logistica',
    'fact_evaluacion_proveedores'
]

for ds in datasets:
    nb_path = f"{ds}/02_pca_tsne_pipeline.ipynb"
    if os.path.exists(nb_path):
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                new_source = []
                for line in cell['source']:
                    if "palette=['#EF476F', '#06D6A0']" in line:
                        line = line.replace("palette=['#EF476F', '#06D6A0']", "palette=['#EF476F', '#06D6A0', '#FFD166']")
                    new_source.append(line)
                cell['source'] = new_source
                
        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)

print("Paletas de color actualizadas a 3 clases.")
