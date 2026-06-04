import json

files = [
    'fact_competencia/01_analisis_correlacion_separabilidad.ipynb',
    'fact_evaluacion_proveedores/01_analisis_correlacion_separabilidad.ipynb'
]

for filepath in files:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            nb = json.load(f)

        changed = False
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                new_source = []
                for line in cell['source']:
                    # Fix the bad indentation for "else:" that comes after "if len(highly_corr) > 0:"
                    if line == "    else:\n" or line == "    else:":
                        line = line.replace("    else", "else")
                        changed = True
                    new_source.append(line)
                cell['source'] = new_source

        if changed:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=1, ensure_ascii=False)
            print(f"Fixed indentation in {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
