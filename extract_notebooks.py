import json

notebooks = [
    'fact_ventas/red_neuronal_ventas.ipynb',
    'fact_abastecimiento_logistica/03_red_neuronal.ipynb',
    'fact_competencia/03_red_neuronal.ipynb'
]

for nb_path in notebooks:
    try:
        with open(nb_path, encoding='utf-8') as f:
            nb = json.load(f)
        out_path = 'temp_' + nb_path.split('/')[-1].replace('.ipynb', '.py')
        with open(out_path, 'w', encoding='utf-8') as out:
            code_cells = [''.join(cell['source']) for cell in nb['cells'] if cell['cell_type'] == 'code']
            out.write('\n\n'.join(code_cells))
        print(f"Extracted {nb_path} to {out_path}")
    except Exception as e:
        print(f"Error extracting {nb_path}: {e}")
