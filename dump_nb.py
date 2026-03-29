import json

with open('notebook/kaggle_f5tts_mamba3_training.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('tmp_full_nb.py', 'w', encoding='utf-8') as f:
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            f.write(f"\n# === CELL {i} ===\n")
            f.write(source)
            f.write("\n")
