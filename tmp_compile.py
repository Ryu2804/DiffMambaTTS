import json

with open('notebook/kaggle_f5tts_mamba3_training.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        print(f"Checking cell {i}")
        try:
            compile(source, f"cell_{i}", "exec")
        except SyntaxError as e:
            print(f"Syntax error in cell {i}: {e}")
