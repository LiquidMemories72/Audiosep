import json

notebook_path = 'kaggle_train_libri4mix.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# The last cell is the training cell
source = notebook['cells'][-1]['source']

for i, line in enumerate(source):
    if line.startswith('!python scripts/train.py'):
        source[i] = '!torchrun --standalone --nproc_per_node=2 scripts/train.py configs/kaggle_libri4mix.yaml \\\n'
        break

notebook['cells'][-1]['source'] = source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)
