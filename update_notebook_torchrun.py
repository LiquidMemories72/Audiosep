import json
notebook_path = 'kaggle_train_libri4mix.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)
for cell in notebook['cells']:
    for i, line in enumerate(cell.get('source', [])):
        if line.startswith('!torchrun --standalone --nproc_per_node=2 scripts/train.py'):
            cell['source'][i] = '!torchrun --standalone --nproc_per_node=2 scripts/train.py configs/kaggle_libri4mix.yaml \\\n'
            for j in range(i+1, len(cell['source'])):
                if '--data_folder' in cell['source'][j]:
                    cell['source'].insert(j, '    --precision fp16 \\\n')
                    break
            break
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)