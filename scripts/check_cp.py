import os, json
path = '/app/data/processed/checkpoint.json'
if os.path.exists(path):
    with open(path) as f:
        data = json.load(f)
    print(f"Checkpoint exists: {data['completed_count']} samples")
    print(f"First sample content preview: {data['samples'][0]['conversations'][1]['content'][:200]}")
else:
    print("No checkpoint file")
