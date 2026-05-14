import sys
sys.path.insert(0, '/app')
from src.teacher.prompts import generate_dm_messages
msgs = generate_dm_messages("Test question?")
for m in msgs:
    print(f"--- {m['role']} ---")
    print(m['content'][:800])
    print()
