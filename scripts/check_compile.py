import inspect
from unsloth.models import rl

src = inspect.getsource(rl)
lines = src.split('\n')

# Print lines around 1385-1430 (where compile options are patched)
for i in range(1385, min(1430, len(lines))):
    print(f"{i}: {lines[i]}")
