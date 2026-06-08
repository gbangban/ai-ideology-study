import shutil, os
cache = "/app/unsloth_compiled_cache"
if os.path.exists(cache):
    shutil.rmtree(cache)
    print(f"Deleted {cache}")
else:
    print(f"{cache} does not exist")
