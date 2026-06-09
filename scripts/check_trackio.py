import os, json
d = "/tmp/trackio-cache"
if os.path.isdir(d):
    print("Contents:", os.listdir(d))
    for f in os.listdir(d):
        fp = os.path.join(d, f)
        if os.path.isfile(fp):
            print(f"  {f}: {os.path.getsize(fp)} bytes")
        else:
            print(f"  {f}/ ({len(os.listdir(fp))} items)")
else:
    print(f"{d} does not exist")
