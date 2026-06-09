import os
for k in ["TRACKIO_SERVER_URL", "TRACKIO_PROJECT", "TRACKIO_WRITE_TOKEN", "TRACKIO_DIR"]:
    v = os.environ.get(k, "<NOT SET>")
    if "TOKEN" in k:
        v = v[:10] + "..." if len(v) > 10 else v
    print(f"  {k}={v}")
