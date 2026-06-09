import sys, os, json, urllib.request, urllib.error

print("=== Trackio Diagnostic ===")
print(f"Python: {sys.version}")
print(f"Working dir: {os.getcwd()}")

# Check if trackio is installed
try:
    import trackio
    print(f"trackio file: {trackio.__file__}")
    print(f"trackio dir: {[x for x in dir(trackio) if not x.startswith('_')]}")
except ImportError as e:
    print(f"FAIL: trackio not importable: {e}")
    sys.exit(1)

# Check env vars
server_url = os.environ.get("TRACKIO_SERVER_URL", "NOT SET")
project = os.environ.get("TRACKIO_PROJECT", "NOT SET")
write_token = os.environ.get("TRACKIO_WRITE_TOKEN", "NOT SET")
print(f"TRACKIO_SERVER_URL: {server_url}")
print(f"TRACKIO_PROJECT: {project}")
print(f"TRACKIO_WRITE_TOKEN: {'SET' if write_token else 'NOT SET'}")

# Try to reach the server
try:
    req = urllib.request.Request(server_url, method="HEAD")
    resp = urllib.request.urlopen(req, timeout=5)
    print(f"Server reachable: HTTP {resp.status}")
except Exception as e:
    print(f"Server unreachable: {e}")

# Try init + log + finish
try:
    run = trackio.init(
        project=project,
        name="diagnostic-test",
        config={"test": True},
        server_url=server_url,
        auto_log_gpu=True,
    )
    print(f"trackio.init succeeded, run={run}")
    print(f"run type: {type(run)}")
    print(f"run dir: {[x for x in dir(run) if not x.startswith('_')]}")

    trackio.log({"test_metric": 42})
    print("trackio.log succeeded")

    trackio.finish()
    print("trackio.finish succeeded")
except Exception as e:
    print(f"FAIL during init/log/finish: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
