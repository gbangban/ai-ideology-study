import urllib.request, sys

for port in [8080, 8086]:
    try:
        url = f"http://wandb-server:{port}"
        resp = urllib.request.urlopen(url, timeout=5)
        print(f"Port {port}: HTTP {resp.status}")
    except Exception as e:
        print(f"Port {port}: {type(e).__name__}: {e}")
