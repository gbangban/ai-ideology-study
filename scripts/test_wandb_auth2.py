import requests, base64

# Try different auth methods against port 8080
api_key = "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"

# 1. Basic auth
creds = base64.b64encode(f"wandb:{api_key}".encode()).decode()
r1 = requests.post("http://wandb-server:8080/api/v1/settings", headers={"Authorization": f"Basic {creds}", "Content-Type": "application/json"})
print(f"POST /api/v1/settings Basic: {r1.status_code}, content-type: {r1.headers.get('content-type', '?')}")

# 2. X-Api-Key header
r2 = requests.post("http://wandb-server:8080/api/v1/settings", headers={"X-Api-Key": api_key, "Content-Type": "application/json"})
print(f"POST /api/v1/settings X-Api-Key: {r2.status_code}, content-type: {r2.headers.get('content-type', '?')}")

# 3. Try /graphql endpoint
r3 = requests.post("http://wandb-server:8080/api/v1/settings", headers={"Authorization": f"Basic {creds}", "Content-Type": "application/json"}, json={})
print(f"POST /api/v1/settings empty json: {r3.status_code}")
print(f"Body: {r3.text[:300]}")

# 4. Check if the W&B local server has a known API key
r4 = requests.get("http://wandb-server:8080/api/v1/settings", headers={"Authorization": "Basic d2FuZGI6d2FuZGI="})
print(f"Basic wandb:wandb: {r4.status_code}, content-type: {r4.headers.get('content-type', '?')}")
