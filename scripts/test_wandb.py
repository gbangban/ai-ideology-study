import requests

# Test without auth
r1 = requests.get("http://wandb-server:8080/api/v1/settings")
print(f"No auth: {r1.status_code}")

# Test with basic auth
import base64
api_key = "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"
creds = base64.b64encode(f"wandb:{api_key}".encode()).decode()
r2 = requests.get("http://wandb-server:8080/api/v1/settings", headers={"Authorization": f"Basic {creds}"})
print(f"Basic auth: {r2.status_code}")
print(f"Response type: {r2.headers.get('content-type', 'unknown')}")
print(f"Body preview: {r2.text[:200]}")

# Try with X-Api-Key header instead
r3 = requests.get("http://wandb-server:8080/api/v1/settings", headers={"X-Api-Key": api_key})
print(f"X-Api-Key: {r3.status_code}")
print(f"Body preview: {r3.text[:200]}")
