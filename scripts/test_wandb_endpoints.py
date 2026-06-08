import requests

# Check what endpoints exist
endpoints = [
    "http://wandb-server:8080/",
    "http://wandb-server:8080/api/v1/settings",
    "http://wandb-server:8080/graphql",
    "http://wandb-server:8080/app",
]

for url in endpoints:
    r = requests.get(url, allow_redirects=False)
    print(f"GET {url}: {r.status_code}, redirect: {r.headers.get('location', 'none')}, type: {r.headers.get('content-type', '?')[:50]}")

# Try the graphql endpoint with basic auth
import base64
api_key = "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"
creds = base64.b64encode(f"wandb:{api_key}".encode()).decode()

query = """
{
  viewer {
    id
    flags
  }
}
"""
r = requests.post("http://wandb-server:8080/graphql", headers={"Authorization": f"Basic {creds}"}, json={"query": query})
print(f"\nGraphQL Basic auth: {r.status_code}")
print(f"Body: {r.text[:300]}")

# Try without auth
r2 = requests.post("http://wandb-server:8080/graphql", json={"query": query})
print(f"\nGraphQL no auth: {r2.status_code}")
print(f"Body: {r2.text[:300]}")
