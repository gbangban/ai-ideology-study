# WSL2 GPU Setup Analysis - 2026-04-29

## Problem
Docker GPU passthrough fails on WSL2. NVIDIA Container Toolkit 1.19.0 cannot find `libnvidia-ml.so.1` (lives at `/usr/lib/wsl/lib/` not standard path). `load-kmods = true` also fails since WSL2 has no kernel modules.

## Environment
- WSL2 Ubuntu 24.04, NVIDIA driver 595.97, CUDA 13.2, RTX 5090 (32GB)
- Docker Engine inside WSL2 (separate daemon from Docker Desktop)
- Python 3.9.12 (anaconda base), conda env `ai-projects` (Python 3.12.3, empty)
- Docker Desktop on Windows with WSL2 backend integration

## Decision: Option 1 (Docker Desktop) - CHosen

Chose Docker Desktop as the single Docker daemon. WSL2 Docker Engine will be removed.

### Why
- GPU passthrough works automatically via Docker Desktop
- Unsloth Studio container (`silly_blackwell`) already on Docker Desktop
- Project container (`suspicious_spence`) already ran successfully on Docker Desktop (exit 0)
- WSL2 Docker Engine GPU fix is fragile (breaks on WSL2 updates)

### CLI Workaround
Named pipe limitation from WSL2 (`docker logs/exec/inspect` fail) solved via:
- `ddk` script at `scripts/ddk` - bridges to Windows PowerShell for full CLI access
- `docker compose` commands work natively from WSL2 through named pipe
- See `.kilo/plans/docker-integration-analysis.md` Section 7 for full details

### Migration Steps
```bash
# 1. Uninstall WSL2 Docker Engine
sudo apt remove docker-ce docker-ce-cli docker-compose-plugin docker-buildx-plugin docker-ce-rootless-extras

# 2. Clean up
sudo rm -rf /var/lib/docker /var/run/docker.sock

# 3. Set default context
docker context set-default desktop-linux

# 4. Rebuild on Docker Desktop
docker compose up --build -d
```
