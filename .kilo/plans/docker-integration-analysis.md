# Docker Integration Analysis - Unsloth Studio + DM-Align Project

> **Date**: 2026-04-29 | **Status**: Analysis | **Author**: Kilo

---

## 1. Current State

### Infrastructure
```
Windows Host
├── Docker Desktop (GPU-enabled, WSL2 backend)
│   └── silly_blackwell (Unsloth Studio, unsloth/unsloth image)
│       └── Web UI at localhost:8888
│   └── suspicious_spence (ml-lora-training-training:latest) [PREVIOUS RUN]
│
└── WSL2 (Ubuntu 24.04)
    ├── Docker Engine 29.3.1 (SEPARATE daemon, NO GPU access)
    │   └── ml-training:latest image (21.6GB, built 3h ago)
    │   └── chatterbox-tts containers
    └── /home/yao/projects/ml-lora-training/
        ├── src/teacher/          <- Data generation
        ├── src/student/          <- DPO training script
        ├── data/                 <- Datasets
        ├── docker/              <- Dockerfile + .dockerignore
        └── docker-compose.yml   <- Orchestrates project containers
```

### Verified Facts
- **RTX 5090**: 32GB VRAM, 31.8GB currently in use by something else
- **NVIDIA Driver**: 595.97, CUDA 13.2 on host
- **Docker Desktop**: `unsloth/unsloth` image for Studio, `ml-lora-training-training:latest` for project
- **WSL2 Docker Engine**: NVIDIA runtime 1.19.0 installed but GPU passthrough FAILS
- **WSL2 Docker GPU failure**: `libnvidia-ml.so.1` at `/usr/lib/wsl/lib/` not found inside containers
- **Docker Desktop GPU**: Works (suspicious_spence ran successfully with exit 0)
- **Studio container**: Exited 137 (OOM killed) 8 days ago
- **Project container on Docker Desktop**: `suspicious_spence` exited 0 - integration already worked

---

## 2. The Core Problem: Two Docker Daemons

```
┌──────────────────────────────────────────────────────────────────┐
│  DOCKER DESKTOP (Windows)                                        │
│  Context: desktop-linux                                          │
│  GPU: ✅ Works (NVIDIA runtime via Windows driver passthrough)    │
│  Containers: silly_blackwell (Studio), suspicious_spence (project)│
│  CLI from WSL2: ⚠️  ps/images only (named pipe, no logs/inspect)  │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  WSL2 DOCKER ENGINE (Ubuntu)                                     │
│  Context: default                                                │
│  GPU: ❌ FAILS - nvidia-container-runtime can't find libs         │
│  Containers: chatterbox-tts, ml-training:latest image only        │
│  CLI from WSL2: ✅ Full access (unix socket /var/run/docker.sock) │
└──────────────────────────────────────────────────────────────────┘
```

### Why WSL2 Docker Engine Can't Use GPU

```
nvidia-container-runtime (v1.19.0) tries to:
  1. Load libnvidia-ml.so.1 → lives at /usr/lib/wsl/lib/ (host path)
  2. Inside container, /usr/lib/wsl/lib/ doesn't exist
  3. Result: "cannot open shared object file: no such file or directory"

Config issue in /etc/nvidia-container-runtime/config.toml:
  - load-kmods = true → WSL2 has no kernel modules (should be false)
  - ldconfig = "@/usr/lib/wsl/lib" → points to dir, not binary
```

**Fixing WSL2 Docker GPU is possible but fragile** - requires `load-kmods = false`, CDI spec generation, and library path workarounds. Docker Desktop handles this automatically.

---

## 3. Consolidation Options: Detailed Consequence Analysis

### OPTION A: Migrate Everything to WSL2 Docker Engine

**Action**: Uninstall Docker Desktop GPU features, fix WSL2 Docker Engine for GPU, run ALL containers on WSL2 daemon.

#### What Changes
| Component | Before | After |
|-----------|--------|-------|
| Studio | Docker Desktop (`silly_blackwell`) | WSL2 Docker Engine (new container) |
| Project | Docker Desktop (`suspicious_spence`) | WSL2 Docker Engine (same docker-compose) |
| GPU Access | Docker Desktop handles it | Manual NVIDIA runtime config in WSL2 |
| CLI Access | Limited (`docker ps` only) | Full (`docker logs`, `inspect`, `exec`) |
| Docker Context | `desktop-linux` | `default` |

#### Consequences

**✅ Benefits**
1. **Full Docker CLI access** - `docker logs`, `docker exec`, `docker inspect` all work from WSL2 terminal
2. **Single daemon** - No context switching, no confusion about which containers are where
3. **Faster I/O** - WSL2 Docker Engine uses overlayfs directly, no Windows interop layer
4. **Debugging** - Can `docker exec -it ml-training bash` to debug inside containers
5. **No Docker Desktop license** - Docker Desktop requires paid license for large organizations
6. **Image sharing** - Both Studio and project containers share same image cache

**❌ Costs**
1. **GPU fix is fragile** - Requires manual config:
   ```toml
   # /etc/nvidia-container-runtime/config.toml
   load-kmods = false          # WSL2 has no kernel modules
   # Need CDI mode, not legacy mode
   ```
   Plus library path fixes. This breaks on WSL2 updates.
2. **Studio container recreation** - `silly_blackwell` won't transfer. Need to recreate with:
   ```bash
   docker run -d --gpus all -p 8888:8888 \
     -v /home/yao/projects/ml-lora-training:/workspace \
     --name silly_blackwell unsloth/unsloth
   ```
   Any Studio state (trained models, configs) inside the old container is lost unless exported first.
3. **NVIDIA runtime maintenance** - WSL2 updates may break `nvidia-container-runtime` compatibility
4. **No Docker Desktop UI** - Lose the Desktop app for container management (minor)
5. **Risk of breakage** - If NVIDIA runtime fails, ALL containers lose GPU access simultaneously

#### Verdict: **Risky** - GPU access in WSL2 Docker Engine is a known pain point that breaks on updates. Not worth the maintenance burden.

---

### OPTION B: Migrate Everything to Docker Desktop

**Action**: Uninstall WSL2 Docker Engine (`sudo apt remove docker-ce docker-ce-cli docker-compose-plugin docker-buildx-plugin`), use ONLY Docker Desktop for all containers.

#### What Changes
| Component | Before | After |
|-----------|--------|-------|
| Studio | Docker Desktop (stays) | Docker Desktop (unchanged) |
| Project | WSL2 Docker Engine image | Docker Desktop (docker-compose) |
| GPU Access | Already works | Already works |
| CLI Access | Full on WSL2 daemon | Limited from WSL2 (`ps` only) |
| Docker Context | `default` | `desktop-linux` |

#### Consequences

**✅ Benefits**
1. **GPU access guaranteed** - Docker Desktop handles NVIDIA passthrough automatically, survives WSL2 updates
2. **Studio stays intact** - `silly_blackwell` container preserved, no recreation needed
3. **Proven working** - `suspicious_spence` already ran successfully on Docker Desktop
4. **Shared WSL2 filesystem** - Docker Desktop mounts WSL2 paths natively:
   ```
   WSL2: /home/yao/projects/ml-lora-training/data/
   Container: /home/yao/projects/ml-lora-training/data/ (same path)
   ```
5. **Single daemon** - No confusion, no context switching
6. **Network isolation** - Both containers on same Docker Desktop bridge network
7. **No fragile config** - No manual NVIDIA runtime tweaking needed

**❌ Costs**
1. **Limited CLI from WSL2** - Named pipe (`npipe:////./pipe/...`) only supports `ps`, `images`, `start`, `stop`, `rm`. Cannot `docker logs`, `docker exec`, `docker inspect` from WSL2 bash.
   - **Workaround**: Use Windows PowerShell/CMD for full CLI access, or use Docker Desktop UI
   - **Workaround**: `docker context use desktop-linux && docker compose up` works for compose operations
2. **Docker Desktop required running** - Must have Docker Desktop app open on Windows side
3. **Slightly slower I/O** - Docker Desktop's WSL2 backend adds a layer vs direct WSL2 daemon
4. **Docker Desktop license** - Free for personal/small business use, paid for large organizations
5. **WSL2 Docker Engine removal** - Lose the 21.6GB `ml-training:latest` image in WSL2 (rebuild on Docker Desktop)
6. **VRAM contention** - Both Studio and project containers share the same 32GB GPU. Need to run sequentially, not simultaneously.

#### Verdict: **Recommended** - GPU reliability and working Studio integration outweigh CLI limitations.

---

### OPTION C: Hybrid (Docker Desktop for Studio, Native WSL2 for Scripts)

**Action**: Keep Docker Desktop for Studio only. Run all project scripts natively in WSL2 (no Docker). Uninstall WSL2 Docker Engine.

#### What Changes
| Component | Before | After |
|-----------|--------|-------|
| Studio | Docker Desktop | Docker Desktop (unchanged) |
| Teacher | Docker container | Native Python in WSL2 |
| DPO | Docker container | Native Python in WSL2 |
| SFT | Studio UI | Studio UI (unchanged) |
| Docker in WSL2 | Docker Engine 29.3.1 | Removed |

#### Consequences

**✅ Benefits**
1. **Simplest architecture** - No Docker in WSL2 at all, just Python scripts
2. **Direct GPU access** - PyTorch in WSL2 uses NVIDIA driver directly, no container overhead
3. **Full debugging** - Run scripts with `python3 -m pdb`, no container isolation
4. **Fastest iteration** - Edit code, run script immediately, no rebuild needed
5. **No image management** - No Docker images to build, push, or maintain
6. **Studio untouched** - Docker Desktop runs Studio, nothing else

**❌ Costs**
1. **Conda environment setup** - Need to install full ML stack in WSL2:
   ```bash
   conda activate ai-projects
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   pip install unsloth transformers datasets peft trl accelerate bitsandbytes
   ```
   This is ~15GB of packages in the conda env.
2. **No environment isolation** - Python packages pollute conda env, potential version conflicts
3. **Reproducibility loss** - Docker image guarantees exact versions; conda env drifts over time
4. **Data exchange with Studio** - Manual upload/download via Studio UI for every step:
   - Generate data → Upload `sft_dataset.jsonl` to Studio
   - Studio SFT → Download LoRA adapter to WSL2
   - Run DPO → Upload DPO adapter to Studio Chat
5. **VRAM management** - Need to stop Studio container before running training scripts (32GB shared)
6. **Docker files become obsolete** - `docker/`, `docker-compose.yml` no longer needed

#### Verdict: **Good for development phase** - Fastest iteration, but manual data exchange with Studio is tedious.

---

## 4. Side-by-Side Comparison

| Factor | A: WSL2 Docker Engine | B: Docker Desktop Only | C: Native WSL2 Scripts |
|--------|----------------------|----------------------|----------------------|
| **GPU Reliability** | ❌ Fragile, breaks on updates | ✅ Automatic, proven | ✅ Direct driver access |
| **CLI Access** | ✅ Full (logs, exec, inspect) | ⚠️ Limited (ps, start, stop) | N/A (no Docker) |
| **Studio State** | ❌ Lost, must recreate | ✅ Preserved | ✅ Preserved |
| **Data Exchange** | ✅ Shared volumes | ✅ Shared volumes | ❌ Manual UI upload/download |
| **Setup Complexity** | 🔴 High (fix NVIDIA runtime) | 🟢 Low (uninstall WSL2 Docker) | 🟡 Medium (install packages) |
| **Maintenance** | 🔴 High (WSL2 updates break) | 🟢 Low (Docker Desktop auto-updates) | 🟡 Medium (package drift) |
| **Reproducibility** | ✅ Docker images | ✅ Docker images | ❌ Conda env drift |
| **Iteration Speed** | 🟡 Medium (rebuild images) | 🟡 Medium (rebuild images) | 🟢 Fast (edit and run) |
| **VRAM Management** | ⚠️ Sequential only | ⚠️ Sequential only | ⚠️ Sequential only |
| **Debugging** | ✅ `docker exec` | ❌ No `docker exec` from WSL2 | ✅ Direct Python debug |

---

## 5. Recommendation: Option B (Docker Desktop Only)

### Rationale

1. **GPU access is the critical path** - Without reliable GPU access, nothing works. Docker Desktop is proven to work; WSL2 Docker Engine is fragile.
2. **Integration already worked** - `suspicious_spence` (exit 0) proves the docker-compose works on Docker Desktop.
3. **CLI limitations are manageable** - Docker Desktop UI + Windows PowerShell cover the gaps. For compose operations, `docker compose up` works from WSL2.
4. **Studio state preserved** - No need to recreate `silly_blackwell` or migrate its data.

### Implementation

```bash
# 1. Uninstall WSL2 Docker Engine (preserves images until daemon stopped)
sudo apt remove docker-ce docker-ce-cli docker-compose-plugin docker-buildx-plugin docker-ce-rootless-extras

# 2. Clean up Docker data in WSL2
sudo rm -rf /var/lib/docker /var/run/docker.sock

# 3. Set Docker Desktop as default context
docker context set-default desktop-linux

# 4. Rebuild project image on Docker Desktop
docker compose up --build -d

# 5. Verify GPU access
docker exec ml-training nvidia-smi
```

### VRAM Warning

Both Studio and project containers share 32GB VRAM. **Always run sequentially**:
```bash
# Stop Studio before running project container
docker stop silly_blackwell
docker compose up -d

# After project work, stop and restart Studio
docker compose down
docker start silly_blackwell
```

---

## 7. Workarounds for Docker Desktop CLI Limitations from WSL2

The core problem: Docker Desktop exposes its API via a Windows named pipe
(`npipe:////./pipe/dockerDesktopLinuxEngine`). WSL2 can read basic info through
this pipe, but streaming operations (`docker logs`, `docker exec`, `docker attach`)
fail with "protocol not available."

Here are **four concrete workarounds**, ranked by practicality:

### Workaround 1: Windows PowerShell / CMD (Recommended)

Open a **Windows PowerShell** or **CMD** window and run Docker commands there:
```powershell
# From Windows PowerShell
docker logs ml-training
docker exec -it ml-training bash
docker inspect ml-training
```

**Pros**: Full Docker CLI access, no setup needed, reliable
**Cons**: Must switch to Windows terminal for advanced commands
**Verdict**: Simplest. Use WSL2 for `docker compose up/down/ps`, Windows terminal for `logs/exec/inspect`.

### Workaround 2: socat Named-Pipe-to-Unix-Socket Proxy

Install socat in WSL2 and create a proxy that bridges the Windows named pipe to a Unix socket:

```bash
# Install socat
sudo apt install socat

# Create a Unix socket that proxies to the Docker Desktop named pipe
# (Requires running from Windows side via wsl.exe, or using a Windows proxy tool)
```

**Problem**: WSL2 cannot directly access Windows named pipes from inside Linux. socat
in WSL2 cannot connect to `\\.\pipe\dockerDesktopLinuxEngine`. This workaround
requires a **Windows-side proxy** (e.g., a small Windows service that listens on a TCP
port and forwards to the named pipe).

**Verdict**: Complex to set up, fragile. Not recommended unless you have a Windows-side
proxy already running.

### Workaround 3: Docker Desktop API via HTTP (docker-proxy)

Docker Desktop exposes a remote API. Can use `docker context create` with a TCP endpoint,
but Docker Desktop doesn't expose TCP by default on Windows.

Alternative: Use the **Docker Desktop API directly** with curl from WSL2:
```bash
# Docker Desktop exposes API on the WSL2 integration endpoint
# Check if accessible:
curl -s --unix-socket /var/run/docker.sock http://localhost/containers/json 2>&1
```

This works for the WSL2 Docker Engine daemon, NOT Docker Desktop. For Docker Desktop,
the named pipe is the only endpoint.

**Verdict**: Not feasible without modifying Docker Desktop settings.

### Workaround 4: Docker Compose Logging (Best for This Project)

For our specific use case, `docker compose` handles most operations. The limitation
only affects `docker logs` and `docker exec`. Practical alternatives:

**For logs**: Use Docker Compose's built-in log following:
```bash
# From WSL2 - this WORKS with Docker Desktop context
docker context use desktop-linux
docker compose logs -f training
```

**For exec**: Run commands via `docker compose run`:
```bash
# Instead of: docker exec -it ml-training bash
# Use:
docker compose run --rm training bash
```

**For inspect**: Use compose config and ps:
```bash
docker compose ps
docker compose config
```

**For container management**:
```bash
docker compose up -d       # Start
docker compose down        # Stop
docker compose restart     # Restart
docker compose logs -f     # Follow logs
docker compose ps          # Status
```

**Verdict**: **This is the practical solution for this project.** Docker Compose covers
90% of daily operations. For the remaining 10% (debugging inside a running container),
use Windows PowerShell.

### Summary Table

| Workaround | Setup Effort | Coverage | Reliability |
|------------|-------------|----------|-------------|
| 1. Windows PowerShell | None | 100% | Excellent |
| 2. socat proxy | High | 100% | Fragile |
| 3. Docker API HTTP | High | 70% | N/A (not feasible) |
| 4. Docker Compose | None | 90% | Excellent |

### Recommended Approach: Workaround 1 + 4 Combined

```bash
# WSL2 terminal (90% of operations):
docker context use desktop-linux
docker compose up -d
docker compose logs -f training
docker compose ps
docker compose down

# Windows PowerShell (10% - debugging):
docker exec -it ml-training bash
docker inspect ml-training
docker events
```

---

## 9. Appendix: Technical Details

### WSL2 Docker GPU Failure Root Cause

```
/etc/nvidia-container-runtime/config.toml:
  load-kmods = true     ← WSL2 has no kernel modules → FAILS
  ldconfig = "@/usr/lib/wsl/lib" ← Points to directory, not binary

nvidia-container-runtime expects:
  - Kernel modules (nvidia, nvidia-uvm, nvidia-drm) → WSL2 doesn't have these
  - libnvidia-ml.so.1 in standard path → It's at /usr/lib/wsl/lib/
  - /dev/nvidia* devices → WSL2 provides these via different mechanism

Docker Desktop solves this by:
  - Running its own lightweight Linux VM with proper kernel
  - Mounting NVIDIA libraries from Windows driver
  - Using WSL2 interop for filesystem access
```

### Discovered Container Configurations

#### Studio Container (`silly_blackwell`)
```
Image: unsloth/unsloth
Status: Exited 137 (OOM killed, 8 days ago)
User: unsloth:runtimeusers
WorkingDir: /workspace
Entrypoint: /usr/local/bin/entrypoint.sh
Exposed Ports: 22/tcp, 8000/tcp, 8888/tcp
Port Bindings: 2222->22, 8000->8000, 8888->8888
Mounts:
  - ~/.cache/huggingface → /root/.cache/huggingface
  - ~/workspace/unsloth/work → /workspace/work
Environment:
  - JUPYTER_PASSWORD=mypassword
  - HF_HOME=/workspace/.cache/huggingface
  - UNSLOTH_DOCKER=1
Network: bridge
```

#### Project Container (`suspicious_spence`)
```
Image: ml-lora-training-training:latest
Status: Exited 0 (clean exit, 9 days ago)
Mounts: NONE (no volume mounts configured)
Port Bindings: NONE
Network: bridge
Runtime: runc (NOT nvidia - no GPU reservation)
ShmSize: 64MB (NOT 16GB as configured in docker-compose.yml)
Logs: Only ran nvidia-smi health check, then exited
```

**Note**: `suspicious_spence` had no volume mounts and no GPU reservation, meaning
the previous docker-compose run on Docker Desktop did not use our current compose
configuration. The container was likely created with a different compose file or
`docker run` command.

### WSL2 Docker Engine Images (to be removed)

| Image | Size |
|-------|------|
| `ml-training:latest` | 21.6GB |
| `docker-chatterbox-tts:latest` | 24.3GB |
| `docker-frontend:latest` | 49.4MB |
| `nvidia/cuda:12.6.0-devel-ubuntu22.04` | 7.17GB |

### CLI Bridge: `ddk` Script

Created at `scripts/ddk` - a wrapper that calls Docker Desktop via Windows
PowerShell from WSL2, bypassing the named pipe limitation:

```bash
#!/bin/bash
exec /mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe docker "$@"
```

Added to PATH via `~/.bashrc`. Usage:
```bash
ddk ps -a                    # List all Docker Desktop containers
ddk logs ml-training         # Stream logs (works!)
ddk inspect silly_blackwell  # Full JSON inspect (works!)
ddk exec -it ml-training bash # Interactive shell (works when container running)
```

**Verified working commands**: `ps`, `logs`, `logs --tail`, `inspect` (JSON), `exec`.
**Known limitation**: `docker inspect --format '{{...}}'` - Go template syntax
conflicts with PowerShell parsing. Use full JSON output and pipe to `python3 -m json.tool` instead.
