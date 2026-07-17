# E2E VM Setup (Mac mini Apple Silicon)

Run E2E **inside each VM**. No SSH orchestration from Mac host.

## Model

```
Mac mini (UTM)
 ├── macOS native     -> ./run-all-e2e
 ├── Ubuntu ARM VM    -> git pull && ./scripts/e2e/run-ubuntu-e2e.sh
 ├── Arch ARM VM      -> git pull && ./scripts/e2e/run-arch-e2e.sh
 └── Windows ARM VM   -> git pull; .\scripts\e2e\run-windows-e2e.ps1
```

You create VMs once, install dev deps once, clone repo once. Daily loop = pull + run script.

---

## Install UTM (Mac host, once)

```bash
brew install --cask utm
```

Create each VM in UTM GUI (QEMU backend recommended). Use **ARM64** ISOs on Apple Silicon.

| VM | Suggested name | ISO |
|----|----------------|-----|
| Ubuntu | `dronmakr-ubuntu-e2e` | [Ubuntu Desktop 24.04 LTS ARM64](https://ubuntu.com/download/desktop) |
| Arch | `dronmakr-arch-e2e` | [Arch Linux ARM64](https://archlinuxarm.org/) or official aarch64 ISO |
| Windows | `dronmakr-windows-e2e` | [Windows 11 ARM64](https://www.microsoft.com/software-download/windows11) |

Resources per VM: 4 CPU, 8 GB RAM, 60+ GB disk.

Take a **snapshot** after one-time setup below. Revert snapshot when guest gets dirty.

---

## Shared requirements (all guests)

Inside each VM:

1. **Git** + clone:

   ```bash
   git clone https://github.com/nafeu/dronmakr.git ~/dronmakr
   cd ~/dronmakr
   ```

2. **Surge XT** VST3 installed (tests fail immediately if missing)

3. **Python 3.10+**, **Node 20+**, **Rust stable** (needed for `npm run build`)

4. Repo bootstrap:

   ```bash
   python3 -m venv venv
   source venv/bin/activate          # Windows: .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   npm ci
   npx playwright install chromium --with-deps   # Linux
   npx playwright install chromium               # Windows / macOS
   ```

5. **Fast dev loop** (no Tauri package build):

   ```bash
   E2E_BACKEND_ONLY=1 ./scripts/e2e/run-ubuntu-e2e.sh   # or arch / windows ps1
   ```

6. **Full packaged-app loop** (build inside same VM — required on ARM; CI tarballs are x64):

   ```bash
   npm run build
   ./scripts/e2e/run-ubuntu-e2e.sh
   ```

---

## Ubuntu ARM VM (one-time)

```bash
sudo apt update
sudo apt install -y build-essential curl git python3 python3-venv python3-pip \
  libsndfile1 libwebkit2gtk-4.1-0 patchelf pkg-config xvfb zenity

# Node 20
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

# Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Surge XT — install Linux ARM64 VST3 under ~/.vst3/
mkdir -p ~/.vst3
# copy or install Surge XT.vst3 bundle here

# dronmakr (see shared bootstrap above)
git clone https://github.com/nafeu/dronmakr.git ~/dronmakr
cd ~/dronmakr
# ... venv, pip, npm, playwright ...
```

**Daily:**

```bash
cd ~/dronmakr
git pull
./scripts/e2e/run-ubuntu-e2e.sh
```

---

## Arch ARM VM (one-time)

```bash
sudo pacman -Syu --needed base-devel git curl python python-pip \
  libsndfile webkit2gtk-4.1 patchelf pkgconf xvfb run zenity nodejs npm

# Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Surge XT under ~/.vst3/
mkdir -p ~/.vst3

git clone https://github.com/nafeu/dronmakr.git ~/dronmakr
cd ~/dronmakr
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
npm ci
npx playwright install chromium --with-deps
```

**Daily:**

```bash
cd ~/dronmakr
git pull
./scripts/e2e/run-arch-e2e.sh
```

---

## Windows ARM VM (one-time)

In PowerShell (admin for OpenSSH optional):

```powershell
# Git, Python 3.10+, Node 20 — via winget or installers
winget install Git.Git Python.Python.3.10 OpenJS.NodeJS.LTS

# Rust: https://rustup.rs/

# Surge XT VST3 — typical path:
# $env:LOCALAPPDATA\Programs\Common\VST3\

git clone https://github.com/nafeu/dronmakr.git $HOME\dronmakr
cd $HOME\dronmakr
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
npm ci
npx playwright install chromium
```

**Daily:**

```powershell
cd $HOME\dronmakr
git pull
.\scripts\e2e\run-windows-e2e.ps1
```

---

## macOS native (no VM)

On Mac host directly:

```bash
cd ~/Development/github/dronmakr
git pull
./run-all-e2e --backend-only          # fast
./run-all-e2e --build                 # packaged .app
```

---

## Env knobs

| Variable | Purpose |
|----------|---------|
| `E2E_BACKEND_ONLY=1` | Sidecar only — skip Tauri package |
| `DRONMAKR_TEST_PLUGIN_PATHS` | Override VST scan dirs |
| `E2E_APP_BINARY` | Explicit packaged binary path |
| `E2E_ARTIFACTS_DIR` | Playwright report/screenshots output |

---

## Scripts reference

| OS | Script |
|----|--------|
| macOS | `./run-all-e2e` or `scripts/e2e/run-macos-e2e.sh` |
| Ubuntu | `scripts/e2e/run-ubuntu-e2e.sh` |
| Arch | `scripts/e2e/run-arch-e2e.sh` |
| Linux (auto) | `scripts/e2e/run-linux-e2e.sh` |
| Windows | `scripts/e2e/run-windows-e2e.ps1` |

See also: [testing-plan.md](testing-plan.md)
