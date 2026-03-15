#!/bin/sh
set -e

PACKAGE="openbrowser-ai"
MIN_PYTHON_MAJOR=3
MIN_PYTHON_MINOR=12

# --- Colors (disabled if not a terminal) ---
if [ -t 1 ]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  BOLD='\033[1m'
  NC='\033[0m'
else
  RED=''
  GREEN=''
  YELLOW=''
  BOLD=''
  NC=''
fi

info()  { printf "${GREEN}%s${NC}\n" "$1"; }
warn()  { printf "${YELLOW}%s${NC}\n" "$1"; }
error() { printf "${RED}%s${NC}\n" "$1"; }
bold()  { printf "${BOLD}%s${NC}\n" "$1"; }

# --- Parse args ---
LOCAL_INSTALL=false
SKIP_BROWSER=false
for arg in "$@"; do
  case "$arg" in
    --local) LOCAL_INSTALL=true ;;
    --no-browser) SKIP_BROWSER=true ;;
    --help|-h)
      echo "Usage: install.sh [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --local        Install to ~/.local/bin (no sudo required)"
      echo "  --no-browser   Skip Chromium installation"
      echo "  -h, --help     Show this help message"
      exit 0
      ;;
  esac
done

# --- Detect OS ---
OS="$(uname -s)"
case "$OS" in
  Linux*)                    OS_NAME="Linux" ;;
  Darwin*)                   OS_NAME="macOS" ;;
  MINGW*|MSYS*|CYGWIN*)     OS_NAME="Windows" ;;
  *)
    error "Unsupported OS: $OS"
    echo "OpenBrowser supports macOS, Linux, and Windows."
    exit 1
    ;;
esac

# --- Find Python 3.12+ ---
PYTHON=""
find_python() {
  # On Windows (Git Bash), try the py launcher first
  if [ "$OS_NAME" = "Windows" ] && command -v py >/dev/null 2>&1; then
    for pyver in "-3" "-3.13" "-3.12"; do
      version=$(py "$pyver" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null) || continue
      major=$(echo "$version" | cut -d. -f1)
      minor=$(echo "$version" | cut -d. -f2)
      if [ "$major" -gt "$MIN_PYTHON_MAJOR" ] || { [ "$major" -eq "$MIN_PYTHON_MAJOR" ] && [ "$minor" -ge "$MIN_PYTHON_MINOR" ]; }; then
        PYTHON=$(py "$pyver" -c "import sys; print(sys.executable)" 2>/dev/null)
        return 0
      fi
    done
  fi
  for cmd in python3.13 python3.12 python3 python; do
    if command -v "$cmd" >/dev/null 2>&1; then
      version=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null) || continue
      major=$(echo "$version" | cut -d. -f1)
      minor=$(echo "$version" | cut -d. -f2)
      if [ "$major" -gt "$MIN_PYTHON_MAJOR" ] || { [ "$major" -eq "$MIN_PYTHON_MAJOR" ] && [ "$minor" -ge "$MIN_PYTHON_MINOR" ]; }; then
        PYTHON="$cmd"
        return 0
      fi
    fi
  done
  return 1
}

# --- Install methods (in preference order) ---
install_with_uv() {
  command -v uv >/dev/null 2>&1 || return 1
  info "Installing with uv..."
  uv tool install "$PACKAGE"
}

install_with_pipx() {
  command -v pipx >/dev/null 2>&1 || return 1
  info "Installing with pipx..."
  pipx install --python "$PYTHON" "$PACKAGE"
}

install_with_pip() {
  [ -n "$PYTHON" ] || return 1
  # Use the discovered Python's pip module to ensure version match
  info "Installing with $PYTHON -m pip..."
  if [ "$LOCAL_INSTALL" = true ]; then
    "$PYTHON" -m pip install --user "$PACKAGE"
    warn "Installed to ~/.local/bin -- make sure it is in your PATH"
  else
    "$PYTHON" -m pip install "$PACKAGE"
  fi
}

# --- Main ---
bold "OpenBrowser Installer"
echo "====================="
echo ""
echo "  OS:      $OS_NAME ($(uname -m))"

if ! find_python; then
  error "Python ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}+ is required but not found."
  echo ""
  echo "Install Python:"
  if [ "$OS_NAME" = "macOS" ]; then
    echo "  brew install python@3.12"
  elif [ "$OS_NAME" = "Windows" ]; then
    echo "  winget install Python.Python.3.12"
  else
    echo "  sudo apt install python3.12   # Debian/Ubuntu"
    echo "  sudo dnf install python3.12   # Fedora"
  fi
  echo "  https://www.python.org/downloads/"
  exit 1
fi

echo "  Python:  $PYTHON ($("$PYTHON" --version 2>&1))"
echo ""

if install_with_uv; then
  INSTALLER="uv"
elif install_with_pipx; then
  INSTALLER="pipx"
elif install_with_pip; then
  INSTALLER="pip"
else
  error "No Python package manager found."
  echo ""
  echo "Install one of: uv, pipx, or pip"
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
  echo "  brew install pipx && pipx ensurepath"
  exit 1
fi

# --- Install Chromium ---
if [ "$SKIP_BROWSER" = false ]; then
  echo ""
  info "Installing Chromium browser..."
  if command -v openbrowser-ai >/dev/null 2>&1; then
    openbrowser-ai install 2>/dev/null || warn "Chromium install failed (run 'openbrowser-ai install' manually)"
  elif command -v uvx >/dev/null 2>&1; then
    uvx playwright install chromium 2>/dev/null || warn "Chromium install failed (run 'openbrowser-ai install' manually)"
  else
    warn "Chromium install skipped. Please run 'openbrowser-ai install' manually after installation completes."
  fi
fi

# --- Done ---
echo ""
info "OpenBrowser installed successfully! (via $INSTALLER)"
echo ""
echo "  Get started:"
echo "    openbrowser-ai --help"
echo "    openbrowser-ai -c \"await navigate('https://example.com')\""
echo ""
echo "  Docs: https://docs.openbrowser.me"
echo ""
