#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/install.sh [--pi] [--headless] [--tflite]

Options:
  --pi        Assume Raspberry Pi and prefer headless OpenCV
  --headless  Install opencv-python-headless instead of opencv-python
  --tflite    Install tflite-runtime
USAGE
}

HEADLESS=0
TFLITE=0

for arg in "$@"; do
  case "$arg" in
    --pi)
      HEADLESS=1
      ;;
    --headless)
      HEADLESS=1
      ;;
    --tflite)
      TFLITE=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $arg"
      usage
      exit 1
      ;;
  esac
  shift || true
 done

OS_NAME="$(uname -s)"
PYTHON_BIN=""

ensure_python() {
  if command -v python3 >/dev/null 2>&1; then
    if python3 - <<'PY' >/dev/null 2>&1
import sys
sys.exit(0 if sys.version_info >= (3,10) else 1)
PY
    then
      PYTHON_BIN="python3"
      return 0
    fi
  fi
  return 1
}

install_python_linux() {
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y python3.10 python3.10-venv python3.10-distutils
    PYTHON_BIN="python3.10"
    return 0
  fi
  return 1
}

install_python_macos() {
  if command -v brew >/dev/null 2>&1; then
    brew install python@3.11
    PYTHON_BIN="python3"
    return 0
  fi
  return 1
}

if ! ensure_python; then
  case "$OS_NAME" in
    Linux)
      if ! install_python_linux; then
        echo "Could not install Python automatically. Please install Python 3.10+ and rerun." >&2
        exit 1
      fi
      ;;
    Darwin)
      if ! install_python_macos; then
        echo "Homebrew not found. Install Python 3.10+ and rerun." >&2
        exit 1
      fi
      ;;
    *)
      echo "Unsupported OS: $OS_NAME. Install Python 3.10+ manually." >&2
      exit 1
      ;;
  esac
fi

if [ -z "$PYTHON_BIN" ]; then
  PYTHON_BIN="python3"
fi

if [ ! -d .venv ]; then
  "$PYTHON_BIN" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

if [ "$HEADLESS" -eq 1 ]; then
  python -m pip uninstall -y opencv-python || true
  python -m pip install opencv-python-headless
fi

if [ "$TFLITE" -eq 1 ]; then
  python -m pip install tflite-runtime
fi

echo "Install complete. Activate with: source .venv/bin/activate"
