# Activate the local uv Python 3.12 env, then verify TVM:
#   source ./env.sh
#   python relax_basic.py
#
# Optional: use a local TVM checkout instead of the pip wheel:
#   export TVM_HOME=/path/to/your/tvm
#   source ./env.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/.venv/bin/activate"
fi

if [[ -n "${TVM_HOME:-}" ]]; then
  export PYTHONPATH="${TVM_HOME}/python:${TVM_HOME}/.local/python${PYTHONPATH:+:${PYTHONPATH}}"
  # Linux shared libs from a source build
  export LD_LIBRARY_PATH="${TVM_HOME}/build/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  # macOS
  export DYLD_LIBRARY_PATH="${TVM_HOME}/build/lib${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}"
fi

python -c "import tvm; from tvm import relax; print('tvm', getattr(tvm, '__version__', '?'), 'from', tvm.__file__)"
