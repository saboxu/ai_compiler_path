# Source before running Relax examples in this folder:
#   source ./env.sh
#   python relax_fuse_ops_by_pattern.py
#
# Points at the local Apache TVM checkout + matching tvm_ffi.

TVM_HOME="${TVM_HOME:-/Users/saboxu/Downloads/codes/tvm}"

export PYTHONPATH="${TVM_HOME}/python:${TVM_HOME}/.local/python${PYTHONPATH:+:${PYTHONPATH}}"

# Prefer TVM-built shared libs over a mismatched conda tvm_ffi.
export DYLD_LIBRARY_PATH="${TVM_HOME}/build/lib${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}"

python -c "import tvm; from tvm import relax; print('tvm', getattr(tvm, '__version__', '?'), 'from', tvm.__file__)"
