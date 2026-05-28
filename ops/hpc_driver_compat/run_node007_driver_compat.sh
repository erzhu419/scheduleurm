#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ROOT}/logs"
mkdir -p "${LOG_DIR}"

echo "== identity =="
hostname
date -Is
uname -r

echo "== driver =="
export PATH="/cm/local/apps/cuda-driver/libs/current/bin:${PATH}"
export LD_LIBRARY_PATH="/cm/local/apps/cuda-driver/libs/current/lib64:${LD_LIBRARY_PATH:-}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
ldconfig -p | egrep 'libcuda|libnvidia-ml' || true

echo "== cuda/cudnn module smoke =="
module purge >/dev/null 2>&1 || true
module load cuda10.1/toolkit/10.1.243
module load cudnn7.6-cuda10.1/7.6.5.32
CUDNN_ROOT="/cm/shared/apps/cudnn7.6-cuda10.1/7.6.5.32"
nvcc -std=c++11 \
  -I"${CUDNN_ROOT}/include" \
  -L"${CUDNN_ROOT}/lib64" \
  -Xlinker -rpath -Xlinker "${CUDNN_ROOT}/lib64" \
  "${ROOT}/cuda_cudnn_smoke.cu" \
  -lcudnn \
  -o "${LOG_DIR}/cuda_cudnn_smoke"
"${LOG_DIR}/cuda_cudnn_smoke" 2>&1 | tee "${LOG_DIR}/cuda_cudnn_smoke.log"

echo "== matlab gpu smoke =="
module purge >/dev/null 2>&1 || true
module load matlab/2018a
MATLAB_STATUS=OK
set +e
timeout 240 matlab -nodisplay -nosplash -singleCompThread \
  -r "run('${ROOT}/matlab_gpu_smoke.m')" \
  > "${LOG_DIR}/matlab_gpu_smoke.log" 2>&1
status=$?
set -e
if [[ "${status}" -ne 0 ]]; then
  tail -n 120 "${LOG_DIR}/matlab_gpu_smoke.log" || true
  if grep -Eq 'License checkout failed|License Manager Error -15' "${LOG_DIR}/matlab_gpu_smoke.log"; then
    MATLAB_STATUS=LICENSE_BLOCKED
    echo "MATLAB_GPU_SMOKE=LICENSE_BLOCKED"
  else
    exit "${status}"
  fi
fi
tail -n 80 "${LOG_DIR}/matlab_gpu_smoke.log"

echo "== fluent gpu smoke =="
module purge >/dev/null 2>&1 || true
module load ansys/fluent
export FLUENT_ARCH=lnamd64
export LD_LIBRARY_PATH="/cm/local/apps/cuda-driver/libs/current/lib64:${LD_LIBRARY_PATH:-}"
timeout 240 fluent 3ddp -g -driver null -t1 -gpgpu=1 -i "${ROOT}/fluent_gpu_smoke.jou" \
  > "${LOG_DIR}/fluent_gpu_smoke.log" 2>&1 || {
    status=$?
    tail -n 160 "${LOG_DIR}/fluent_gpu_smoke.log" || true
    exit "${status}"
  }
egrep -i 'CUDA visible GPUs|CUDA runtime version|Driver version|Number of GPUs|NVIDIA|license enables|Error|Fail' "${LOG_DIR}/fluent_gpu_smoke.log" || true

if [[ "${MATLAB_STATUS}" == "OK" ]]; then
  echo "ALL_DRIVER_COMPAT_SMOKES_OK"
else
  echo "DRIVER_COMPAT_SMOKES_OK_EXCEPT_MATLAB_LICENSE_BLOCKED"
fi
