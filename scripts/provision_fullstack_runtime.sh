#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
HOST_TAG="$(hostname 2>/dev/null | tr -c 'A-Za-z0-9_.-' '_' || echo host)"
OUT_ROOT="${OUT_ROOT:-/tmp/scheduleurm-fullstack-runtime-${HOST_TAG}-${RUN_ID}}"
KIND_VERSION="${KIND_VERSION:-v0.32.0}"
NVIDIA_CUDA_IMAGE="${NVIDIA_CUDA_IMAGE:-nvidia/cuda:12.6.2-base-ubuntu22.04}"
DOCKER_CHANNEL="${DOCKER_CHANNEL:-official}"

mkdir -p "${OUT_ROOT}"
exec > >(tee -a "${OUT_ROOT}/${MODE}.log") 2>&1

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

need_root() {
  if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: mode ${MODE} must run as root" >&2
    exit 2
  fi
}

record_file() {
  local path="$1"
  local out="$2"
  if [ -e "${path}" ]; then
    {
      echo "### ${path}"
      ls -ld "${path}" || true
      if [ -L "${path}" ]; then
        readlink -f "${path}" || true
      fi
    } >> "${out}"
  fi
}

snapshot_cuda_state() {
  local tag="$1"
  local out="${OUT_ROOT}/cuda_cudnn_state_${tag}.txt"
  : > "${out}"
  log "Recording CUDA/cuDNN/NVIDIA package state: ${out}"
  {
    echo "### identity"
    date -u
    hostname
    uname -a
    echo
    echo "### nvidia-smi"
    command -v nvidia-smi || true
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true
    echo
    echo "### dpkg cuda/cudnn/nvidia/docker/kube/go"
    dpkg-query -W -f='${db:Status-Abbrev} ${binary:Package} ${Version}\n' 2>/dev/null \
      | grep -Ei 'cuda|cudnn|nvidia|docker|containerd|kubectl|kube|golang|(^..  go$)' \
      | sort || true
    echo
    echo "### apt holds"
    apt-mark showhold 2>/dev/null | sort || true
  } >> "${out}"

  for p in \
    /usr/local/cuda \
    /usr/local/cuda-12 \
    /usr/local/cuda-12.6 \
    /usr/include/cudnn.h \
    /usr/include/cudnn_version.h \
    /usr/lib/x86_64-linux-gnu/libcudnn.so \
    /usr/lib/x86_64-linux-gnu/libcudnn.so.9
  do
    record_file "${p}" "${out}"
  done
}

preflight() {
  log "Preflight start; output root ${OUT_ROOT}"
  snapshot_cuda_state "pre"
  {
    echo "### os"
    cat /etc/os-release 2>/dev/null || true
    echo
    echo "### tools"
    for x in sudo apt-get curl wget gpg install docker dockerd containerd kubectl kind go helm nvidia-smi systemctl service; do
      if command -v "${x}" >/dev/null 2>&1; then
        echo "${x}=$(command -v "${x}")"
      else
        echo "${x}=MISSING"
      fi
    done
    echo
    echo "### versions"
    docker --version 2>&1 || true
    kubectl version --client=true 2>&1 || true
    kind version 2>&1 || true
    go version 2>&1 || true
    echo
    echo "### apt sources"
    find /etc/apt/sources.list /etc/apt/sources.list.d -maxdepth 1 -type f -print -exec sed -n '1,120p' {} \; 2>/dev/null || true
    echo
    echo "### disk"
    df -h /
  } > "${OUT_ROOT}/preflight_tools.txt"
  log "Preflight done"
}

reject_cuda_mutations_from_apt_sim() {
  local sim_file="$1"
  local bad_file="${OUT_ROOT}/apt_cuda_mutation_blockers.txt"
  grep -E '^(Inst|Remv|Conf) ' "${sim_file}" \
    | grep -Ei 'cuda|cudnn|nvidia-driver|nvidia-dkms|nvidia-kernel|nvidia-utils|nvidia-headless|xserver-xorg-video-nvidia|libnvidia-(gl|compute|encode|decode|extra)' \
    | grep -Evi 'libnvidia-container|nvidia-container-toolkit' \
    > "${bad_file}" || true
  if [ -s "${bad_file}" ]; then
    log "ABORT: apt simulation would touch CUDA/cuDNN/NVIDIA driver packages"
    cat "${bad_file}"
    exit 3
  fi
}

apt_simulate_and_install() {
  local label="$1"
  shift
  local sim="${OUT_ROOT}/apt_sim_${label}.txt"
  log "APT dry-run ${label}: $*"
  apt-get -s install -y "$@" > "${sim}"
  reject_cuda_mutations_from_apt_sim "${sim}"
  log "APT install ${label}: $*"
  DEBIAN_FRONTEND=noninteractive apt-get install -y "$@"
}

configure_docker_repo() {
  if [ -f /etc/apt/keyrings/docker.asc ] && grep -R "download.docker.com" /etc/apt/sources.list.d >/dev/null 2>&1; then
    log "Docker apt repository already configured"
    return
  fi
  log "Configuring Docker apt repository"
  apt-get update
  apt_simulate_and_install docker_prereqs ca-certificates curl gnupg
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc
  . /etc/os-release
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update
}

install_docker() {
  if [ "${DOCKER_CHANNEL}" = "ubuntu" ]; then
    log "Installing Docker from Ubuntu repository"
    apt-get update
    apt_simulate_and_install docker_ubuntu docker.io containerd
  else
    configure_docker_repo
    apt_simulate_and_install docker_engine docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  fi
  if command -v systemctl >/dev/null 2>&1 && systemctl list-unit-files docker.service >/dev/null 2>&1; then
    systemctl enable --now docker || systemctl restart docker || true
  else
    service docker start || true
  fi
}

install_kubectl() {
  if command -v kubectl >/dev/null 2>&1; then
    log "kubectl already installed: $(command -v kubectl)"
    return
  fi
  log "Installing kubectl binary from Kubernetes release channel"
  local tmp
  tmp="$(mktemp -d)"
  local stable
  stable="$(curl -L -s https://dl.k8s.io/release/stable.txt)"
  curl -L -o "${tmp}/kubectl" "https://dl.k8s.io/release/${stable}/bin/linux/amd64/kubectl"
  curl -L -o "${tmp}/kubectl.sha256" "https://dl.k8s.io/release/${stable}/bin/linux/amd64/kubectl.sha256"
  (cd "${tmp}" && echo "$(cat kubectl.sha256)  kubectl" | sha256sum --check)
  install -m 0755 "${tmp}/kubectl" /usr/local/bin/kubectl
  rm -rf "${tmp}"
}

install_kind() {
  if command -v kind >/dev/null 2>&1; then
    log "kind already installed: $(command -v kind)"
    return
  fi
  log "Installing kind ${KIND_VERSION}"
  local tmp
  tmp="$(mktemp -d)"
  curl -L -o "${tmp}/kind" "https://kind.sigs.k8s.io/dl/${KIND_VERSION}/kind-linux-amd64"
  install -m 0755 "${tmp}/kind" /usr/local/bin/kind
  rm -rf "${tmp}"
}

install_go() {
  if command -v go >/dev/null 2>&1; then
    log "go already installed: $(command -v go)"
    return
  fi
  apt-get update
  apt_simulate_and_install golang golang-go
}

configure_nvidia_container_repo() {
  if [ -f /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg ] \
    && grep -R "nvidia.github.io/libnvidia-container" /etc/apt/sources.list.d >/dev/null 2>&1; then
    log "NVIDIA Container Toolkit apt repository already configured"
    return
  fi
  log "Configuring NVIDIA Container Toolkit apt repository"
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update
}

install_nvidia_container_toolkit() {
  configure_nvidia_container_repo
  apt_simulate_and_install nvidia_container_toolkit nvidia-container-toolkit
  if command -v nvidia-ctk >/dev/null 2>&1 && command -v docker >/dev/null 2>&1; then
    log "Configuring Docker NVIDIA runtime"
    nvidia-ctk runtime configure --runtime=docker
    if command -v systemctl >/dev/null 2>&1 && systemctl list-unit-files docker.service >/dev/null 2>&1; then
      systemctl restart docker || service docker restart || true
    else
      service docker restart || true
    fi
  fi
}

install_core() {
  need_root
  preflight
  snapshot_cuda_state "before_install"
  install_docker
  install_kubectl
  install_kind
  install_go
  install_nvidia_container_toolkit
  snapshot_cuda_state "after_install"
  log "Install core done"
}

verify() {
  log "Verify start"
  {
    echo "### versions"
    docker --version 2>&1 || true
    docker info 2>&1 | sed -n '1,120p' || true
    kubectl version --client=true 2>&1 || true
    kind version 2>&1 || true
    go version 2>&1 || true
    nvidia-ctk --version 2>&1 || true
    echo
    echo "### docker hello-world"
    docker run --rm hello-world 2>&1 || true
    echo
    echo "### docker gpu nvidia-smi"
    docker run --rm --gpus all "${NVIDIA_CUDA_IMAGE}" nvidia-smi 2>&1 || true
  } | tee "${OUT_ROOT}/verify_runtime.txt"
  snapshot_cuda_state "post_verify"
  log "Verify done"
}

case "${MODE}" in
  preflight)
    preflight
    ;;
  install-core)
    install_core
    ;;
  verify)
    verify
    ;;
  *)
    echo "Usage: $0 {preflight|install-core|verify}" >&2
    exit 2
    ;;
esac

echo "${OUT_ROOT}"
