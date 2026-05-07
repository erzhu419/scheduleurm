#!/usr/bin/env bash
# install_slurm_node.sh — install slurm + munge on a single node from source.
#
# Runs on the target node (locally OR ssh'd onto remote). Handles:
#   1. Detect existing slurm install → exit early
#   2. Install build deps via apt (Ubuntu / Debian) or yum (RHEL family)
#   3. Acquire slurm source: from --source-dir (if rsync'd) OR `git clone --depth 1 -b TAG`
#   4. Build slurm: ./configure --prefix=PREFIX --sysconfdir=/etc/slurm; make -j; make install
#   5. Set up munge: apt install munge + generate /etc/munge/munge.key if missing
#   6. Create slurm user (uid 64030 to match Debian convention if available)
#   7. Write a sensible default slurm.conf + gres.conf based on probed hardware
#   8. Install systemd units + start slurmctld + slurmd
#
# Exit codes:
#   0 = success (slurm running, sinfo works)
#   2 = already installed (no action; caller should treat as success)
#   3 = source acquisition failed (no --source-dir AND github unreachable)
#   4 = build failed
#   5 = post-install (munge / config / systemd) failed
#
# Sudo: requires passwordless sudo OR --sudo-pass <pwd> (read from stdin if --sudo-pass=-).
#
# Usage:
#   install_slurm_node.sh --tag slurm-23.11.10-1                   # clone github, build
#   install_slurm_node.sh --source-dir /tmp/scheduleurm/slurm-src  # use pre-rsync'd source
#   install_slurm_node.sh --tag XX --sudo-pass=- <<< "PASSWORD"

set -eo pipefail

# --- args --------------------------------------------------------------------
SLURM_TAG="slurm-23.11.10-1"   # known-good LTS; matches jtl110gpu2's apt 23.11.4 closely
SOURCE_DIR=""
PREFIX="/usr/local"
SUDO_PASS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag)        SLURM_TAG="$2"; shift 2 ;;
        --source-dir) SOURCE_DIR="$2"; shift 2 ;;
        --prefix)     PREFIX="$2"; shift 2 ;;
        --sudo-pass=-) SUDO_PASS="$(cat)"; shift ;;
        --sudo-pass)  SUDO_PASS="$2"; shift 2 ;;
        -h|--help)    sed -n '2,30p' "$0" | sed 's/^# \?//' ; exit 0 ;;
        *) echo "unknown arg: $1" >&2 ; exit 64 ;;
    esac
done

# --- sudo wrapper ------------------------------------------------------------
SUDO() {
    if [[ -n "$SUDO_PASS" ]]; then
        echo "$SUDO_PASS" | sudo -S -p '' "$@"
    else
        sudo -n "$@" 2>/dev/null || sudo "$@"
    fi
}

log()   { echo "[install_slurm_node] $*" >&2 ; }
fatal() { log "FATAL: $*" ; exit "${2:-1}" ; }

# --- 1. detect existing install ---------------------------------------------
if command -v sbatch >/dev/null 2>&1 && command -v squeue >/dev/null 2>&1; then
    log "slurm already installed at: $(command -v sbatch)"
    log "version: $(sbatch --version 2>/dev/null | head -1 || echo unknown)"
    exit 2
fi

# --- 2. install build deps ---------------------------------------------------
log "installing build deps via apt"
if command -v apt-get >/dev/null 2>&1; then
    SUDO apt-get update -qq 2>&1 | tail -3 || true
    # core build chain
    DEPS=(build-essential autoconf libtool pkg-config make gcc git)
    # munge
    DEPS+=(libmunge-dev libmunge2 munge)
    # slurm runtime libs
    DEPS+=(libssl-dev libnl-3-dev libnl-route-3-dev libpam0g-dev libreadline-dev)
    # python3 dev (for slurm's PMIx + python bindings if needed; safer to have)
    DEPS+=(python3-dev)
    # systemd dev (for unit installation hooks)
    DEPS+=(libsystemd-dev) || true
    SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${DEPS[@]}" 2>&1 | tail -5 \
        || fatal "apt install of build deps failed" 4
else
    fatal "non-apt OS not supported by this installer; install slurm + munge manually" 4
fi

# --- 3. acquire source -------------------------------------------------------
WORK=/tmp/scheduleurm-slurm-build-$$
mkdir -p "$WORK"
cd "$WORK"

if [[ -n "$SOURCE_DIR" ]]; then
    log "using pre-staged source from $SOURCE_DIR"
    if [[ ! -d "$SOURCE_DIR" ]]; then
        fatal "source dir $SOURCE_DIR does not exist" 3
    fi
    # The source dir may be the slurm-src/ root (containing configure or autogen.sh)
    # OR a parent containing slurm-src/. Try both.
    if [[ -f "$SOURCE_DIR/configure" || -f "$SOURCE_DIR/configure.ac" ]]; then
        SRC="$SOURCE_DIR"
    elif [[ -d "$SOURCE_DIR/slurm-src" && -f "$SOURCE_DIR/slurm-src/configure" ]]; then
        SRC="$SOURCE_DIR/slurm-src"
    elif [[ -d "$SOURCE_DIR/slurm-src" && -f "$SOURCE_DIR/slurm-src/configure.ac" ]]; then
        SRC="$SOURCE_DIR/slurm-src"
    else
        fatal "couldn't find configure or configure.ac under $SOURCE_DIR" 3
    fi
else
    log "cloning $SLURM_TAG from https://github.com/SchedMD/slurm.git"
    if ! timeout 300 git clone --depth 1 --branch "$SLURM_TAG" \
            https://github.com/SchedMD/slurm.git slurm-src 2>&1 | tail -5; then
        fatal "git clone failed (likely github unreachable from this node); rerun with --source-dir" 3
    fi
    SRC="$WORK/slurm-src"
fi

# --- 4. build ----------------------------------------------------------------
cd "$SRC"
log "configuring slurm at $SRC (prefix=$PREFIX)"
if [[ -f autogen.sh && ! -f configure ]]; then
    ./autogen.sh 2>&1 | tail -3
fi
./configure --prefix="$PREFIX" --sysconfdir=/etc/slurm \
    --runstatedir=/run --enable-pam --without-rpath \
    2>&1 | tail -10 || fatal "configure failed" 4

JOBS=$(nproc 2>/dev/null || echo 4)
log "compiling with -j$JOBS (this typically takes 10-20 min)"
make -j"$JOBS" 2>&1 | tail -8 || fatal "make failed" 4

log "installing to $PREFIX"
SUDO make install 2>&1 | tail -5 || fatal "make install failed" 4

# Make sure the new binaries are on PATH for subsequent steps
export PATH="$PREFIX/bin:$PREFIX/sbin:$PATH"
hash -r

# --- 5. munge setup ----------------------------------------------------------
log "configuring munge"
# Generate keyfile if missing (single-node default; for multi-node user must sync)
if [[ ! -f /etc/munge/munge.key ]]; then
    SUDO bash -c "dd if=/dev/urandom bs=1 count=1024 of=/etc/munge/munge.key 2>/dev/null && \
                  chown munge:munge /etc/munge/munge.key && chmod 0400 /etc/munge/munge.key"
fi
SUDO systemctl enable --now munge 2>&1 | tail -3 || true

# --- 6. slurm user + runtime dirs -------------------------------------------
if ! id slurm >/dev/null 2>&1; then
    log "creating slurm user"
    SUDO useradd -r -M -s /usr/sbin/nologin -d /var/lib/slurm slurm \
        2>&1 | tail -2 || true
fi

SUDO mkdir -p /etc/slurm /var/spool/slurmctld /var/spool/slurmd /var/log/slurm
SUDO chown -R slurm:slurm /var/spool/slurmctld /var/spool/slurmd /var/log/slurm

# --- 7. write default slurm.conf + gres.conf --------------------------------
HOST=$(hostname)
NPROC=$(nproc)
MEM_MB=$(awk '/^MemTotal:/{print int($2/1024)}' /proc/meminfo)
NVIDIA_DEV_COUNT=$(ls /dev/nvidia[0-9] 2>/dev/null | wc -l)
GPU_GRES_LINE=""
GRES_TYPES_LINE=""

if [[ "$NVIDIA_DEV_COUNT" -gt 0 ]]; then
    GPU_GRES_LINE=" Gres=gpu:$NVIDIA_DEV_COUNT"
    GRES_TYPES_LINE="GresTypes=gpu"
fi

if [[ ! -f /etc/slurm/slurm.conf ]]; then
    log "writing default slurm.conf (single-node, ~$NPROC CPUs, ~${MEM_MB}MB RAM)"
    SUDO bash -c "cat > /etc/slurm/slurm.conf <<EOF
ClusterName=local
SlurmctldHost=$HOST
SlurmUser=slurm
AuthType=auth/munge

StateSaveLocation=/var/spool/slurmctld
SlurmdSpoolDir=/var/spool/slurmd
SlurmctldPidFile=/run/slurmctld.pid
SlurmdPidFile=/run/slurmd.pid

SlurmctldLogFile=/var/log/slurm/slurmctld.log
SlurmdLogFile=/var/log/slurm/slurmd.log

SchedulerType=sched/backfill
SelectType=select/cons_tres
SelectTypeParameters=CR_Core_Memory
$GRES_TYPES_LINE

# proctrack/linuxproc + task/affinity: cgroup-free; works on any kernel cgroup version
ProctrackType=proctrack/linuxproc
TaskPlugin=task/affinity
JobAcctGatherType=jobacct_gather/linux
JobAcctGatherFrequency=30

ReturnToService=2
SlurmctldTimeout=120
SlurmdTimeout=300
MessageTimeout=10
MinJobAge=300

NodeName=$HOST CPUs=$NPROC RealMemory=$((MEM_MB - 2000))$GPU_GRES_LINE State=UNKNOWN
PartitionName=local Nodes=ALL Default=YES MaxTime=INFINITE State=UP
EOF
chown root:slurm /etc/slurm/slurm.conf && chmod 644 /etc/slurm/slurm.conf"

    if [[ "$NVIDIA_DEV_COUNT" -gt 0 ]]; then
        log "writing gres.conf for $NVIDIA_DEV_COUNT GPU(s)"
        SUDO bash -c "cat > /etc/slurm/gres.conf <<EOF
$(for i in $(seq 0 $((NVIDIA_DEV_COUNT - 1))); do
    echo \"NodeName=$HOST Name=gpu File=/dev/nvidia$i\"
done)
EOF
chown root:slurm /etc/slurm/gres.conf && chmod 644 /etc/slurm/gres.conf"
    fi
else
    log "/etc/slurm/slurm.conf already exists; leaving in place"
fi

# --- 8. systemd units + start ------------------------------------------------
# Source-built slurm doesn't ship systemd units automatically; the source has them in
# etc/slurmctld.service and etc/slurmd.service. Install if not already present.
for unit in slurmctld slurmd; do
    if [[ ! -f /etc/systemd/system/$unit.service && -f "$SRC/etc/$unit.service" ]]; then
        SUDO cp "$SRC/etc/$unit.service" /etc/systemd/system/
    fi
done
SUDO systemctl daemon-reload

log "starting slurmctld"
SUDO systemctl enable --now slurmctld 2>&1 | tail -2 || fatal "slurmctld start failed" 5
sleep 2

log "starting slurmd"
SUDO systemctl enable --now slurmd 2>&1 | tail -2 || fatal "slurmd start failed" 5
sleep 3

# --- 9. verify ---------------------------------------------------------------
if sinfo >/dev/null 2>&1; then
    log "INSTALL OK"
    sinfo
    exit 0
else
    log "INSTALL: daemons started but sinfo failed"
    SUDO journalctl -u slurmctld -n 10 --no-pager 2>&1 | tail -10
    exit 5
fi
