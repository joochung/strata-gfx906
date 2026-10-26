#!/usr/bin/env bash
# run-p6.0-probe.sh - operator-launched P6.0 small-collective probe (task #35).
#
# The probe maps ~6 MB x cards across the 4 cards (< 1% of one card): a different risk class
# from an engine, but the standing one-engine rule still applies, so this script REFUSES while
# any engine is alive (vLLM, strata-server, glm/ds41 engines). Numbers are the P6 decode-TP
# kill gate - a design input, not a speed claim (AGENTS.md).
#
# Usage (free window, everything else stopped - typically via unverbraucht/run-with-free-gpus.sh):
#   tools/gfx906/run-p6.0-probe.sh [modes]      modes default: kern,bounce,peer,rccl
set -u
cd "$(dirname "$0")/../.."
TS=$(date +%Y%m%d-%H%M%S)
mkdir -p logs
LOG="logs/p6.0-probe-${TS}.log"
RCCL_LOG="logs/p6.0-probe-${TS}.rccl-info.log"

if pgrep -f "engine/strata|vllm|glm_generate|ds41_generate" >/dev/null 2>&1; then
    echo "refusing: another engine is alive - one HIP engine at a time (SVM crash rule)"; exit 3
fi
AVAIL=$(( $(awk '/MemAvailable/{print $2}' /proc/meminfo) / 1048576 ))
[ "$AVAIL" -ge 10 ] || { echo "refusing: ${AVAIL} GiB available < 10 (probe pins ~5 GiB + headroom)"; exit 3; }

echo "=== p6.0 probe start $(date +%T) (avail ${AVAIL} GiB) ===" | tee "$LOG"
echo "--- sysfs link state (authority for link claims):" | tee -a "$LOG"
for b in 23:00.0 26:00.0 43:00.0 46:00.0; do
    echo "  $b cur=$(cat /sys/bus/pci/devices/0000:$b/current_link_speed) x$(cat /sys/bus/pci/devices/0000:$b/current_link_width)" \
         "max=$(cat /sys/bus/pci/devices/0000:$b/max_link_speed) x$(cat /sys/bus/pci/devices/0000:$b/max_link_width)" | tee -a "$LOG"
done

BIN=build-906/probe-collective
if [ ! -x "$BIN" ] || [ tools/gfx906/probe-collective.cu -nt "$BIN" ]; then
    echo "--- building probe (links RCCL here only):" | tee -a "$LOG"
    /opt/rocm/bin/hipcc --offload-arch=gfx906 -O2 -pthread \
        -I/opt/rocm/include -L/opt/rocm/lib -lrccl \
        -o "$BIN" tools/gfx906/probe-collective.cu 2>&1 | tee -a "$LOG"
    [ -x "$BIN" ] || { echo "build failed"; exit 1; }
fi

# RCCL transport choice (P2P vs SHM per channel) is the receipt's provenance for the rccl rows:
# capture INFO to a side log, print the transport lines inline.
# THREE guarded runs, each with its own timeout, because a hung spin on this box burns the whole
# window silently (witness r2: kernel XFB peer stores stalled probe r2 past its 900 s timeout).
# rccl-default may TIMEOUT if RCCL picks the XFB P2P protocol - that is a FINDING, not a failure;
# rccl-shm (NCCL_P2P_DISABLE=1) is the fallback transport measurement.
export RCCL_DEBUG=INFO

rc_all=0

echo "--- [1/3] kern+bounce+peer (timeout 600)" | tee -a "$LOG"
timeout 600 "$BIN" "${1:-kern,bounce,peer}" 2>>"$RCCL_LOG" | tee -a "$LOG"
rc=${PIPESTATUS[0]}; [ $rc -eq 0 ] || { echo "p6.0: kern/bounce/peer rc=$rc" | tee -a "$LOG"; rc_all=$rc; }

echo "--- [2/3] rccl default transport (timeout 240)" | tee -a "$LOG"
timeout 240 env NCCL_DEBUG=INFO "$BIN" rcclmp 0,1 2>>"$RCCL_LOG" | tee -a "$LOG"
rc=${PIPESTATUS[0]}
if [ $rc -eq 124 ]; then
    echo "FINDING: rccl DEFAULT transport TIMED OUT at 240 s on this box (P2P/XFB protocol suspected)."
    echo "         decode-TP must not assume rccl-default; see rccl-shm below." | tee -a "$LOG"
else
    [ $rc -eq 0 ] || { echo "p6.0: rccl-default rc=$rc" | tee -a "$LOG"; rc_all=$rc; }
fi

echo "--- [3/3] rccl with NCCL_P2P_DISABLE=1 (timeout 240)" | tee -a "$LOG"
timeout 240 env NCCL_DEBUG=INFO NCCL_P2P_DISABLE=1 "$BIN" rcclmp 0,1 2>>"$RCCL_LOG" | tee -a "$LOG"
rc=${PIPESTATUS[0]}
if [ $rc -eq 124 ]; then
    echo "p6.0: rccl-shm ALSO timed out" | tee -a "$LOG"; rc_all=4
elif [ $rc -ne 0 ]; then
    echo "p6.0: rccl-shm rc=$rc" | tee -a "$LOG"; rc_all=$rc
fi

grep -E "via |Transport|P2P|SHM" "$RCCL_LOG" | sort | uniq -c | sort -rn | head -40 | tee -a "$LOG"
echo "=== p6.0 probe rc=$rc log=$LOG rccl-info=$RCCL_LOG ===" | tee -a "$LOG"
if [ "$rc" -ne 0 ]; then
    echo "failure ladder: re-run with HIP_LAUNCH_BLOCKING=1 and report the first FATAL line;"
    echo "a MISMATCH sanity column is a FAILURE of that transport, not a noise column."
fi
exit $rc
