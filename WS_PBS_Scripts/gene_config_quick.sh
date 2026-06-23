#!/bin/bash

# =========================
# PBS directives (used on HPC only)
# =========================
#PBS -l select=1:ncpus=32
#PBS -l walltime=00:05:00
#PBS -q project
#PBS -P hpc2601012
#PBS -N gen_lc

# =========================
# Environment detection
# =========================

if [ -n "$PBS_O_WORKDIR" ]; then
    WORKDIR="$PBS_O_WORKDIR"
    JOBID="$PBS_JOBID"
    JOBNAME="$PBS_JOBNAME"
else
    WORKDIR=$(pwd)
    JOBID=$(date +%Y%m%d_%H%M%S)
    JOBNAME="local_gen_lc"
fi

cd "$WORKDIR"

# =========================
# Logging
# =========================

LOGDIR="${base_dir:-./logs}"
mkdir -p "$LOGDIR"

LOGFILE="$LOGDIR/${JOBNAME}.o${JOBID}"
ERRFILE="$LOGDIR/${JOBNAME}.e${JOBID}"

exec > "$LOGFILE"
exec 2> "$ERRFILE"

# =========================
# Conda setup
# =========================
if [ -f "/mnt/home/project/cshukla.gitika/anaconda3/etc/profile.d/conda.sh" ]; then
    source /mnt/home/project/cshukla.gitika/anaconda3/etc/profile.d/conda.sh
elif command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    echo "ERROR: conda not found"
    exit 1
fi

#source /mnt/home/project/cshukla.gitika/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate genenv_gitika

start_time=$(date +%s)

echo "=================================="
echo "Pipeline started at: $(date)"
echo "=================================="

echo "rsrp1=$ratio1"
echo "rsrp2=$ratio2"

# =========================
# Inputs
# =========================

RSRP1=${ratio1:-9}
RSRP2=${ratio2:-10}
SEED=${seed:-0}

# =========================
# Python config generation (ROBUST PATH HANDLING)
# =========================

python3 << EOF
import sys
import json
from pathlib import Path

# -------------------------
# Robust project root detection
# -------------------------
# Works on both PBS and local execution
try:
    base_dir = Path(__file__).resolve().parent.parent
except NameError:
    base_dir = Path.cwd().resolve().parent

clean_codes_dir = base_dir / "clean_codes_v2"
sys.path.insert(0, str(clean_codes_dir))

from paths import *

rsrp1 = int("$RSRP1")
rsrp2 = int("$RSRP2")
seed = int("$SEED")

input_file = base_dir / "Config" / "train_config_debug.json"

with open(input_file, "r") as f:
    config = json.load(f)

config["rsrp1"] = rsrp1
config["rsrp2"] = rsrp2
config["seed"] = seed

config_out_dir = Path(Config_Dir)
config_out_dir.mkdir(parents=True, exist_ok=True)

outfile = Path(Config_Dir) / f"train_{rsrp1}_{rsrp2}.json"

with open(outfile, "w") as f:
    json.dump(config, f, indent=4)

print(f"Created: {outfile}")
print(f"configuration updated successfully")
EOF

# =========================
# End timing
# =========================

end_time=$(date +%s)
elapsed=$((end_time - start_time))

echo "=================================="
echo "Job ended at: $(date)"
echo "Total walltime (seconds): $elapsed"

printf "Total walltime (hh:mm:ss): %02d:%02d:%02d\n" \
    $((elapsed/3600)) \
    $((elapsed%3600/60)) \
    $((elapsed%60))
echo "=================================="