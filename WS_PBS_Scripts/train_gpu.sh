#!/bin/bash

# =========================
# PBS directives (HPC only)
# =========================
#PBS -l select=1:ncpus=32:mpiprocs=1:ngpus=1
#PBS -l walltime=96:00:00
#PBS -q regular
#PBS -P hpc2601012
#PBS -N train

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
    JOBNAME="train_gpu_local"
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
# Conda setup (HPC + Local safe)
# =========================

if [ -f "/mnt/home/project/cshukla.gitika/anaconda3/etc/profile.d/conda.sh" ]; then
    source /mnt/home/project/cshukla.gitika/anaconda3/etc/profile.d/conda.sh
elif command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    echo "ERROR: conda not found"
    exit 1
fi

conda activate genenv_gitika

# =========================
# Start
# =========================

start_time=$(date +%s)

echo "=================================="
echo "Pipeline started at: $(date)"
echo "JOBNAME=$JOBNAME"
echo "JOBID=$JOBID"
echo "=================================="

# =========================
# Inputs
# =========================

RSRP1=${ratio1:-9}
RSRP2=${ratio2:-10}
SEED=${seed:-123}

echo "rsrp1=$RSRP1"
echo "rsrp2=$RSRP2"
echo "seed=$SEED"

# =========================
# Run training
# =========================

python ../clean_codes_v2/run_train_pipeline.py \
    --config-file train_${ratio1}_${ratio2}.json \
    --train 1

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