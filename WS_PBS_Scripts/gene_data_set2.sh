#!/bin/bash

# =========================
# PBS directives (used only on HPC)
# =========================
#PBS -l select=1:ncpus=32
#PBS -l walltime=96:00:00
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
    JOBNAME="gen_lc_s2_local"
fi

cd "$WORKDIR"

# =========================
# Inputs (PBS or local)
# =========================

ratio1=${ratio1:-9}
ratio2=${ratio2:-10}
seed=${seed:-0}
base_dir=${base_dir:-"./logs"}

mkdir -p "$base_dir"

# =========================
# Logging (same for both)
# =========================

LOGFILE="$base_dir/${JOBNAME}.o${JOBID}"
ERRFILE="$base_dir/${JOBNAME}.e${JOBID}"

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

# =========================
# Start
# =========================

start_time=$(date +%s)

echo "=================================="
echo "Job started at: $(date)"
echo "JOBNAME=$JOBNAME"
echo "JOBID=$JOBID"
echo "ratio1=$ratio1 ratio2=$ratio2 seed=$seed"
echo "base_dir=$base_dir"
echo "=================================="

# =========================
# Work
# =========================

python ../clean_codes_v2/run_train_pipeline.py \
    --config-file train_${ratio1}_${ratio2}.json \
    --train 0 --N 5 --fresh_run 0 \
    > "${base_dir}/N5_${JOBID}.log" 2>&1 && \

python ../clean_codes_v2/run_train_pipeline.py \
    --config-file train_${ratio1}_${ratio2}.json \
    --train 0 --N 6 --fresh_run 0 \
    > "${base_dir}/N6_${JOBID}.log" 2>&1 && \

python ../clean_codes_v2/run_train_pipeline.py \
    --config-file train_${ratio1}_${ratio2}.json \
    --train 0 --N 7 --fresh_run 0 \
    > "${base_dir}/N7_${JOBID}.log" 2>&1 && \

python ../clean_codes_v2/run_train_pipeline.py \
    --config-file train_${ratio1}_${ratio2}.json \
    --train 0 --N 8 --fresh_run 0 \
    > "${base_dir}/N8_${JOBID}.log" 2>&1

# =========================
# End timing
# =========================

end_time=$(date +%s)
elapsed=$((end_time - start_time))

echo "=================================="
echo "Job ended at: $(date)"
printf "Walltime: %02d:%02d:%02d\n" \
    $((elapsed/3600)) \
    $((elapsed%3600/60)) \
    $((elapsed%60))
echo "=================================="