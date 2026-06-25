#!/bin/bash

# =========================
# PBS directives (HPC only)
# =========================
#PBS -l select=1:ncpus=32
#PBS -l walltime=00:10:00
#PBS -q project
#PBS -P hpc2601012
#PBS -N gen_shapes

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
    JOBNAME="gen_shapes_local"
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

echo "rsrp1=$ratio1"
echo "rsrp2=$ratio2"

# =========================
# Inputs
# =========================

RSRP1=${ratio1:-2}
RSRP2=${ratio2:-3}
SEED=${seed:-0}

echo "Running Kepler pipeline"

python ../clean_codes_v2/run_kepler_pipeline.py

echo "Running shape generation pipeline"

# =========================
# Shape generation (unchained version for safety in both local + HPC)
# =========================

# for N in $(seq 1 12); do
#     echo "Running N=$N"

#     python ../clean_codes_v2/run_train_pipeline.py \
#         --config-file $config_file \
#         --train 0 \
#         --N $N \
#         --fresh_run 2 \
#         > "${base_dir}/N${N}_${JOBID}.log" 2>&1
# done

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