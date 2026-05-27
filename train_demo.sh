#!/bin/bash

MASTER_LOG="pipeline_genshape.log"

start_time=$(date +%s)

echo "==================================" >> $MASTER_LOG
echo "Pipeline started at: $(date)" >> $MASTER_LOG
echo "==================================" >> $MASTER_LOG

echo "Generate Light Curves"

python clean_codes_v2/run_train_pipelinev2.py --config-file demo_config.json --train 0 --N 1 --fresh_run 1 > logs/lc1.log 2>&1 && \
python clean_codes_v2/run_train_pipelinev2.py --config-file demo_config.json --train 0 --N 2 --fresh_run 1 > logs/lc2.log 2>&1 && \
python clean_codes_v2/run_train_pipelinev2.py --config-file demo_config.json --train 0 --N 3 --fresh_run 1 > logs/lc3.log 2>&1 && \
end_time=$(date +%s)
elapsed=$((end_time - start_time))

echo "Pipeline ended at: $(date)" >> $MASTER_LOG
echo "Total walltime (seconds): $elapsed" >> $MASTER_LOG
echo "Total walltime (hh:mm:ss): $(printf '%02d:%02d:%02d\n' $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60)))" >> $MASTER_LOG
echo "==================================" >> $MASTER_LOG
