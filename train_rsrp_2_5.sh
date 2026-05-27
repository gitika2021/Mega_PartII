#!/bin/bash

MASTER_LOG="pipeline_time_rsrp_2_5.log"

start_time=$(date +%s)

echo "==================================" >> $MASTER_LOG
echo "Pipeline started at: $(date)" >> $MASTER_LOG
echo "==================================" >> $MASTER_LOG

python clean_codes_v2/run_train_pipeline.py --config-file train_2_5.json --train 0 --N 1 --fresh_run 1 > logs/lcb1_rsrp_2_5.log 2>&1 && \
python clean_codes_v2/run_train_pipeline.py --config-file train_2_5.json --train 0 --N 2 --fresh_run 1 > logs/lcb2_rsrp_2_5.log 2>&1 && \
python clean_codes_v2/run_train_pipeline.py --config-file train_2_5.json --train 0 --N 3 --fresh_run 1 > logs/lcb3_rsrp_2_5.log 2>&1 && \
python clean_codes_v2/run_train_pipeline.py --config-file train_2_5.json --train 1 --fresh_run 0 > logs/train_rsrp_2_5.log 2>&1

end_time=$(date +%s)
elapsed=$((end_time - start_time))

echo "Pipeline ended at: $(date)" >> $MASTER_LOG
echo "Total walltime (seconds): $elapsed" >> $MASTER_LOG
echo "Total walltime (hh:mm:ss): $(printf '%02d:%02d:%02d\n' $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60)))" >> $MASTER_LOG
echo "==================================" >> $MASTER_LOG
