#!/bin/bash

MASTER_LOG="pipeline_time.log"

start_time=$(date +%s)

echo "==================================" >> $MASTER_LOG
echo "Pipeline started at: $(date)" >> $MASTER_LOG
echo "==================================" >> $MASTER_LOG

python clean_codes_v2/run_train_pipeline.py --config-file train_7_9_50k.json --train 0 --N 1 --fresh_run 0 > lcb1.log 2>&1 && \
python clean_codes_v2/run_train_pipeline.py --config-file train_7_9_50k.json --train 0 --N 2 --fresh_run 0 > lcb2.log 2>&1 && \
python clean_codes_v2/run_train_pipeline.py --config-file train_7_9_50k.json --train 1 --fresh_run 0 > train.log 2>&1

end_time=$(date +%s)

echo "Pipeline ended at: $(date)" >> $MASTER_LOG
echo "Total Time: $((end_time - start_time)) seconds" >> $MASTER_LOG
echo "==================================" >> $MASTER_LOG
