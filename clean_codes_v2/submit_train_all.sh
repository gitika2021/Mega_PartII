jid=$(qsub train_array.pbs)
basejid=$(echo $jid | cut -d'[' -f1)

qsub -W depend=afterok:${basejid} train_final.pbs