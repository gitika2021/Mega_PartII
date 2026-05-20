jid=$(qsub gene_data_rsrp_12_15.pbs)
basejid=$(echo $jid | cut -d'[' -f1)

qsub -W depend=afterok:${basejid} train_data_rsrp_12_15.pbs