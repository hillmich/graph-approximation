#!/usr/bin/env bash

source master-convert/bin/activate

date_format='%Y-%m-%dT%H-%M-%S'
log_file="figures/run_stats_$(date +"${date_format}").log"

mkdir -p figures

start_time=$(date)
echo Started:   ${start_time}
echo ${start_time} > ${log_file}

find "results/clustered/" -type d -links 2 -print0 | \
    sort -z | \
    xargs --null -n 1 python src/crop_stats.py --max-degree 5 --debug |& \
    tee -a ${log_file}

echo "Started:  ${start_time}"
echo "Finished: $(date)"
