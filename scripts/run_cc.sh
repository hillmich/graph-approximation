#!/usr/bin/env bash

#Permutationen durchlaufen lassen:
#- tracking auf Adrians Daten
#- tracking auf eigenen Bubbles (mit 2,4,5,6 Pixeln)
#- look-ahead 2,4,6,8

source master-convert/bin/activate

date_format='%Y-%m-%dT%H-%M-%S'
log_file="results/run_cc_$(date +"${date_format}").log"



start_time=$(date)
echo Started:   ${start_time}

mkdir -p "results"
echo ${start_time} > ${log_file}

for crop in ./data/bwsmooth/*/
do
    for radius in 2 4 5 6
    do
        echo -n $(date -u +"${date_format}") convert_bw.py radius: ${radius} "${crop}"
        python src/convert_bw.py --min-radius ${radius} --reverse --output-dir "results/converted/bw_r${radius}" "${crop}" &>> ${log_file}
        echo " --> $?"
    done
done

for crop in ./data/matlab/*/
do
    echo -n $(date -u +"${date_format}") convert_matlab.py "${crop}"
    python src/convert_matlab.py --output-dir "results/converted/matlab" "${crop}" &>> ${log_file}
    echo " --> $?"
done


for crop in ./results/converted/*/*/
do
    algo_base="$(basename $(dirname "${crop}"))"
    gap=0

    for steps in 1 2 4 6 8
    do
        let "gap = $steps - 1"
        echo -n $(date -u +"${date_format}") cluster_ahead.py look-ahead: ${steps} gap: ${gap} "${crop}"
        python src/cluster_ahead.py --output-dir "results/clustered/${algo_base}_ahead_s${steps}_g${gap}/" \
            --look-ahead-steps ${steps} \
            --max-gap-fill ${gap} \
            "${crop}" &>> ${log_file}
        echo " --> $?"
    done
done

tar -zcf results.tar.gz results

echo "Started:  ${start_time}"
echo "Finished: $(date)"
