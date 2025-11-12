#!/bin/bash

module load cuda
source ~/.bashrc
conda activate zq_torch 

sparse_ratio=${1:-0.7}
causal_flag=${2:-true}

if [ "$causal_flag" == "true" ]; then
    causal="--causal"
    suffix="causal"
else
    causal=""
    suffix="nocausal"
fi

OUTPUT_FILE=./test/new_benchmark_${suffix}_results_sparse${sparse_ratio}.json

SEQLENS=(1024 2048 4096 8192 16384 32768 65536 131072)

for seqlen in "${SEQLENS[@]}"
do
    echo "Benchmarking with unit_seqlen $seqlen, sparse_ratio $sparse_ratio, $suffix"
    python ./test/test_speedup.py \
        --unit_seqlen $seqlen \
        --nruns 15 \
        --nwarmup 5 \
        $causal \
        --save_benchmark_to_file $OUTPUT_FILE \
        --sparse_ratio $sparse_ratio
done