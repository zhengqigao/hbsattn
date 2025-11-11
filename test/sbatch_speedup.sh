#!/bin/bash
#SBATCH --job-name=speedup_bench
#SBATCH --output=./slurmlogs/sbatch_speedup_%j.log
#SBATCH --partition mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G


# example: sbatch sbatch_speedup.sh 0.5 causal
# $1: sparse_ratio，默认 0.7
# $2: causal（true/false），默认 true


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

OUTPUT_FILE=./test/benchmark_${suffix}_results_sparse${sparse_ratio}.json

SEQLENS=(128 256 512 1024 2048 4096 8192 16384 32768 65536 131072)

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