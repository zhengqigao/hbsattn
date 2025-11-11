#!/bin/bash

sbatch ./test/sbatch_speedup.sh 0.3 true
sbatch ./test/sbatch_speedup.sh 0.5 true
sbatch ./test/sbatch_speedup.sh 0.7 true

sbatch ./test/sbatch_speedup.sh 0.3 false
sbatch ./test/sbatch_speedup.sh 0.5 false
sbatch ./test/sbatch_speedup.sh 0.7 false
