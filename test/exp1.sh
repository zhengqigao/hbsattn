#!/bin/bash

# sbatch ./test/sbatch_speedup.sh 0.3 true
# sbatch ./test/sbatch_speedup.sh 0.5 true
# sbatch ./test/sbatch_speedup.sh 0.7 true

# sbatch ./test/sbatch_speedup.sh 0.3 false
# sbatch ./test/sbatch_speedup.sh 0.5 false
# sbatch ./test/sbatch_speedup.sh 0.7 false

./test/bash_speedup.sh 0.3 true
./test/bash_speedup.sh 0.4 true
./test/bash_speedup.sh 0.5 true
./test/bash_speedup.sh 0.6 true
./test/bash_speedup.sh 0.7 true
./test/bash_speedup.sh 0.8 true
./test/bash_speedup.sh 0.9 true


./test/bash_speedup.sh 0.3 false
./test/bash_speedup.sh 0.4 false
./test/bash_speedup.sh 0.5 false
./test/bash_speedup.sh 0.6 false
./test/bash_speedup.sh 0.7 false
./test/bash_speedup.sh 0.8 false
./test/bash_speedup.sh 0.9 false
