
echo "Benchmarking with unit_seqlen 128"
python ./test/test_speedup.py --unit_seqlen 128 --nruns 15 --nwarmup 5

echo "Benchmarking with unit_seqlen 256"
python ./test/test_speedup.py --unit_seqlen 256 --nruns 15 --nwarmup 5

echo "Benchmarking with unit_seqlen 512"
python ./test/test_speedup.py --unit_seqlen 512 --nruns 15 --nwarmup 5

echo "Benchmarking with unit_seqlen 1024"
python ./test/test_speedup.py --unit_seqlen 1024 --nruns 15 --nwarmup 5

echo "Benchmarking with unit_seqlen 2048"
python ./test/test_speedup.py --unit_seqlen 2048 --nruns 15 --nwarmup 5

echo "Benchmarking with unit_seqlen 4096"
python ./test/test_speedup.py --unit_seqlen 4096 --nruns 15 --nwarmup 5

echo "Benchmarking with unit_seqlen 8192"
python ./test/test_speedup.py --unit_seqlen 8192 --nruns 15 --nwarmup 5

echo "Benchmarking with unit_seqlen 16384"
python ./test/test_speedup.py --unit_seqlen 16384 --nruns 15 --nwarmup 5

echo "Benchmarking with unit_seqlen 32768"
python ./test/test_speedup.py --unit_seqlen 32768 --nruns 15 --nwarmup 5

echo "Benchmarking with unit_seqlen 65536"
python ./test/test_speedup.py --unit_seqlen 65536 --nruns 15 --nwarmup 5

echo "Benchmarking with unit_seqlen 131072"
python ./test/test_speedup.py --unit_seqlen 131072 --nruns 15 --nwarmup 5