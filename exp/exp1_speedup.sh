cd ..

python test_speedup.py --unit_seqlen 128 --nruns 15 --nwarmup 5
python test_speedup.py --unit_seqlen 256 --nruns 15 --nwarmup 5
python test_speedup.py --unit_seqlen 512 --nruns 15 --nwarmup 5
python test_speedup.py --unit_seqlen 1024 --nruns 15 --nwarmup 5
python test_speedup.py --unit_seqlen 2048 --nruns 15 --nwarmup 5
python test_speedup.py --unit_seqlen 4096 --nruns 15 --nwarmup 5
python test_speedup.py --unit_seqlen 8192 --nruns 15 --nwarmup 5
python test_speedup.py --unit_seqlen 16384 --nruns 15 --nwarmup 5
python test_speedup.py --unit_seqlen 32768 --nruns 15 --nwarmup 5
python test_speedup.py --unit_seqlen 65536 --nruns 15 --nwarmup 5
python test_speedup.py --unit_seqlen 131072 --nruns 15 --nwarmup 5