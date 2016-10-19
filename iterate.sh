#!/bin/bash
#steps = 10000
#feature_iterations = 2096920

memory=1000

for i in `seq 0 210`;
do
		bsub -J job_chain_$i -n 1 -R "rusage[mem=$memory]" "python3 main.py predict_cut_iterate $i"
		echo $i
done  
 
