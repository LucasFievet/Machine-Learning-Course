#!/bin/bash
#steps = 10000
#feature_iterations = 2096920
#iterations = 53
#threads = 48

memory=1000

for i in `seq 0 4`;
do
	for j in `seq 0 47`;
	do
		num=$(expr $i \* 48 + $j)
		bsub -J job_chain_$num -n 1 -R "rusage[mem=$memory]" "python3 main.py predict_cut_iterate $num"
		echo $num
	done
done  
 
