#!/bin/bash
#steps = 1000
#feature_iterations = 2096920
#iterations = 53
#threads = 40

memory=1000

for i in `seq 0 52`;
do
	for j in `seq 0 39`;
	do		
		num=$(expr $i \* 40 + $j)
		if [ $i -eq 0 ]
		then
			bsub -J job_chain_$j -n 1 -R "rusage[mem=$memory]" "python3 main.py predict_cut_iterate $num"
		else
			bsub -J job_chain_$j -w "done(job_chain_$j)" -R -n 1 "rusage[mem=$memory]" "python3 main.py predict_cut_iterate $num"
		fi
	done
done  
 
