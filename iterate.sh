#!/bin/bash
bsub -J job_chain -B -N -R "rusage[mem=5000]" "python3 Machine-Learning-Course/main.py predict_cut_iterate"
for i in `seq 2 21`;
do
	bsub -J job_chain -w "done(job_chain)" -B -N -R "rusage[mem=5000]" "python3 Machine-Learning-Course/main.py predict_cut_iterate"
done  
