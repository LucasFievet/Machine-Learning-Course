#!/bin/bash
for i in `seq 0 20`;
do
	bsub -J job$i -B -N -R "rusage[mem=5000]" "python3 Machine-Learning-Course/main.py predict_cut_iterate $i"
done  
