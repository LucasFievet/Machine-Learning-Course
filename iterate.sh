#!/bin/bash
for i in `seq 0 20`;
do
	bsub -J job$i -B -N -R "rusage[mem=1200]" "python3 main.py predict_cut_iterate $i"
done  
