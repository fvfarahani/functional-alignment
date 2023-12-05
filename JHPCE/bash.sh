#!/bin/bash
#$ -pe local 10
#$ -l gpu,mem_free=300G,h_vmem=300G
#$ -M meet10may@gmail.com
#$ -m ea
#$ -o /users/tnath1/output
#$ -e /users/tnath1/error
module load conda
conda activate myclone
python3 run_mediation.py
