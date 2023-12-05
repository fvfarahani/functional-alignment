#!/bin/bash
#$ -pe local 1
#$ -l gpu,mem_free=50G,h_vmem=50G
#$ -m e -M farzad.vasheghani@gmail.com
#$ -o /users/ffarahan/output
#$ -e /users/ffarahan/error
module load conda
condo activate py3
python test.py
