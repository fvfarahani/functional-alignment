#!/bin/bash
# $ -pe local 12
#$ -l mem_free=3G,h_vmem=3G,h_fsize=1G
#$ -m a -M farzad.vasheghani@gmail.com
#$ -o /dcs05/ciprian/smart/farahani/SL-CHA/code/out_graph
#$ -e /dcs05/ciprian/smart/farahani/SL-CHA/code/err_graph
module load conda
conda activate py3
python graph.py