#!/bin/bash
# $ -pe local 12
#$ -l mem_free=50G,h_vmem=50G,h_fsize=10G
#$ -m a -M farzad.vasheghani@gmail.com
#$ -o /dcs05/ciprian/smart/farahani/SL-CHA/code/out_id_acc
#$ -e /dcs05/ciprian/smart/farahani/SL-CHA/code/err_id_acc
module load conda
conda activate py3
python id_acc.py