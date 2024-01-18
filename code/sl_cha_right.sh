#!/bin/bash
#$ -pe local 1
#$ -l mem_free=300G,h_vmem=300G,h_fsize=300G
#$ -m a -M farzad.vasheghani@gmail.com
#$ -o /dcs05/ciprian/smart/farahani/SL-CHA/code/out_haR
#$ -e /dcs05/ciprian/smart/farahani/SL-CHA/code/err_haR
module load conda
conda activate py3
python sl_cha_right.py