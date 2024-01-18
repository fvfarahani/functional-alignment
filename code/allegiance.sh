#!/bin/bash
#$ -l mem_free=20G,h_vmem=20G,h_fsize=5G
#$ -m a -M farzad.vasheghani@gmail.com
#$ -o /dcs05/ciprian/smart/farahani/SL-CHA/code/out_allegiance
#$ -e /dcs05/ciprian/smart/farahani/SL-CHA/code/err_allegiance
module load conda
conda activate py3
python allegiance.py