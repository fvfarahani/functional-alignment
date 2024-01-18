#!/bin/bash
# $ -pe local 12
#$ -l mem_free=50G,h_vmem=50G,h_fsize=50G
#$ -m a -M farzad.vasheghani@gmail.com
#$ -o /dcs05/ciprian/smart/farahani/SL-CHA/code/out_meso_fine
#$ -e /dcs05/ciprian/smart/farahani/SL-CHA/code/err_meso_fine
module load conda
conda activate py3
python meso_fine.py