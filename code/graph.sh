#!/bin/bash
#SBATCH --partition=shared
#SBATCH --time=50:00:00					# time limit (D-HH:MM:SS)
#SBATCH --output=/dcs05/ciprian/smart/farahani/SL-CHA/code/out/slurm-%j.out
#SBATCH --error=/dcs05/ciprian/smart/farahani/SL-CHA/code/err/slurm-%j.err
#SBATCH --job-name=GRAPH				# name job for easier spotting, controlling
#SBATCH --cpus-per-task=1				# number of cores
#SBATCH --ntasks=1					# number of tasks running in parallel
#SBATCH --mem=1G					# memory per __node__
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=farzad.vasheghani@gmail.com

module load conda
conda activate py3
python graph.py