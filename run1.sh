#!/bin/bash
#SBATCH -c2
#SBATCH --time=2-0

# export PYTHONPATH=$PYTHONPATH:"$(dirname "$(pwd)")"/Marabou

# python3 WatermarkVerification1.py --model mnist.w.wm --epsilon_max 0.5 --epsilon_interval 0.001

echo $SLURM_ARRAY_TASK_ID