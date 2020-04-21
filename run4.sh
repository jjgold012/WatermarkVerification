#!/bin/bash

#SBATCH -c2
#SBATCH --time=7-0
#SBATCH --mem-per-cpu=4096
#SBATCH --mail-type=BEGIN,END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jjgold@cs.huji.ac.il    # Where to send mail	
#SBATCH --array=1-5

export PYTHONPATH=$PYTHONPATH:"$(dirname "$(pwd)")"/Marabou

python3 WatermarkVerification4.py --model mnist.w.wm --num_of_inputs $SLURM_ARRAY_TASK_ID --epsilon_max 300 --start 0 --finish 99

