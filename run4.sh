#!/bin/bash

#SBATCH -c4
#SBATCH --time=7-0
#SBATCH --mem-per-cpu=4096
#SBATCH --mail-type=BEGIN,END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jjgold@cs.huji.ac.il    # Where to send mail	
#SBATCH --array=0-99

export PYTHONPATH=$PYTHONPATH:"$(dirname "$(pwd)")"/Marabou

start=$(($SLURM_ARRAY_TASK_ID*100))
finish=$((start+99))

python3 WatermarkVerification4.py --model mnist.w.wm --epsilon_max 100 --start $start --finish $finish

