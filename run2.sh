#!/bin/bash
#SBATCH --time=7-0
#SBATCH --mem-per-cpu=4096
#SBATCH --mail-type=BEGIN,END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jjgold@cs.huji.ac.il    # Where to send mail	
#SBATCH --array=0-49
#SBATCH -c4

export PYTHONPATH=$PYTHONPATH:"$(dirname "$(pwd)")"/Marabou

start=$SLURM_ARRAY_TASK_ID*2
finish=$start+2

python3 WatermarkVerification2.py --model mnist.w.wm --epsilon_max 200 --start $start --finish $finish
# python3 WatermarkVerification2.py --model test --epsilon_max 3 --epsilon_interval 0.5 --finish 2
