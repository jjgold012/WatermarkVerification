#!/bin/bash
#SBATCH -c2
#SBATCH --time=2-0
#SBATCH --mem-per-cpu=2g

export PYTHONPATH=$PYTHONPATH:"$(dirname "$(pwd)")"/Marabou

python3 WatermarkVerification1.py --model mnist.w.wm --epsilon_max 0.5 --epsilon_interval 0.001