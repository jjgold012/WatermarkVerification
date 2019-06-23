#!/bin/bash
#SBATCH --killable
#SBATCH -c2
#SBATCH --time=2-0
#SBATCH --mem=2g

export PYTHONPATH=$PYTHONPATH:"$(dirname "$(pwd)")"/Marabou

# python3 WatermarkVerification2.py --model mnist.w.wm --epsilon_max 100
python3 WatermarkVerification1.py --model mnist.w.wm --epsilon_max 0.5 --epsilon_interval 0.001