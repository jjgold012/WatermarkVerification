#!/bin/bash
#SBATCH --time=7-0
#SBATCH --mem-per-cpu=4096
#SBATCH -c4

export PYTHONPATH=$PYTHONPATH:"$(dirname "$(pwd)")"/Marabou

python3 WatermarkVerification2.py --model mnist.w.wm --epsilon_max 200
# python3 WatermarkVerification2.py --model test --epsilon_max 3 --epsilon_interval 0.5
