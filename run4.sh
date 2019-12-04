#!/bin/bash
#SBATCH --time=7-0

export PYTHONPATH=$PYTHONPATH:"$(dirname "$(pwd)")"/Marabou

python3 WatermarkVerification4.py --model mnist.w.wm
# python3 WatermarkVerification2.py --model test --epsilon_max 100
