#!/bin/bash
#SBATCH --killable
#SBATCH -c2
#SBATCH --time=2-0
#SBATCH --mem=2g

export PYTHONPATH=$PYTHONPATH:"$(dirname "$(pwd)")"/Marabou

python3 WatermarkVerification2.py --model mnist.w.wm --epsilon_max 100
# python3 WatermarkVerification2.py --model test --input_path ./test_data/test_images.npy --epsilon_max 100