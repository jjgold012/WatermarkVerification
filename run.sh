#!/bin/bash
# A simple script to run the verification jobs

export PYTHONPATH=$PYTHONPATH:"$(dirname "$(pwd)")"/Marabou

exec python3 WatermarkVerification2.py --model mnist.w.wm --epsilon_max 100