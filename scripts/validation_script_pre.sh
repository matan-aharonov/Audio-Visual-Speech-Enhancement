#!/bin/bash
# Activate virtual environment
source /cs/engproj/322/new_avse_venv/bin/activate
# Activate modules
source /cs/engproj/322/load_module.sh
# Preprocess validation data
python /cs/engproj/322/avse/speech_enhancer.py -bd /cs/engproj/322/out_preprocess_validation preprocess -dn yakov_validation -ds /cs/engproj/322/data -n /cs/engproj/322/raw_noise/libri-speech/validation -s yakov_validation
