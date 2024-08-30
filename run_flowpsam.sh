#!/bin/bash
eval "$(conda shell.bash hook)"
source ~/.bashrc
conda deactivate
conda deactivate
conda activate flowsam
cd ~/flowsam
echo "Argument received: $1"
python evaluation.py --model=flowpsam --dataset=example --flow_gaps=1 \
--max_obj=10 --num_gridside=20 --ckpt_path=/home/jess/flowsam/data/frame_level_flowpsam_vitbvith_train_on_oclrsyn_dvs16.pth --save_path=/home/jess/flowsam/masks/flowpsam --dir_names=$1