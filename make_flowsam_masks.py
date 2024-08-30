import os
import subprocess

# Define the base directory containing the directories you want to iterate over
base_dir = "/home/jess/TaskSeg/tsg/jpgs"

# Define the common part of the command
base_command = (
    "python evaluation.py --model=flowisam --dataset=example --max_frame_gap=9 "
    "--max_obj=10 --num_gridside=20 "
    "--ckpt_path=/home/jess/Downloads/frame_level_flowisam_vitb_train_on_oclrsyn_dvs16.pth "
    "--save_path=/home/jess/flowsam/masks/flowisam"
)

# Iterate through each directory in the base directory
for dir_name in os.listdir(base_dir):
    dir_path = os.path.join(base_dir, dir_name)
    
    if os.path.isdir(dir_path):
        # Create the full command with the current directory name
        full_command = f"{base_command} --dir_names={dir_name}"
        
        # Print the command to verify (optional)
        print(f"Running command: {full_command}")
        
        # Run the command
        subprocess.run(full_command, shell=True, check=True)
