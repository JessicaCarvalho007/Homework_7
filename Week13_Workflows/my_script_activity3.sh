#!/bin/bash

# Creates the folder
mkdir new_folder

# Move to that folder
cd new_folder

# Copy scripts to this folder (The point means here. Syntax cp [what] [where])
cp ../session2_activity2/case_a/my_functions.py .
cp ../session2_activity2/case_a/current_project.py .

# Rename one file by moving it
mv current_project.py my_project.py

# Execute the main python script
python my_project.py

# Remove the functions script
rm my_functions.py