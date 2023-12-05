#!/bin/bash

# navigate to the directory where your npy files are located
cd /dcs05/ciprian/smart/farahani/SL-CHA/ts/REST2_RL_aligned/

# loop through each npy file in the directory and rename it
for file in *.npy
do
    # extract the subject ID from the file name
    sbj_ID=$(echo $file | grep -oE '_[0-9]+' | grep -oE '[0-9]+')
    # rename the file with the desired format
    mv $file "REST2_RL_CHA_${sbj_ID}.npy"
done

