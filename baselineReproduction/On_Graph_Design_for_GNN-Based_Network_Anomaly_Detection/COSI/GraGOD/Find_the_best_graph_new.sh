#!/bin/bash

#unfinished version of new way to use several graphs...

NUMBER_OF_TEST=5

MY_PARAM_FILE="/path/GraGOD/models/gcn/params_swat.yaml"
MY_GRAPH="/path_for_random_matrix/random_swat/adj"



# Create three arrays
declare -A lists

lists[0]="10 90 3" # kalof params
lists[1]="20 70 4" #mb params
lists[2]="10 60 7" #cosi param

version=0


# Iterate over all combinations
for j in 0 1 2; do

    #sanity check
    echo "List $i: ${lists[$i]}"
    first_element=(${lists[$i]})  
    echo "${first_element[0]}" 


  for i in $(seq 0 "$NUMBER_OF_TEST"); do
      echo "$i"
      EDGE_PATH="${MY_GRAPH}${i}.csv"
      echo $EDGE_PATH

      TRAIN_OUTPUT=$(nohup python models/train.py \
      --model gcn \
      --dataset swat \
      --params_file $MY_PARAM_FILE \
      --edge_path $EDGE_PATH \
      )
      #"$EDGE_PATH" \       
      # Define the file and new version number
      FILE=$MY_PARAM_FILE
      NEW_VERSION=$version   #$i  # Change this to the desired version number

      # Use sed to replace the version number in ckpt_folder
      sed -i  "s/window_size: ${lists}/" "$FILE"
      sed -i  "s/hidden_dim: ${dim}/" "$FILE"
      sed -i  "s/n_layers: ${layer_num}/" "$FILE"
      # Use sed to replace the version number in ckpt_folder
      sed -i -E "s|(ckpt_folder: \"output/gdn/version_)[0-9]+(\")|\1${NEW_VERSION}\2|" "$FILE"



      # Test the model
      TEST_OUTPUT=$(nohup python models/predict.py \
      --model gcn \
      --dataset swat \
      --params_file $MY_PARAM_FILE \
      --edge_path $EDGE_PATH \
      >> results_gradgod.txt &)

      version++
  done



done
