#!/bin/bash


NUMBER_OF_TEST=5

MY_PARAM_FILE="/path/GraGOD/models/gcn/params_swat.yaml"
MY_GRAPH="/my_graph_path.csv"

sliding_windows=(5 10 20 30 40 50 60)
dims=(30 40 50 60 70 80 90)

layer_nums=(1 2 3 4 5 6 7 8 9)

i=0
for sliding_window in "${sliding_windows[@]}"; do
    for dim in "${dims[@]}"; do
        for layer_num in "${layer_nums[@]}"; do
            ((i++))
            echo "$i"
            echo $EDGE_PATH

            TRAIN_OUTPUT=$(nohup python models/train.py \
            --model gcn \
            --dataset swat \
            --params_file $MY_PARAM_FILE \
            --edge_path $EDGE_PATH \
            )

            FILE=$MY_PARAM_FILE
            NEW_VERSION=$i  # Change this to the desired version number

    # Use sed to replace the version number in ckpt_folder
            sed -i  "s/window_size: ${sliding_window}/" "$FILE"
            sed -i  "s/hidden_dim: ${dim}/" "$FILE"
            sed -i  "s/n_layers: ${layer_num}/" "$FILE"

            sed -i -E "s|(ckpt_folder: \"output/gcn/version_)[0-9]+(\")|\1${NEW_VERSION}\2|" "$FILE"



            # Test the model
            TEST_OUTPUT=$(nohup python models/predict.py \
            --model gcn \
            --dataset swat \
            --params_file $MY_PARAM_FILE \
            --edge_path $EDGE_PATH \
            >> results_gradgod.txt &)
        done
    done

done

FILE="results_gradgod.txt"

# Initialize an empty array
auc_scores=()

while IFS= read -r line; do
  # Check for the beginning of a "Test" section
  if [[ "$line" =~ "------- Test -------" ]]; then
    in_test_section=1  # We are in the Test section now
  fi

  # If we are in the "Test" section, look for "Auc" and store the value
  if [[ $in_test_section -eq 1 ]]; then
    # Look for "Auc" and extract the value next to it (numeric value)
    if [[ "$line" =~ "Auc" ]]; then
      # Extract numeric value after "Auc"
        echo "$line"  # Print the entire line
        numbers=($(echo "$line" | grep -o '[0-9.]\+'))

        # Print the extracted numbers
        echo "Extracted Numbers: ${numbers[@]}"
        # Extract numeric value after "Auc"
        auc_value=$(echo "$line" | awk -F '|' '{print $2}' | tr -d ' ')
        auc_scores+=("$numbers[@]")
    fi
  fi

  # If we reach another section (e.g., "Validation"), stop collecting AUC scores
  if [[ "$line" =~ "------- Val -------" ]]; then
    in_test_section=0  # No longer in the "Test" section
  fi
done < $FILE  # Replace with your actual file name

# Print all collected AUC scores
echo "AUC Scores for Test sets:"
for auc in "${auc_scores}"; do
    echo "$auc"
done

# Final best result
#echo "Best F1: $BEST_F1"
#echo "Best Parameters: $BEST_PARAMS"

