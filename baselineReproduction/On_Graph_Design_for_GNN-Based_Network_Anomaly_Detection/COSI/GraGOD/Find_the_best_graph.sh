#!/bin/bash


NUMBER_OF_TEST=2

MY_PARAM_FILE="/path/GraGOD/models/gcn/params_swat.yaml"


# Iterate over all combinations
for i in $(seq 0 "$NUMBER_OF_TEST"); do
    echo "$i"

    TRAIN_OUTPUT=$(python models/train.py \
     --model gcn \
     --dataset swat \
     --params_file $MY_PARAM_FILE \
      --edge_path /path_for_edges/edges_index_swat.txt \
    )
            
    # Define the file and new version number
    FILE=$MY_PARAM_FILE
    NEW_VERSION=$i  # Change this to the desired version number

    # Use sed to replace the version number in ckpt_folder
    sed -i -E "s|(ckpt_folder: \"output/gdn/version_)[0-9]+(\")|\1${NEW_VERSION}\2|" "$FILE"



    # Test the model
    TEST_OUTPUT=$(python models/predict.py \
     --model gcn \
     --dataset swat \
     --params_file $MY_PARAM_FILE \
     --edge_path /path_for_edges/edges_index_swat.txt \
    >> results_gradgod.txt &)


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
        #auc_value=$(echo "$line" | awk -F '|' '{print $2}' | tr -d ' ')
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
for auc in "${auc_scores[@]}"; do
    echo "$auc"
done

# Final best result
#echo "Best F1: $BEST_F1"
#echo "Best Parameters: $BEST_PARAMS"
