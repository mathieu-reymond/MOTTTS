#!/bin/bash

experiments_folder="./experiments"
results_folder="$experiments_folder/results"
plot_folder="$experiments_folder/plots"

mkdir -p "$plot_folder"

if [ $# -eq 0 ]; then
    extension="png"
else
    extension=$1
fi

# ./Scripts/plotResults.py --output="$plot_folder/nodes_all.$extension"  "$results_folder" nodes --nodes=11 --timesteps=10000 scql mauce rnd llr

./Scripts/plotResults.py --output="$plot_folder/uniring.$extension" "$results_folder" uniring rnd rnd2 lp
