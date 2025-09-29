#!/bin/bash

# Get the directory where this script is located
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Initialize the array
list_of_directories=()

# Populate the array with directories starting with 'simulation_'
for dir in simulation_*/; do
    [ -d "$dir" ] && list_of_directories+=("$dir")
done

# Optional: print the array to verify
for d in "${list_of_directories[@]}"; do
    echo "$d"
done


