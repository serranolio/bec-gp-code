#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Parameter arrays
detuning_start=-5000
detuning_end=1500
lattice_strength_start=0.2
lattice_strength_end=0.2
ramp_times=($(seq -f "%.0f" 10 10 100))
samples=($(seq 0 1))

# Files to copy
files=(execute_simulation.py parameters.py discrete_hankel_transform.py system_functions.py)
for s in "${samples[@]}"; do
    (
    for t in "${ramp_times[@]}"; do
        (
            dir_name="simulation_sample_${s}_delta_i_${detuning_start}_delta_f_${detuning_end}_omega_l_i_${lattice_strength_start}_omega_l_f_${lattice_strength_end}_ramp_time_${t}ms"
            mkdir -p "$dir_name"

            cp "${files[@]}" "$dir_name/"

            sed -i "s|^sample *=.*|sample = ${s}|" "$dir_name/execute_simulation.py"
            sed -i "s|^omega_l_i *=.*|omega_l_i = ${lattice_strength_start}|" "$dir_name/execute_simulation.py"
            sed -i "s|^omega_l_f *=.*|omega_l_f = ${lattice_strength_end}|" "$dir_name/execute_simulation.py"
	    sed -i "s|^delta_i *=.*|delta_i = ${detuning_start} / f_recoil|" "$dir_name/execute_simulation.py"
            sed -i "s|^delta_f *=.*|delta_f = ${detuning_end} / f_recoil|" "$dir_name/execute_simulation.py"
            sed -i "s|^ramp_time_ms *=.*|ramp_time_ms = ${t}|" "$dir_name/execute_simulation.py"
        ) &

    done
    ) &

done
wait  # wait for any remaining jobs

