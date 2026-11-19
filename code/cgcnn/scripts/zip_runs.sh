
set -euo pipefail
shopt -s nullglob



# ##### HYPERPARAM TUNING #####

# dirs=(out_mt_b32_mae_loss_[bgp][ilp]_hp*)

# valid_dirs=()
# for d in "${dirs[@]}"; do
#     [[ -d "$d" ]] && valid_dirs+=("$d")
# done

# if (( ${#valid_dirs[@]} == 0 )); then
#     echo "No hyperparam experiment directories found."
#     exit 1
# fi

# echo "Matching experiment directories:"
# printf '%s\n' "${valid_dirs[@]}"

# read -r -p "Create TAR.GZ archive? (y/N) " answer

# if [[ "$answer" == "y" ]]; then
#     tar -czvf zip_hyperparam.tar.gz "${valid_dirs[@]}"
#     echo "Archive successfully created."
# else
#     echo "Cancelled."
# fi


##### PERSLAY RUNS #####

dirs=(out_mt_b32_mae_loss_[bgp]p_s{42..46})

valid_dirs=()
for d in "${dirs[@]}"; do
    [[ -d "$d" ]] && valid_dirs+=("$d")
done

if (( ${#valid_dirs[@]} == 0 )); then
    echo "No perslay experiment directories found."
    exit 1
fi

echo "Matching experiment directories:"
printf '%s\n' "${valid_dirs[@]}"

read -r -p "Create TAR.GZ archive? (y/N) " answer

if [[ "$answer" == "y" ]]; then
    tar -czvf zip_perslay.tar.gz "${valid_dirs[@]}"
    echo "Archive successfully created."
else
    echo "Cancelled."
fi



# ##### MULTITASK MT TUNING #####

# dirs=(out_mt_dg*)

# valid_dirs=()
# for d in "${dirs[@]}"; do
#     [[ -d "$d" ]] && valid_dirs+=("$d")
# done

# if (( ${#valid_dirs[@]} == 0 )); then
#     echo "No multitask MT experiment directories found."
#     exit 1
# fi

# echo "Matching experiment directories:"
# printf '%s\n' "${valid_dirs[@]}"

# read -r -p "Create TAR.GZ archive? (y/N) " answer

# if [[ "$answer" == "y" ]]; then
#     tar -czvf zip_multitask_mt.tar.gz "${valid_dirs[@]}"
#     echo "Archive successfully created."
# else
#     echo "Cancelled."
# fi




# find . -mindepth 1 -maxdepth 1 -type d |
# sort > all_out_directories.txt

# tar -czvf mt_dg.tar.gz out_mt_dg*
