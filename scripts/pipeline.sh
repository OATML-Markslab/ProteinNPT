source config.sh  # To get $proteinnpt_data_path
conda activate proteinnpt_env
#source $proteinnpt_data_path/proteinnpt_env/bin/activate # Uncomment if using python venv instead of conda env

export assay_data_location="Replace this string with the path to the assay data"
export MSA_location="Replace this string with the path to the MSA data"
export target_seq="Replace this string with the wild type sequence mutated in the assay"

python pipeline.py \
    --proteinnpt_data_location ${proteinnpt_data_path} \
    --assay_data_location ${assay_data_location} \
    --MSA_location ${MSA_location} \
    --target_seq ${target_seq} \
    --fold_variable_name fold_random_5 \
    --test_fold_index 0
