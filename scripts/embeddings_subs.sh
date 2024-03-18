source ./config.sh
conda activate proteinnpt_env

export assay_index=0 # Replace with index of desired DMS assay in the ProteinGym reference file (`utils/proteingym`)
export batch_size=1
export max_positions=1024

export model_type='MSA_Transformer' # [MSA_Transformer|Tranception|ESM1v]
export model_location=$MSA_Transformer_location # [MSA_Transformer_location|Tranception_location|ESM1v_location]
export num_MSA_sequences=384 # Used in MSA Transformer only
export embeddings_folder=$embeddings_subs_singles_data_folder/$model_type

python embeddings.py \
    --assay_reference_file_location ${DMS_reference_file_path_subs} \
    --assay_index ${assay_index} \
    --model_type ${model_type} \
    --model_location ${model_location} \
    --input_data_location ${CV_subs_singles_data_folder} \
    --output_data_location ${embeddings_folder} \
    --batch_size ${batch_size} \
    --max_positions ${max_positions} \
    --num_MSA_sequences ${num_MSA_sequences} \
    --MSA_data_folder ${DMS_MSA_data_folder} \
    --MSA_weight_data_folder ${DMS_MSA_weights_folder} \
    --path_to_hhfilter ${path_to_hhfilter}