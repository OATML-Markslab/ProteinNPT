source ./config.sh
source activate proteinnpt_env

export model_config_location=$ProteinNPT_config_location # [ProteinNPT_config_location|Embeddings_MSAT_config_location|Embeddings_Tranception_config_location|Embeddings_ESM1v_config_location|OHE_config_location|OHE_TranceptEVE_config_location]
export sequence_embeddings_folder=$MSAT_embeddings_folder # [MSAT_embeddings_folder|Tranception_embeddings_folder|ESM1v_embeddings_folder]

export fold_variable_name='fold_random_5' #[fold_random_5 | fold_contiguous_5 | fold_modulo_5]
export assay_index=1 #Replace with index of desired DMS assay in the ProteinGym reference file (`utils/proteingym`)
export model_name_suffix='All_singles_final' #Give a name to the model

python train.py \
    --data_location ${proteinnpt_data_path} \
    --assay_reference_file_location ${DMS_reference_file_path_subs} \
    --model_config_location ${model_config_location} \
    --fold_variable_name ${fold_variable_name} \
    --assay_index ${assay_index} \
    --target_config_location ${target_config_location_fitness} \
    --zero_shot_fitness_predictions_location ${zero_shot_fitness_predictions_substitutions} \
    --training_fp16 \
    --sequence_embeddings_folder ${sequence_embeddings_folder} \
    --model_name_suffix ${model_name_suffix}