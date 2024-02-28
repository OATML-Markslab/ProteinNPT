source ./config.sh
source activate proteinnpt_env

export model_config_location=$Embeddings_Tranception_config_location #Path to model config [Embeddings_MSAT_config_location|Embeddings_Tranception_config_location|Embeddings_ESM1v_config_location]
export sequence_embeddings_folder=$Tranception_embeddings_indels_folder #Path to sequence embeddings [MSAT_embeddings_indels_folder|Tranception_embeddings_indels_folder|ESM1v_embeddings_indels_folder]

export target_config_location=$target_config_location_fitness
export assay_data_folder=$CV_indels_data_folder
export augmentation="zero_shot_fitness_predictions_covariate" #[Overwrite to "None" for ESM models as zero-shot fitness predictions are unavailable for indels | "zero_shot_fitness_predictions_covariate" for Tranception]

export fold_variable_name='fold_random_5'
export assay_index=0 #Replace with index of desired DMS assay in the ProteinGym reference file (`utils/proteingym`)
export model_name_suffix='All_indels_final' #Give a name to the model

python train.py \
    --data_location ${proteinnpt_data_path} \
    --assay_reference_file_location ${DMS_reference_file_path_indels} \
    --model_config_location ${model_config_location} \
    --target_config_location ${target_config_location} \
    --assay_data_folder ${assay_data_folder} \
    --augmentation ${augmentation} \
    --zero_shot_fitness_predictions_location ${zero_shot_fitness_predictions_indels} \
    --fold_variable_name ${fold_variable_name} \
    --assay_index ${assay_index} \
    --training_fp16 \
    --sequence_embeddings_folder ${sequence_embeddings_folder} \
    --model_name_suffix ${model_name_suffix} \
    --indel_mode