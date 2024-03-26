source ./config.sh
source activate proteinnpt_env

export DMS_index=0 #Replace with index of desired DMS assay in the ProteinGym reference file (`utils/proteingym`)

#########################################################################
########################Zero-shot MSA Transformer########################
#########################################################################

export model_checkpoint=$MSA_Transformer_location
export output_folder=$zero_shot_fitness_predictions_substitutions/MSA_Transformer
export scoring_strategy=masked-marginals
export model_type=MSA_Transformer
export random_seeds="1 2 3 4 5"

python zero_shot_fitness_esm.py \
    --model-location ${model_checkpoint} \
    --model_type ${model_type} \
    --dms_index ${DMS_index} \
    --dms_mapping ${DMS_reference_file_path_subs} \
    --dms-input ${CV_subs_singles_data_folder} \
    --dms-output ${output_folder} \
    --scoring-strategy ${scoring_strategy} \
    --msa-path ${DMS_MSA_data_folder} \
    --msa-weights-folder ${DMS_MSA_weights_folder} \
    --seeds ${random_seeds}

#########################################################################
#############################Zero-shot ESM1v#############################
#########################################################################

# Five checkpoints for ESM-1v
export model_checkpoint1=$ESM1v_location
export model_checkpoint2=$proteinnpt_data_path/ESM/ESM1v/esm1v_t33_650M_UR90S_2.pt
export model_checkpoint3=$proteinnpt_data_path/ESM/ESM1v/esm1v_t33_650M_UR90S_3.pt
export model_checkpoint4=$proteinnpt_data_path/ESM/ESM1v/esm1v_t33_650M_UR90S_4.pt
export model_checkpoint5=$proteinnpt_data_path/ESM/ESM1v/esm1v_t33_650M_UR90S_5.pt
export model_checkpoint="${model_checkpoint1} ${model_checkpoint2} ${model_checkpoint3} ${model_checkpoint4} ${model_checkpoint5}"
export output_folder=$zero_shot_fitness_predictions_substitutions/ESM1v
export model_type=ESM1v
export scoring_strategy=masked-marginals
export scoring_window=optimal

python zero_shot_fitness_esm.py \
    --model-location ${model_checkpoint} \
    --model_type ${model_type} \
    --dms_index ${DMS_index} \
    --dms_mapping ${DMS_reference_file_path_subs} \
    --dms-input ${CV_subs_singles_data_folder} \
    --dms-output ${output_folder} \
    --scoring-strategy ${scoring_strategy} \
    --scoring-window ${scoring_window}


#########################################################################
###############################Tranception###############################
#########################################################################

export checkpoint=$Tranception_location
export output_folder=$zero_shot_fitness_predictions_substitutions/Tranception

python zero_shot_fitness_tranception.py \
                --checkpoint ${checkpoint} \
                --DMS_reference_file_path ${DMS_reference_file_path_subs} \
                --DMS_data_folder ${CV_subs_singles_data_folder} \
                --DMS_index ${DMS_index} \
                --output_scores_folder ${output_folder} \
                --inference_time_retrieval \
                --MSA_folder ${DMS_MSA_data_folder} \
                --MSA_weights_folder ${DMS_MSA_weights_folder}


#########################################################################
###############################Merge scores##############################
#########################################################################

python ../proteinnpt/utils/merge_zero_shot.py \
                --DMS_reference_file_path ${DMS_reference_file_path_subs} \
                --DMS_mutants_folder ${CV_subs_singles_data_folder} \
                --zero_shot_scores_folder ${zero_shot_fitness_predictions_substitutions} \
                --DMS_index ${DMS_index}
