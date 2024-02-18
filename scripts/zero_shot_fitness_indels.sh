source ./config.sh
source activate proteinnpt_env

#########################################################################
###############################Tranception###############################
#########################################################################

export checkpoint=$Tranception_location
export output_scores_folder=$zero_shot_fitness_predictions_indels/Tranception

export DMS_index=0 #Replace with index of desired DMS assay in the ProteinGym reference file (`utils/proteingym`)
export clustal_omega_location=$path_to_clustalomega
# Leveraging retrieval when scoring indels require batch size of 1 (no retrieval can use any batch size fitting in memory)
export batch_size_inference=1

python zero_shot_fitness_tranception.py \
                --checkpoint ${checkpoint} \
                --batch_size_inference ${batch_size_inference} \
                --DMS_reference_file_path ${DMS_reference_file_path_indels} \
                --DMS_data_folder ${CV_indels_data_folder} \
                --DMS_index ${DMS_index} \
                --output_scores_folder ${output_scores_folder} \
                --indel_mode \
                --clustal_omega_location ${clustal_omega_location} \
                --inference_time_retrieval \
                --MSA_folder ${DMS_MSA_data_folder} \
                --MSA_weights_folder ${DMS_MSA_weights_folder} \
                --scoring_window 'optimal'

#########################################################################
###############################Merge scores##############################
#########################################################################

python ../proteinnpt/utils/merge_zero_shot.py \
                --DMS_reference_file_path ${DMS_reference_file_path_indels} \
                --DMS_mutants_folder ${CV_indels_data_folder} \
                --zero_shot_scores_folder ${zero_shot_fitness_predictions_indels} \
                --DMS_index ${DMS_index} \
                --indel_mode
