# This config file contains the paths to all relevant data objects used by the ProteinNPT codebase. Only the next two lines should be updated based on your particular setup
export proteinnpt_data_path="Replace this string with the path to the folder where you downloaded the core ProteinNPT files (ProteinNPT_data.zip)"
export proteinnpt_repo_path="Replace this string with the path to the root of your local copy of the ProteinNPT folder"

# Reference files for substitution and indel assays
export DMS_reference_file_path_subs=$proteinnpt_repo_path/utils/proteingym/DMS_substitutions.csv
export DMS_reference_file_path_indels=$proteinnpt_repo_path/utils/proteingym/DMS_indels.csv
export DMS_reference_file_path_multi_objectives=$proteinnpt_repo_path/utils/proteingym/DMS_multi_objectives.csv

# Model config files
export ProteinNPT_config_location=$proteinnpt_repo_path/proteinnpt/model_configs/PNPT_final.json
export Embeddings_MSAT_config_location=$proteinnpt_repo_path/baselines/model_configs/Embeddings_MSAT_final.json
export Embeddings_Tranception_config_location=$proteinnpt_repo_path/baselines/model_configs/Embeddings_Tranception_final.json
export Embeddings_ESM1v_config_location=$proteinnpt_repo_path/baselines/model_configs/Embeddings_ESM1v_final.json
export OHE_config_location=$proteinnpt_repo_path/baselines/model_configs/OHE_not_augmented.json
export OHE_MSAT_config_location=$proteinnpt_repo_path/baselines/model_configs/OHE_MSAT_final.json
export OHE_Tranception_config_location=$proteinnpt_repo_path/baselines/model_configs/OHE_Tranception_final.json
export OHE_ESM1v_config_location=$proteinnpt_repo_path/baselines/model_configs/OHE_ESM1v_final.json
export OHE_DeepSequence_config_location=$proteinnpt_repo_path/baselines/model_configs/OHE_DeepSequence_final.json
export OHE_TranceptEVE_config_location=$proteinnpt_repo_path/baselines/model_configs/OHE_TranceptEVE_final.json

# Target config files
export target_config_location_fitness=$proteinnpt_repo_path/utils/target_configs/fitness.json
export target_config_location_multi_objectives_2=$proteinnpt_repo_path/utils/target_configs/fitness_multi_objectives_2.json
export target_config_location_multi_objectives_3=$proteinnpt_repo_path/utils/target_configs/fitness_multi_objectives_3.json

# Folders containing the mutated sequences and CV folds
export CV_subs_singles_data_folder=$proteinnpt_data_path/data/fitness/substitutions_singles #Folder containing single substitutions folds
export CV_subs_multiples_data_folder=$proteinnpt_data_path/data/fitness/substitutions_multiples #Folder containing multiple substitutions folds
export CV_indels_data_folder=$proteinnpt_data_path/data/fitness/indels #Folder containing indels fold

# Folders where embeddings are to be stored
export embeddings_subs_singles_data_folder=$proteinnpt_data_path/data/embeddings/substitutions_singles #Folder for single substitutions embeddings
export embeddings_subs_multiples_data_folder=$proteinnpt_data_path/data/embeddings/substitutions_multiples #Folder for multiple substitutions embeddings
export embeddings_indels_data_folder=$proteinnpt_data_path/data/embeddings/indels #Folder indels embeddings

# Folders containing multiple sequence alignments (MSAs) and MSA weights for all DMS assays
export DMS_MSA_data_folder=$proteinnpt_data_path/data/MSA/MSA_files #Folder containing DMS MSA files
export DMS_MSA_weights_folder=$proteinnpt_data_path/data/MSA/MSA_weights #Folder containing DMS MSA weights

# Folders contraining pretrained model checkpoints
export MSA_Transformer_location=$proteinnpt_data_path/ESM/MSA_Transformer/esm_msa1b_t12_100M_UR50S.pt #Path to MSA Transformer checkpoint
export Tranception_location=$proteinnpt_data_path/Tranception/Tranception_Large #Path to Tranception checkpoint
export ESM1v_location=$proteinnpt_data_path/ESM/ESM1v/esm1v_t33_650M_UR90S_1.pt #Path to ESM1v checkpoint

# Folders containing hhfilter and clustal omega utils
export path_to_hhfilter=$proteinnpt_data_path/utils/hhfilter #Path to hhfilter
export path_to_clustalomega=$proteinnpt_data_path/utils/clustal-omega #Path to clustal omega

# Embedding substitutions (singles) location
export MSAT_embeddings_folder=$proteinnpt_data_path/data/embeddings/substitutions_singles/MSA_Transformer #Path to MSA Transformer embeddings for substitution assays
export Tranception_embeddings_folder=$proteinnpt_data_path/data/embeddings/substitutions_singles/Tranception #Path to Tranception embeddings for substitution assays
export ESM1v_embeddings_folder=$proteinnpt_data_path/data/embeddings/substitutions_singles/ESM1v #Path to ESM1v embeddings for substitution assays

# Embedding indels location
export MSAT_embeddings_indels_folder=$proteinnpt_data_path/data/embeddings/indels/MSA_Transformer #Path to MSA Transformer embeddings for indel assays
export Tranception_embeddings_indels_folder=$proteinnpt_data_path/data/embeddings/indels/Tranception #Path to Tranception embeddings for indel assays
export ESM1v_embeddings_indels_folder=$proteinnpt_data_path/data/embeddings/indels/ESM1v #Path to ESM1v embeddings for indels assays

# Folder containing zero-shot fitness predictions
export zero_shot_fitness_predictions_substitutions=$proteinnpt_data_path/data/zero_shot_fitness_predictions/substitutions #Path to zero-shot predictions for substitution assays
export zero_shot_fitness_predictions_indels=$proteinnpt_data_path/data/zero_shot_fitness_predictions/indels #Path to zero-shot predictions for indel assays