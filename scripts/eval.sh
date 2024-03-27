source ./config.sh
source activate proteinnpt_env

export model_location="Path to model checkpoint (e.g., $DATA_PATH/checkpoint/model_name_BLAT_ECOLX_Jacquier_2013_fold-1/final/checkpoint.t7)"
export assay_data_location="Path to assay file with train/test sequences (e.g., $CV_subs_singles_data_folder/BLAT_ECOLX_Jacquier_2013.csv)" #DMS_score for test points may be nan
export embeddings_location="Path to train/test sequences embeddings (e.g., $MSAT_embeddings_folder/BLAT_ECOLX_Jacquier_2013.h5)"
export zero_shot_fitness_predictions_location="Path to train/test zero-shot fitness predictions (e.g., $zero_shot_fitness_predictions_substitutions/BLAT_ECOLX_Jacquier_2013.csv)"
export fold_variable_name='fold_random_5' #name of fold variable in assay data that identifies train vs test folds
export test_fold_index=1 #Index of test fold
export output_scores_location="Path to file where test scores should be stored (eg., $DATA_PATH/model_predictions/BLAT_ECOLX_Jacquier_2013_scores.csv)"

python eval.py \
    --model_location ${model_location} \
    --assay_data_location ${assay_data_location} \
    --embeddings_location ${embeddings_location} \
    --zero_shot_fitness_predictions_location ${zero_shot_fitness_predictions_location} \
    --fold_variable_name ${fold_variable_name} \
    --test_fold_index ${test_fold_index} \
    --output_scores_location ${output_scores_location}