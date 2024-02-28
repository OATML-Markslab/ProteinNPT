import argparse
import os,json
import pandas as pd
import warnings
import proteinnpt

from proteinnpt.utils import cv_split
from proteinnpt.utils.tranception.utils.scoring_utils import get_mutated_sequence
from proteinnpt.utils.msa_utils import MSA_processing
warnings.filterwarnings("ignore", category=UserWarning, module='transformers')

parser = argparse.ArgumentParser(description='End-to-end ProteinNPT pipeline on new assays')
###Required parameters
parser.add_argument('--assay_data_location', default=None, type=str, required=True, help='Path to assay data csv (expects a csv, with at least three columns: mutant or mutated_sequence | DMS_score | fold_variable_name)')
parser.add_argument('--target_seq', default=None, type=str, required=True, help='WT sequence mutated in the assay')
parser.add_argument('--proteinnpt_data_location', type=str, required=True, help='Path to core ProteinNPT datasets (e.g., MSA files, DMS assays, pretrained model checkpoints). Training output will also be stored there (i.e., checkpoints and test set predictions).')
###Optional parameters
#General
parser.add_argument('--indel_mode', action='store_true', help='Use this mode if wokring with indel assays')
parser.add_argument('--model_name_suffix', default='base_pipeline', type=str, help='Suffix to reference model')
parser.add_argument('--model_name', default="PNPT_MSAT_final", type=str, help='Name of model used for training and inference [Optional -- defaults to ProteinNPT]')
parser.add_argument('--fold_variable_name', default="train_test_split", type=str, help='Name of cross-validation fold variable name in the assay data (Optional -- defaults to "train_test_split" with 0 for train, 1 for test)')
parser.add_argument('--test_fold_index', default=1, type=int, help='Index of test fold (if "-1", we loop through all folds iteratively)')
parser.add_argument('--model_config_location', default=None, type=str, help='Path to main model config file that specifies all parameters as needed [Optional - if None, we default to the config from model_type]')
parser.add_argument('--target_config_location', default=None, type=str, help='Config file for assays to be used for modelings [Optional - if None, we default to single property prediction, and expects the variable name is DMS_score in the assay csv]')
parser.add_argument('--sequence_embeddings_folder', default=None, type=str, help='Location of stored embeddings on disk [Optional -- defaults to stadard embeddings location in data folder]')
parser.add_argument('--number_folds', default=5, type=int, help='Number of folds to create if fold_variable_name not in assay file')
#Embeddings
parser.add_argument('--MSA_location', default=None, type=str, help='Path to MSA file (expects .a2m)')
parser.add_argument('--max_positions', default=1024, type=int, help='Maximum context length of embedding model')
parser.add_argument('--batch_size', default=1, type=int, help='Eval batch size')
parser.add_argument('--embeddings_MSAT_num_MSA_sequences', default=384, type=int, help='Num MSA sequences to score each sequence with')
parser.add_argument('--MSA_sequence_weights_theta', default=0.2, type=float, help='Num MSA sequences to score each sequence with')
parser.add_argument('--MSA_start', default=None, type=int, help='Index of first AA covered by the MSA relative to target_seq coordinates (1-indexing)')
parser.add_argument('--MSA_end', default=None, type=int, help='Index of last AA covered by the MSA relative to target_seq coordinates (1-indexing)')
args = parser.parse_args()

default_config_mapping = {
    "PNPT_MSAT_final": "../proteinnpt/proteinnpt/model_configs/PNPT_final.json",
    "Embeddings_MSAT_final": "../proteinnpt/baselines/model_configs/Embeddings_MSAT_final.json",
    "Embeddings_Tranception_final": "../proteinnpt/baselines/model_configs/Embeddings_Tranception_final.json",
    "Embeddings_ESM1v_final": "../proteinnpt/baselines/model_configs/Embeddings_ESM1v_final.json",
    "OHE_MSAT_final": "../proteinnpt/baselines/model_configs/OHE_MSAT_final.json",
}
model_location_mapping = {
    "MSA_Transformer": args.proteinnpt_data_location + os.sep + "ESM/MSA_Transformer/esm_msa1b_t12_100M_UR50S.pt",
    "Tranception": args.proteinnpt_data_location + os.sep + "Tranception/Tranception_Large",
    "ESM1v": args.proteinnpt_data_location + os.sep + "ESM/ESM1v/esm1v_t33_650M_UR90S_1.pt",
    "ESM1v-2": args.proteinnpt_data_location + os.sep + "ESM/ESM1v/esm1v_t33_650M_UR90S_2.pt",
    "ESM1v-3": args.proteinnpt_data_location + os.sep + "ESM/ESM1v/esm1v_t33_650M_UR90S_3.pt",
    "ESM1v-4": args.proteinnpt_data_location + os.sep + "ESM/ESM1v/esm1v_t33_650M_UR90S_4.pt",
    "ESM1v-5": args.proteinnpt_data_location + os.sep + "ESM/ESM1v/esm1v_t33_650M_UR90S_5.pt"
}

if not args.model_config_location: args.model_config_location = default_config_mapping[args.model_name]
model_config = json.load(open(args.model_config_location))
if args.target_config_location is None: args.target_config_location = "../proteinnpt/utils/target_configs/fitness.json"
model_name = model_config['model_name_suffix'] if model_config['model_name_suffix']!="PNPT_MSAT_final" else "ProteinNPT"
DMS_id = args.assay_data_location.split(".csv")[0].split(os.sep)[-1]
print("Training {} model on the {} assay".format(model_name,DMS_id))

assay_data = pd.read_csv(args.assay_data_location,low_memory=False)
assert "mutant" in assay_data.columns or "mutated_sequence" in assay_data.columns, "Could not find mutant nor mutated_sequence columns in assay file"
if "mutated_sequence" not in assay_data.columns:
    assay_data["mutated_sequence"] = assay_data['mutant'].apply(lambda x: scoring_utils.get_mutated_sequence(args.target_seq, x))
if "mutant" not in assay_data.columns:
    assay_data['mutant'] = assay_data.index.apply(lambda x: "mutant_" + str(x))
if args.fold_variable_name not in assay_data.columns:
    print("fold_variable_name not found in assay file. Assigning {} folds at random".format(args.number_folds))
    assay_data = cv_split.create_folds_random(assay_data, n_folds=args.number_folds)
assay_data.to_csv(args.assay_data_location,index=False)

if args.sequence_embeddings_folder is None: args.sequence_embeddings_folder = args.proteinnpt_data_location + os.sep + "data" + os.sep + "embeddings"
embeddings_location = args.sequence_embeddings_folder + os.sep + model_config['aa_embeddings'] + os.sep + DMS_id + ".h5"
weight_file_name = DMS_id + '_theta_' + str(args.MSA_sequence_weights_theta) + ".npy"
if not os.path.exists(embeddings_location) and model_config['aa_embeddings']!="One_hot_encoding":
    print("#"*100+"\n Step1: Computing sequence embeddings for the {} assay \n".format(DMS_id)+"#"*100)
    #Sequence weights computation
    MSA_sequence_weights_location = args.proteinnpt_data_location + os.sep + "data/MSA/MSA_weights" + os.sep + weight_file_name
    if not os.path.exists(MSA_sequence_weights_location) and model_config['aa_embeddings'] in ["MSA_Transformer","Tranception"]: #Doing preprocessing for Tranception since used in zero-shot prediction (Step2)
        print("Computing MSA sequence weights")
        _ = MSA_processing(
                MSA_location=args.MSA_location,
                theta=args.MSA_sequence_weights_theta,
                use_weights=True,
                weights_location=MSA_sequence_weights_location
        )
    #Embeddings computation
    embeddings_run_parameters = "--model_type {} \
    --model_location {} \
    --input_data_location {} \
    --output_data_location {} \
    --batch_size {} \
    --max_positions {} \
    --target_seq {} ".format(
        model_config['aa_embeddings'],
        model_location_mapping[model_config['aa_embeddings']],
        args.assay_data_location,
        args.sequence_embeddings_folder,
        args.batch_size,
        args.max_positions,
        args.target_seq
        )
    if model_config['aa_embeddings']=="MSA_Transformer": 
        embeddings_run_parameters += "--num_MSA_sequences {} ".format(args.embeddings_MSAT_num_MSA_sequences)
        embeddings_run_parameters += "--MSA_location {} ".format(args.MSA_location)
        embeddings_run_parameters += "--MSA_weight_data_folder {} ".format(args.proteinnpt_data_location + os.sep + "data/MSA/MSA_weights")
        embeddings_run_parameters += "--weight_file_name {} ".format(weight_file_name)
        embeddings_run_parameters += "--path_to_hhfilter {} ".format(args.proteinnpt_data_location + os.sep + "utils/hhfilter")
    if args.MSA_start: embeddings_run_parameters += "--MSA_start {} ".format(args.MSA_start)
    if args.MSA_end: embeddings_run_parameters += "--MSA_end {} ".format(args.MSA_end)
    if args.indel_mode: embeddings_run_parameters += "--path_to_clustalomega {} --indel_mode ".format(args.proteinnpt_data_location + os.sep + "utils/clustal-omega")
    os.system("python embeddings.py "+embeddings_run_parameters)
elif model_config['aa_embeddings'] == "One_hot_encoding":
    print("#"*100+"\n Step1: Model uses OHE sequence representation - Skipping embedding computation \n"+"#"*100)
else:
    print("#"*100+"\n Step1: Found embeddings for the {} assay on disk \n".format(DMS_id)+"#"*100)


zero_shot_scores_folder = args.proteinnpt_data_location + os.sep + "data" + os.sep + "zero_shot_fitness_predictions"
zero_shot_scores_folder = zero_shot_scores_folder + os.sep + "indels" if args.indel_mode else zero_shot_scores_folder + os.sep + "substitutions"
zero_shot_fitness_predictions_location = zero_shot_scores_folder + os.sep + DMS_id + ".csv"
score_name_mapping_original_names = {
        "Tranception" : "avg_score",
        "ESM1v" : "Ensemble_ESM1v",
        "MSA_Transformer" : "esm_msa1b_t12_100M_UR50S_ensemble"
    }
score_name_mapping_clean_names = {
        "Tranception" : "Tranception_L",
        "ESM1v" : "ESM1v_ensemble",
        "MSA_Transformer" : "MSA_Transformer_ensemble"
    }
if os.path.exists(zero_shot_fitness_predictions_location):
    zero_shot_file = pd.read_csv(zero_shot_fitness_predictions_location)
    if model_config['aa_embeddings'] in score_name_mapping_clean_names:
        zero_shot_model_computed = score_name_mapping_clean_names[model_config['aa_embeddings']] in zero_shot_file.columns
    elif model_config['aa_embeddings']=="One_hot_encoding":
        zs_model_name = model_config["model_type"].split("_pred")[0]
        zero_shot_model_computed = zs_model_name in zero_shot_file.columns
    else:
        zero_shot_model_computed=False
if model_config['augmentation'] == "None" or (model_config['aa_embeddings'] in ["MSA_Transformer","ESM1v"] and args.indel_mode):
    print("#"*100+"\n Step2: No zero-shot fitness predictions used for the {} assay \n".format(DMS_id)+"#"*100)
elif not os.path.exists(zero_shot_fitness_predictions_location) or not zero_shot_model_computed:
    print("#"*100+"\n Step2: Computing zero-shot fitness predictions for the {} assay \n".format(DMS_id)+"#"*100)
    if model_config['aa_embeddings']=="MSA_Transformer":
        zero_shot_run_parameters = "--model-location {} \
            --model_type MSA_Transformer \
            --dms-input {} \
            --dms-output {} \
            --scoring-strategy masked-marginals \
            --msa-path {} \
            --target_seq {} \
            --msa-weights-folder {} \
            --weight_file_name {} \
            --seeds 1 2 3 4 5 ".format(
                model_location_mapping[model_config['aa_embeddings']],
                args.assay_data_location,
                zero_shot_scores_folder + os.sep + "MSA_Transformer",
                args.MSA_location,
                args.target_seq,
                args.proteinnpt_data_location + os.sep + "data/MSA/MSA_weights",
                weight_file_name
            )
        if args.MSA_start: zero_shot_run_parameters += "--MSA_start {} ".format(args.MSA_start)
        if args.MSA_end: zero_shot_run_parameters += "--MSA_end {} ".format(args.MSA_end)
        os.system("python zero_shot_fitness_esm.py "+zero_shot_run_parameters)
    elif model_config['aa_embeddings']=="ESM1v":
        zero_shot_run_parameters = "--model-location {} {} {} {} {} \
            --model_type ESM1v \
            --dms-input {} \
            --dms-output {} \
            --target_seq {} \
            --scoring-strategy masked-marginals \
            --scoring-window optimal".format(
                model_location_mapping["ESM1v"],
                model_location_mapping["ESM1v-2"],
                model_location_mapping["ESM1v-3"],
                model_location_mapping["ESM1v-4"],
                model_location_mapping["ESM1v-5"],
                args.assay_data_location,
                zero_shot_scores_folder + os.sep + "ESM1v",
                args.target_seq
            )
        os.system("python zero_shot_fitness_esm.py "+zero_shot_run_parameters)
    elif model_config['aa_embeddings']=="Tranception":
        DMS_data_folder = os.sep.join(args.assay_data_location.split(os.sep)[:-1])
        DMS_file_name = args.assay_data_location.split(os.sep)[-1]
        MSA_data_folder = os.sep.join(args.MSA_location.split(os.sep)[:-1])
        MSA_filename = args.MSA_location.split(os.sep)[-1]
        zero_shot_run_parameters = "--checkpoint {} \
            --DMS_data_folder {} \
            --DMS_file_name {} \
            --output_scores_folder {} \
            --inference_time_retrieval \
            --target_seq {} \
            --MSA_folder {} \
            --MSA_filename {} \
            --MSA_weights_folder {} \
            --MSA_weight_file_name {} ".format(
                model_location_mapping["Tranception"],
                DMS_data_folder,
                DMS_file_name,
                zero_shot_scores_folder + os.sep + "Tranception",
                args.target_seq,
                MSA_data_folder,
                MSA_filename,
                args.proteinnpt_data_location + os.sep + "data/MSA/MSA_weights",
                weight_file_name
            )
        if args.MSA_start: zero_shot_run_parameters += "--MSA_start {} ".format(args.MSA_start)
        if args.MSA_end: zero_shot_run_parameters += "--MSA_end {} ".format(args.MSA_end)
        if args.indel_mode: zero_shot_run_parameters += "--indel_mode --clustal_omega_location {} ".format(args.proteinnpt_data_location + os.sep + "utils/clustal-omega")
        os.system("python zero_shot_fitness_tranception.py "+zero_shot_run_parameters)
    #Merge zero-shot files
    merge = assay_data.copy()[['mutant','mutated_sequence','DMS_score']]
    num_mutants = len(merge)
    model_list = ["Tranception", "ESM1v", "MSA_Transformer"] if not args.indel_mode else ["Tranception"]
    for model_name in model_list:
        model_score_file_path = zero_shot_scores_folder + os.sep + model_name + os.sep + DMS_id + ".csv"
        if os.path.exists(model_score_file_path):
            scores = pd.read_csv(model_score_file_path,low_memory=False)
            if 'mutated_sequence' in scores: 
                scores = scores[['mutated_sequence',score_name_mapping_original_names[model_name]]].drop_duplicates()
                merge_key = 'mutated_sequence'
            else:
                scores = scores[['mutant',score_name_mapping_original_names[model_name]]].drop_duplicates()
                merge_key = 'mutant'
            scores.columns = [merge_key, score_name_mapping_clean_names[model_name]]
            merge = pd.merge(merge,scores,on=merge_key,how='inner')
            assert num_mutants==len(merge), "Some mutants were dropped during the merge: {} (pre-merge) vs {} (post-merge)".format(num_mutants,len(merge))
    merge.to_csv(zero_shot_fitness_predictions_location, index=False)

else:
    print("#"*100+"\n Step2: Found zero-shot fitness predictions for the {} assay on disk \n".format(DMS_id)+"#"*100)


print("#"*100+"\n Step3: Training the {} model on the {} assay \n".format(model_name,DMS_id)+"#"*100)
training_run_parameters = "--data_location {} \
    --assay_data_location {} \
    --model_config_location {} \
    --fold_variable_name {} \
    --target_config_location {} \
    --zero_shot_fitness_predictions_location {} \
    --training_fp16 \
    --sequence_embeddings_folder {} \
    --MSA_location {} \
    --MSA_sequence_weights_filename {} \
    --target_seq {} \
    --test_fold_index {} \
    --model_name_suffix {} ".format(
        args.proteinnpt_data_location,
        args.assay_data_location,
        args.model_config_location,
        args.fold_variable_name,
        args.target_config_location,
        zero_shot_scores_folder,
        args.sequence_embeddings_folder + os.sep + model_config['aa_embeddings'],
        args.MSA_location,
        weight_file_name,
        args.target_seq,
        args.test_fold_index,
        args.model_name_suffix   
    )
if args.MSA_start: training_run_parameters += "--MSA_start {} ".format(args.MSA_start)
if args.MSA_end: training_run_parameters += "--MSA_end {} ".format(args.MSA_end)    
if args.indel_mode: 
    if model_config['aa_embeddings'] in ['ESM1v','MSA_Transformer']: training_run_parameters += "--augmentation {} ".format("None")
    training_run_parameters += "--indel_mode"
os.system("python train.py "+training_run_parameters)