import os
import argparse
import pandas as pd

def main(args):
    mapping_protein_seq_DMS = pd.read_csv(args.DMS_reference_file_path)
    DMS_id=mapping_protein_seq_DMS["DMS_id"][args.DMS_index]
    print("Merging all zero-shot scores for DMS: "+str(DMS_id))
    DMS_file_name = mapping_protein_seq_DMS["DMS_filename"][mapping_protein_seq_DMS["DMS_id"]==DMS_id].values[0]
    DMS_data = pd.read_csv(args.DMS_mutants_folder + os.sep + DMS_file_name, low_memory=False)
    try:
        var_list = ['mutant','mutated_sequence','DMS_score','DMS_score_bin']
    except:
        var_list = ['mutant','mutated_sequence']
    merge = DMS_data[var_list]
    num_mutants = len(merge)
    score_name_mapping_original_names = {
        "Tranception" : "avg_score",
        "ESM1v" : "Ensemble_ESM1v",
        "MSA_Transformer" : "esm_msa1b_t12_100M_UR50S_ensemble",
        "DeepSequence": "evol_indices_ensemble",
        "TranceptEVE" : "avg_score"
    }
    score_name_mapping_clean_names = {
        "Tranception" : "Tranception_L",
        "ESM1v" : "ESM1v_ensemble",
        "MSA_Transformer" : "MSA_Transformer_ensemble",
        "DeepSequence": "DeepSequence_ensemble",
        "TranceptEVE": "TranceptEVE_L"
    }
    model_list = ["Tranception", "ESM1v", "MSA_Transformer", "DeepSequence", "TranceptEVE"] if not args.indel_mode else ["Tranception"]
    for model_name in model_list:
        print(model_name)
        model_score_file_path = args.zero_shot_scores_folder + os.sep + model_name + os.sep + DMS_file_name
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
    merge.to_csv(args.zero_shot_scores_folder + os.sep + DMS_file_name, index=False)

if __name__ == '__main__':
    print("Merging zero-shot predictions score files")
    parser = argparse.ArgumentParser(description='Merge scoring')
    parser.add_argument('--DMS_reference_file_path', type=str, help='Path to reference file with list of DMS to score')
    parser.add_argument('--DMS_mutants_folder', type=str, help='Path to folder that contains all mutant files to be scored')
    parser.add_argument('--DMS_index', type=int, help='Index of DMS assay in reference file')
    parser.add_argument('--zero_shot_scores_folder', type=str, help='Path to folder that contains all zero-shot model scores')
    parser.add_argument('--indel_mode', action='store_true', help='Flag to be used when scoring insertions and deletions. Otherwise assumes substitutions')
    args = parser.parse_args()
    main(args)