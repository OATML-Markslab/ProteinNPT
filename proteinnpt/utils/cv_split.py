import os
import random
import numpy as np
import pandas as pd
import argparse

def return_position_single(mutation):
    """Note: Only works for single mutations"""
    position = mutation.split(":")[0][1:-1]
    return int(position)

def keep_singles(DMS, mutant_column='mutant'):
    DMS = DMS[~DMS[mutant_column].str.contains(":")]
    return DMS

def assign_group_stratified(x):
    n_groups = 5 # number of groups to assign
    n_items = len(x) # number of items in the current group
    group_size = int(np.ceil(n_items / n_groups)) # target size of each group
    group_numbers = np.repeat(np.arange(0, n_groups), group_size)[:n_items] # generate group numbers
    np.random.shuffle(group_numbers) # randomly shuffle the group numbers
    return group_numbers

def create_folds_random(DMS, n_folds=5, mutant_column='mutant'):
    column_name = 'fold_random_{}'.format(n_folds)
    try:
        mutated_region_list = DMS[mutant_column].apply(lambda x: return_position_single(x)).unique()
    except:
        print("Mutated region not found from 'mutant' variable -- assuming the full protein sequence is mutated")
        mutated_region_list = range(len(DMS['mutated_sequence'].values[0]))
    len_mutated_region = len(mutated_region_list)
    if len_mutated_region < n_folds:
        raise Exception("Error, there are fewer mutated regions than requested folds")
    DMS[column_name] = np.random.randint(0, n_folds, DMS.shape[0])
    print(DMS[column_name].value_counts())
    return DMS

def create_folds_random_multiples(DMS):
    """Stratifed sampling by mutational depth"""
    DMS['mutation_depth']=DMS['mutant'].apply(lambda x: len(x.split(":")))
    DMS['fold_rand_multiples'] = DMS.groupby('mutation_depth')['mutant'].apply(assign_group_stratified).explode().values
    return DMS

def create_folds_by_position_modulo(DMS, n_folds=5, mutant_column='mutant'):
    column_name = 'fold_modulo_{}'.format(n_folds)
    mutated_region_list = sorted(DMS[mutant_column].apply(return_position_single).unique())
    len_mutated_region = len(mutated_region_list)
    if len_mutated_region < n_folds:
        raise Exception("Error, there are fewer mutated regions than requested folds")
    position_to_fold = {pos: i % n_folds for i, pos in enumerate(mutated_region_list)}
    DMS[column_name] = DMS[mutant_column].apply(lambda x: position_to_fold[return_position_single(x)])
    print(DMS[column_name].value_counts())
    return DMS

def create_folds_by_contiguous_position_discontiguous(DMS, n_folds=5, mutant_column='mutant'):
    column_name = 'fold_contiguous_{}'.format(n_folds)
    mutated_region_list = sorted(DMS[mutant_column].apply(lambda x: return_position_single(x)).unique())
    len_mutated_region = len(mutated_region_list)
    k, m = divmod(len_mutated_region, n_folds)
    folds = [[i] * k + [i] * (i < m) for i in range(n_folds)]
    folds = [item for sublist in folds for item in sublist]
    folds_indices = dict(zip(mutated_region_list, folds))
    if len_mutated_region < n_folds:
        raise Exception("Error, there are fewer mutated regions than requested folds")
    DMS[column_name] = DMS[mutant_column].apply(lambda x: folds_indices[return_position_single(x)])
    print(DMS[column_name].value_counts())
    return DMS

def create_folds_loop(reference_file_location = None, fold_list=[5], dir_path = './ProteinGym/DMS_assays/substitutions', saving_folder = './ProteinGym/data/fitness', DMS_score_name = 'DMS_score', DMS_score_bin_name = 'DMS_score_bin'):
    ProteinGym_ref =  pd.read_csv(reference_file_location, low_memory=False)
    list_errors = []
    for DMS_id, DMS_has_multiples in zip(ProteinGym_ref.DMS_id, ProteinGym_ref.includes_multiple_mutants):
        try:
            if DMS_has_multiples:
                # Remove multiples if present in file
                print(DMS_id)
                DMS_path = dir_path + os.sep + DMS_id + ".csv"
                DMS = pd.read_csv(DMS_path, low_memory=False)
                DMS = create_folds_random_multiples(DMS)
                DMS_saving_path = saving_folder + os.sep + 'multiples/' + DMS_id + ".csv"
                DMS.to_csv(DMS_saving_path, index=False)
                DMS = keep_singles(DMS)
                DMS = DMS[['mutant','mutated_sequence',DMS_score,DMS_score_bin]] if DMS_score_bin_name else DMS[['mutant','mutated_sequence',DMS_score]]
            else:
                DMS_path = dir_path + os.sep + DMS_id + ".csv"
                DMS = pd.read_csv(DMS_path, low_memory=False)

            for n_folds in fold_list:
                try:
                    DMS = create_folds_by_position_modulo(DMS, n_folds=n_folds)
                    DMS = create_folds_random(DMS, n_folds=n_folds)
                    DMS = create_folds_by_contiguous_position_discontiguous(DMS, n_folds=n_folds)
                except Exception as e: 
                    DMS = create_folds_by_position_modulo(DMS, n_folds=4)
                    DMS = create_folds_random(DMS, n_folds=4)
                    DMS = create_folds_by_contiguous_position_discontiguous(DMS, n_folds=4)
                    DMS = DMS.rename(columns={"fold_modulo_4": "fold_modulo_5", "fold_random_4": "fold_random_5", "fold_contiguous_4": "fold_contiguous_5"})
                    DMS = DMS.loc[:,~DMS.columns.duplicated()].copy()
                    print("Error with " + DMS_id)
                    list_errors.append(DMS_id)
                    print(e)
            DMS_saving_path = saving_folder + os.sep + DMS_id + ".csv"
            DMS=DMS[['mutant','mutated_sequence',DMS_score,DMS_score_bin,'fold_random_5','fold_modulo_5','fold_contiguous_5']] if DMS_score_bin_name else DMS[['mutant','mutated_sequence',DMS_score,'fold_random_5','fold_modulo_5','fold_contiguous_5']]
            DMS.to_csv(DMS_saving_path, index=False)
        except:
            print("Error with creating the CV schemes for assay: {}".format(DMS_id))
    return list_errors

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create CV folds on DMS assays')
    parser.add_argument('--reference_file_location', default="/users/pastin/projects/open_source/ProteinNPT/proteingym/DMS_substitutions.csv", type=str, help='Path to reference file with list of assays to score')
    parser.add_argument('--input_DMS_files_location', default='/scratch/pastin/protein/ProteinNPT/ProteinGym/DMS_assays/substitutions/Tsubo', type=str, help='Location of ProteinGym DMS files')
    parser.add_argument('--output_DMS_files_location', default='/scratch/pastin/protein/ProteinNPT/ProteinGym/DMS_assays/substitutions/Tsubo/cv_folds', type=str, help='Location of processed DMS files with cross-validation folds included')
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    list_errors = create_folds_loop(
        reference_file_location = args.reference_file_location, 
        dir_path = args.input_DMS_files_location,
        saving_folder = args.output_DMS_files_location,
        fold_list = [5]
    )

    print(list_errors)