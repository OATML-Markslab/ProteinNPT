import os,sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch
import h5py
from datasets import Dataset
from proteinnpt.utils.tranception.utils.scoring_utils import get_mutated_sequence

def standardize(x, epsilon = 1e-10):
    return (x - x.mean()) / (x.std() + epsilon)

def split_data_based_on_test_fold_index(dataframe, fold_variable_name = 'fold_modulo_5', test_fold_index=None, use_validation_set=True):
    unique_folds = np.unique(dataframe[fold_variable_name])
    num_distinct_folds = len(unique_folds)
    if fold_variable_name=="fold_multiples":
        train = dataframe[dataframe[fold_variable_name]==0]
        if use_validation_set:
            num_mutations_train = int(len(train) * 0.8)
            val = train[num_mutations_train+1:]
            train = train[:num_mutations_train+1]
        else:
            val = None
        test = dataframe[dataframe[fold_variable_name]==1]
    else:
        if use_validation_set:
            test = dataframe[dataframe[fold_variable_name]==test_fold_index]
            val_fold_index = (test_fold_index - 1) % num_distinct_folds
            val = dataframe[dataframe[fold_variable_name]==val_fold_index]
            train = dataframe[ ~ dataframe[fold_variable_name].isin([test_fold_index, val_fold_index])]
        else:
            train = dataframe[dataframe[fold_variable_name]!=test_fold_index]
            val = None
            test = dataframe[dataframe[fold_variable_name]==test_fold_index]
    del train[fold_variable_name]
    if use_validation_set: del val[fold_variable_name]
    del test[fold_variable_name]
    return train, val, test

def cleanup_ids_assay_data(df, indel_mode=False, target_seq=None):
    assert 'mutated_sequence' in df.columns or 'mutant' in df.columns, "assay did not reference mutant nor mutated_sequence"
    if 'mutated_sequence' not in df: 
        df['mutated_sequence'] = df['mutant'].apply(lambda x: get_mutated_sequence(target_seq, x))
    if 'mutant' not in df: 
        if not indel_mode and target_seq is not None: #If substitutions assay, we reconstruct the mutants by comparing the mutated sequences with the target sequence
            df['mutant'] = df["mutated_sequence"].apply(lambda x: ':'.join([wt + str(i+1) + mut for i, (wt, mut) in enumerate(zip(target_seq, x)) if wt != mut]))
        else: #If indels we default to dummy mutant names
            df['mutant'] = df.index.to_series().apply(lambda x: "mutant_" + str(x))
    return df

def get_train_val_test_data(args, assay_file_names):
    target_names = args.target_config.keys() 
    assay_data={}
    merge = None
    main_target_name = None
    main_target_name_count = 0
    for target in target_names:
        if args.target_config[target]["main_target"]: 
            main_target_name=target
            main_target_name_count+=1
    assert main_target_name is not None, "No main target referenced. Please update config to select a unique main target."
    assert main_target_name_count <= 1, "Several main targets referenced. Please update config to select a unique main target."
    
    assay_data[main_target_name] = pd.read_csv(args.target_config[main_target_name]["location"] + os.sep + assay_file_names[main_target_name]) 
    assay_data[main_target_name] = cleanup_ids_assay_data(assay_data[main_target_name])[['mutant','mutated_sequence',args.target_config[main_target_name]["var_name"],args.fold_variable_name]]
    assay_data[main_target_name].columns = ['mutant','mutated_sequence', main_target_name, args.fold_variable_name]
    merge = assay_data[main_target_name]
    
    for target_name in target_names:
        if target_name!=main_target_name:
            assay_data[target_name] = pd.read_csv(args.target_config[target_name]["location"] + os.sep + assay_file_names[target_name]) 
            assay_data[target_name] = cleanup_ids_assay_data(assay_data[target_name])[['mutant',args.target_config[target_name]["var_name"]]]
            assay_data[target_name].columns = ['mutant',target_name]
            merge = pd.merge(merge, assay_data[target_name], how='outer', on='mutant')
            
    if args.augmentation=="zero_shot_fitness_predictions_covariate":
        zero_shot_fitness_predictions = pd.read_csv(args.zero_shot_fitness_predictions_location + os.sep + assay_file_names[main_target_name])
        zero_shot_fitness_predictions = cleanup_ids_assay_data(zero_shot_fitness_predictions)[['mutant',args.zero_shot_fitness_predictions_var_name]]
        zero_shot_fitness_predictions.columns = ['mutant','zero_shot_fitness_predictions']
        zero_shot_fitness_predictions['zero_shot_fitness_predictions'] = standardize(zero_shot_fitness_predictions['zero_shot_fitness_predictions'])
        merge = pd.merge(merge,zero_shot_fitness_predictions,how='left',on='mutant')

    train_val_test_splits = split_data_based_on_test_fold_index(
        dataframe = merge, 
        fold_variable_name = args.fold_variable_name,
        test_fold_index = args.test_fold_index,
        use_validation_set = args.use_validation_set
    )
    splits_dict = {}
    for split_name, split in zip(['train','val','test'], train_val_test_splits):
        if split_name=='val' and not args.use_validation_set: continue
        splits_dict[split_name] = {}
        splits_dict[split_name]['mutant_mutated_seq_pairs'] = list(zip(list(split['mutant']),list(split['mutated_sequence'])))
        raw_targets = {target_name: split[target_name] for target_name in target_names}
        if args.augmentation=="zero_shot_fitness_predictions_covariate": raw_targets['zero_shot_fitness_predictions'] = split['zero_shot_fitness_predictions']
        if split_name=="train":
            raw_targets, target_processing = preprocess_training_targets(raw_targets, args.target_config)
        else:
            raw_targets = preprocess_test_targets(raw_targets, args.target_config, target_processing)
        for target_name in target_names: 
            splits_dict[split_name][target_name] = raw_targets[target_name]
        if args.augmentation=="zero_shot_fitness_predictions_covariate": splits_dict[split_name]['zero_shot_fitness_predictions'] = raw_targets['zero_shot_fitness_predictions']
    # load dict into dataset objects
    train_data = Dataset.from_dict(splits_dict['train'])
    val_data = Dataset.from_dict(splits_dict['val']) if args.use_validation_set else None
    test_data = Dataset.from_dict(splits_dict['test'])
    return train_data, val_data, test_data, target_processing

def preprocess_training_targets(training_targets, target_config):
    """
    training_targets: dict of tensors of target values. Assumed to be 1D continuous values or 1D categorical values
    
    returns:
        - standard scaled versions of continuous targets
        - one-hot encoding versions of categorical targets with 1 extra column for masked token
    """
    target_processing = {}
    for target_name in training_targets.keys():
        if (target_name in target_config and target_config[target_name]["type"]=='continuous') or (target_name=='zero_shot_fitness_predictions'):
            # Standard scale
            target_processing[target_name]={}
            target_processing[target_name]['mean']=np.nanmean(training_targets[target_name])
            target_processing[target_name]['std']=np.nanstd(np.array(training_targets[target_name]))
            target_processing[target_name]['P95']=np.nanquantile(np.array(training_targets[target_name]), q=0.95)
            print("Target processing train set: {}".format(target_processing))
            training_targets[target_name] = (training_targets[target_name] - target_processing[target_name]['mean']) / target_processing[target_name]['std']
        else:
            # One-hot encoding
            target_processing[target_name]={}
            unique_categories = set(training_targets[target_name].dropna().unique())
            assert target_config[target_name]["dim"]==len(unique_categories), "list_dim_input_targets not properly referenced for target indexed: {}".format(target_name)
            category_to_index = dict((c, i) for i, c in enumerate(unique_categories))
            index_to_category = dict((i, c) for i, c in enumerate(unique_categories))
            category_to_index['<mask>'] = len(unique_categories) #Set <mask> category to largest index value
            index_to_category[len(unique_categories)] = '<mask>'
            training_targets[target_name] = torch.tensor([category_to_index[val] for val in training_targets[target_name]])
            target_processing[target_name]['category_to_index'] = category_to_index
            target_processing[target_name]['index_to_category'] = index_to_category
    return training_targets, target_processing

def preprocess_test_targets(test_targets, target_config, target_processing):
    """
    Applies target processing learned from training data
    """
    for target_name in test_targets.keys():
        if (target_name in target_config and target_config[target_name]["type"]=='continuous') or (target_name=='zero_shot_fitness_predictions'):
            test_targets[target_name] = (test_targets[target_name] - target_processing[target_name]['mean']) / target_processing[target_name]['std']
        else:
            test_targets[target_name] = torch.tensor([target_processing[target_name]['category_to_index'][val] for val in test_targets[target_name]])
    return test_targets

def mask_protein_sequences(inputs, alphabet, proba_aa_mask=0.15, proba_random_mutation=0.1, proba_unchanged=0.1):
    """
    Masks amino acids in the MSA at random with proba proba_aa_mask (15% by default).
    inputs: batched tokens (ie., MSA post tokenization)
    Returns the masked batch tokens and the corresponding labels for MLM.
    Adapted from HuggingFace transformers library.
    """
    labels = inputs.clone() # B, N, C
    all_special_tokens = torch.tensor([alphabet.tok_to_idx[x] for x in alphabet.all_special_tokens])
    probability_tensor = torch.full(labels.shape, proba_aa_mask)
    # Ensure we do not mask any special token
    special_tokens_mask = torch.isin(labels,all_special_tokens)
    probability_tensor.masked_fill_(special_tokens_mask, value=0.0)

    masked_indices = torch.bernoulli(probability_tensor).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens (consequently, all special tokens above will be all excluded from the loss automatically --> including any token that was set to <mask> beforehand (eg., missing values))

    # proba_actual_masking of the time (80% by default), we replace masked input tokens with tokenizer.mask_token (<mask>)
    proba_actual_masking = 1.0 - proba_random_mutation - proba_unchanged
    indices_replaced = torch.bernoulli(torch.full(labels.shape, proba_actual_masking)).bool() & masked_indices
    inputs[indices_replaced] = alphabet.mask_idx

    # proba_random_mutation of the time (10% by default), we replace masked input tokens with random word
    if proba_random_mutation > 0:
        rescaled_proba = proba_random_mutation / (proba_random_mutation + proba_unchanged)
        indices_random = torch.bernoulli(torch.full(labels.shape, rescaled_proba)).bool() & masked_indices & ~indices_replaced
        esm_alphabet = True if '<cls>' in alphabet.all_special_tokens else False
        low, high = (4,29) if esm_alphabet else (5,24)
        random_tokens = torch.randint(low=low, high=high, size=labels.shape, dtype=torch.long) #we ensure we dont replace by a special token. high in torch.randint is exclusive.
        inputs[indices_random] = random_tokens[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels, masked_indices
    
def mask_targets(inputs, input_target_type , target_processing, proba_target_mask=0.15, proba_random_mutation=0.1, proba_unchanged=0.1, min_num_labels_masked=None):
    """
    Masks targets at random with proba proba_target_mask.
    inputs is a tensor with the target values for a *single* target.
    The masked batch targets are also concatenated with the corresponding mask tensor (needed before embedding):
        - For continuous targets, masked elements are set to zero (mean of standard normal) and we concatenate a '1' mask value (non-masked elements are left unchnaged & we concatenate a '0' mask value)
        - For discrete targets, we assume that the vocabulary has been extended so that the last category value corresponds to the make token (handled by the target preprocessing function). 
          We simply set that value to 1 (all other values to 0, as per one-hot encoding scheme)
    Returns the masked batch target and the mask tensor.
    Note: Unlike for tokens, a masked target is encoded by a "1.0" (not alphabet.tok_to_idx['<mask>'])
    """
    labels = inputs.clone()
    indices_missing_values = torch.isnan(inputs)
    probability_tensor = torch.full(inputs.shape, proba_target_mask)
    probability_tensor.masked_fill_(indices_missing_values, value=0.0) #This will help ensure we do not mask missing values (we have no ground truth values to compute the loss against for these)
    masked_indices = torch.bernoulli(probability_tensor).bool() #tensor indicating which entries should be masked and used in loss computation.
    
    if min_num_labels_masked is not None and min_num_labels_masked > 0: #Routine to help enforce we mask at least min_num_labels_masked labels, eg., to ensure we do compute a prediction loss
        if masked_indices.int().sum().item() < min_num_labels_masked: #We sample additional masks as needed
            num_additional_masks_needed = min_num_labels_masked - masked_indices.int().sum().item()
            non_masked_indices_positions = np.arange(len(masked_indices))[np.array(~masked_indices)] #convert to numpy boolean as torch does not handle well boolean arrays of size 1
            selected_indices = np.random.choice(non_masked_indices_positions, size = int(num_additional_masks_needed))
            masked_indices = np.array(masked_indices)
            masked_indices[selected_indices] = True
            masked_indices = torch.tensor(masked_indices)
    labels[~masked_indices] = -100 #All non masked items (incl. missing values) are excluded from loss computation.
    
    # proba_actual_masking of the time (80% by default), we replace masked input tokens with tokenizer.mask_token (<mask>)
    proba_actual_masking = 1.0 - proba_random_mutation - proba_unchanged
    indices_replaced = torch.bernoulli(torch.full(labels.shape, proba_actual_masking)).bool() & masked_indices
    if input_target_type=='continuous':
        inputs[indices_replaced] = 0 # Set masked values to mean (ie., zero)
        inputs[indices_missing_values] = 0 # Set missing values to mean (ie., zero)
    else:
        inputs[indices_replaced] = target_processing['category_to_index']['<mask>']
        inputs[indices_missing_values] = target_processing['category_to_index']['<mask>']
    # proba_random_mutation of the time (10% by default), we replace masked input tokens with random word
    if proba_random_mutation > 0:
        rescaled_proba = proba_random_mutation / (proba_random_mutation + proba_unchanged)
        indices_random = torch.bernoulli(torch.full(labels.shape, rescaled_proba)).bool() & masked_indices & ~indices_replaced
        if input_target_type=='continuous':
            random_tokens = torch.tensor(np.random.normal(loc=0.0, scale=1.0, size=labels.shape)).float()
        else:
            random_tokens = torch.randint(low=0, high=target_processing['category_to_index']['<mask>'], size=labels.shape, dtype=torch.long)
        inputs[indices_random] = random_tokens[indices_random]
    
    # concatenate mask value to input tensor (only for continuous; by design we reserved the last category index to <mask> so will be automatically handled in Embedding layer)
    if input_target_type=='continuous':
        indices_masked_or_missing = masked_indices | indices_missing_values
        inputs = torch.cat((inputs.view(-1,1), indices_masked_or_missing.float().view(-1,1)), dim=-1)
    #inputs is the modified version of input in which masked tokens are zeroed out / imputed (to be passed to the model as input). Labels corresponds ground truth values to be predicted at masked positions (-100 otherwise)
    return inputs, labels 

def collate_fn_protein_npt(raw_batch):
    keys = raw_batch[0].keys()
    target_keys = list(set(keys)-set(['mutant_mutated_seq_pairs'])) #target_keys also includes zero-shot fitness predictions if we pass that as input as well
    batch = {}
    for key in keys:
        batch[key] = []
    for item in raw_batch:
        for key in keys:
            batch[key].append(item[key])
    # We convert to tensors all targets (all numeric)
    for key in target_keys:
        batch[key] = torch.tensor(batch[key])
    return batch

def get_optimal_window(mutation_position_relative, seq_len_wo_special, model_window):
    """
    Helper function that selects an optimal sequence window that fits the maximum model context size.
    If the sequence length is less than the maximum context size, the full sequence is returned.
    """
    half_model_window = model_window // 2
    if seq_len_wo_special <= model_window:
        return [0,seq_len_wo_special]
    elif mutation_position_relative < half_model_window:
        return [0,model_window]
    elif mutation_position_relative >= seq_len_wo_special - half_model_window:
        return [seq_len_wo_special - model_window, seq_len_wo_special]
    else:
        window_start = max(0,mutation_position_relative-half_model_window)
        window_end = min(seq_len_wo_special,mutation_position_relative+half_model_window)
        if (window_end - window_start) < model_window: # ensures all sequences will be same lengths (catch all for edge cases, due to non integer half_model_window)
            window_end +=1 
        return [window_start, window_end]

def slice_sequences(list_mutant_mutated_seq_pairs, max_positions=1024, method="rolling", rolling_overlap=100, eval_mode=True, batch_target_labels=None, batch_masked_targets=None, target_names=None, start_idx=1, num_extra_tokens=1):
    """
    rolling: creates overlapping sequence chunks of length args.max_positions - 1 (minus 1 to allow the BOS token addition)
    center: centers sequence slice around mutation
    left: selects the first (args.max_positions - 1) tokens in the sequence
    batch_target_labels are needed in eval_mode with rolling as we do target duplication for the different windows.
    Assumption: all input sequences are of same length.
    num_extra_tokens: 1 is just BOS added (eg., ESM); 2 if BOS and EOS added (eg., Tranception)
    """
    mutant_mutated_seqs = list(zip(*list_mutant_mutated_seq_pairs))
    raw_sequence_length = len(mutant_mutated_seqs[1][0]) # length of first sequence
    all_mutants = mutant_mutated_seqs[0]
    all_mutated_seqs = mutant_mutated_seqs[1]
    scoring_optimal_window = None
    if method=="center":
        mutations_barycenters = [int(np.array([ int(mutation[1:-1]) - start_idx for mutation in mutant.split(':')]).mean()) for mutant in all_mutants]
        scoring_optimal_window = [get_optimal_window(x, raw_sequence_length, max_positions - num_extra_tokens) for x in mutations_barycenters] #Removing 1 from args.max_positions to allow subsequent addition of BOS token
        sliced_mutated_seqs = [all_mutated_seqs[index][scoring_optimal_window[index][0]:scoring_optimal_window[index][1]] for index in range(len(all_mutated_seqs))]
        list_mutant_mutated_seq_pairs = list(zip(all_mutants,sliced_mutated_seqs))
    elif method=="left":
        sliced_mutated_seqs = [all_mutated_seqs[index][0:max_positions - num_extra_tokens] for index in range(len(all_mutated_seqs))] #minus 1 to keep room for BOS token
        list_mutant_mutated_seq_pairs = list(zip(all_mutants,sliced_mutated_seqs))
        scoring_optimal_window = [(0, max_positions - 1)] * len(all_mutated_seqs)
    else:
        print("Sequence slicing method not recognized")
        sys.exit(0)
    if batch_masked_targets is not None: #Protein NPT output
        return list_mutant_mutated_seq_pairs, batch_target_labels, batch_masked_targets, scoring_optimal_window
    else: #Baseline output
        return list_mutant_mutated_seq_pairs, batch_target_labels, scoring_optimal_window

def get_indices_retrieved_embeddings(batch, embeddings_dict_location, number_of_mutated_seqs_to_score=None):
    batch_mutants, batch_sequences = zip(*batch['mutant_mutated_seq_pairs'])
    with h5py.File(embeddings_dict_location, 'r') as h5f:
        num_all_embeddings = len(h5f['mutants'])
        list_mutants = [x.decode('utf-8') for x in h5f['mutants'][:]]
        mutant_indices = range(num_all_embeddings)
    mutants_embeddings_dict = {'mutants': list_mutants, 'mutant_index': mutant_indices}
    mutants_embeddings_df = pd.DataFrame.from_dict(mutants_embeddings_dict, orient='columns')
    if number_of_mutated_seqs_to_score is not None:
        batch_mutants = batch_mutants[:number_of_mutated_seqs_to_score]
    batch_mutants_df = pd.DataFrame(batch_mutants, columns=['mutants'])
    intersection = pd.merge(batch_mutants_df, mutants_embeddings_df, how='inner', on='mutants')
    return np.array(intersection['mutant_index'].values)

def pnpt_spearmanr(prediction,target):
    mask_missing_values = np.isnan(target) | np.equal(target, -100) #In PNPT missing values are never masked so corresponding labels are always set to -100
    return spearmanr(prediction[~mask_missing_values], target[~mask_missing_values])[0] #first value is spearman rho, second is the corresponding p-value           
    
def pnpt_count_non_nan(x):
    missing_mask = np.isnan(x) | np.equal(x,-100)
    return np.count_nonzero(~missing_mask)