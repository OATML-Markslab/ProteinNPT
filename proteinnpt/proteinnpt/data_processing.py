import os
import sys
import numpy as np
import torch
import h5py
from ..utils.data_utils import mask_targets, mask_protein_sequences, slice_sequences, get_indices_retrieved_embeddings
from ..utils.msa_utils import weighted_sample_MSA

def PNPT_sample_training_points_inference(training_sequences, sequences_sampling_method, num_sampled_points):
    replace = True if "with_replacement" in sequences_sampling_method else False
    num_sequences_training_data = len(training_sequences['mutant_mutated_seq_pairs'])
    if (num_sequences_training_data <= num_sampled_points) and (not replace): return range(num_sequences_training_data)
    if "random" in sequences_sampling_method:
        weights = None
    elif "distributed_positions" in sequences_sampling_method:
        print("Predicting with training observations sampled based on positions")
        training_mutants, _ = zip(*training_sequences['mutant_mutated_seq_pairs'])
        training_positions = [int(x[1:-1]) for x in training_mutants]
        weights = [training_positions.count(x) for x in training_positions]
        weights = 1 / np.array(weights)
        weights = weights / weights.sum()
    elif "distributed_target" in sequences_sampling_method:
        print("Predicting with training observations sampled based on target values")
        fitness_values = np.array(training_sequences['fitness'])
        fitness_values_min = fitness_values.min()
        fitness_values_max = fitness_values.max()
        delta_fitness = (fitness_values_max - fitness_values_min) / 20
        shifted_fitness_values = fitness_values - fitness_values_min
        fitness_buckets = [int(x) for x in shifted_fitness_values / delta_fitness]
        weights = [fitness_buckets.count(x) for x in fitness_buckets]
        weights = 1 / np.array(weights)
        weights = weights / weights.sum()
    return np.random.choice(range(num_sequences_training_data), replace=replace, p=weights, size = num_sampled_points)

def process_batch(batch, model, alphabet, args, device, MSA_sequences=None, MSA_weights=None, MSA_start_position=None, MSA_end_position=None, target_processing=None, training_sequences = None, proba_target_mask = 0.15, proba_aa_mask = 0.15, eval_mode = True, start_idx=1, selected_indices_seed=0, indel_mode=False):
    """
    If MSA_sequences is not None, then we sample args.num_MSA_sequences_per_training_instance sequences from it that we add to the batch. 
    
    If training_sequences is not None (eg., in eval mode), then we complement the data with training instances. Each batch instance will have:
        - eval_num_sequences_to_score_per_batch_per_gpu sequences with no label (that we want to score)
        - eval_num_training_sequences_per_batch_per_gpu sequences from training (that we want to use as context)
    """
    target_names = args.target_config.keys()
    target_names_unknown = [x for x in args.target_config.keys() if args.target_config[x]["in_NPT_loss"]]
    raw_sequence_length = len(batch['mutant_mutated_seq_pairs'][0][1])
    number_of_mutated_seqs_to_score = len(batch['mutant_mutated_seq_pairs']) 
    
    batch_masked_targets = {} 
    batch_target_labels = {} 
    for target_name in target_names:
        if target_name not in batch: 
            print("Target values were not passed in input batch. We assume all corresponding values are missing & to be predicted.")
            batch[target_name] = torch.tensor([np.nan] * number_of_mutated_seqs_to_score) # By construction this will set all labels to -100 in subsequent mask_targets.
            eval_mode = True
        if target_name in target_names_unknown:
            masked_targets, target_labels = mask_targets(
                inputs = batch[target_name], 
                input_target_type = args.target_config[target_name]["type"], 
                target_processing = target_processing[target_name], 
                proba_target_mask = proba_target_mask if not eval_mode else 1.0, #In eval mode we mask all input targets
                proba_random_mutation = 0.1 if not eval_mode else 0.0, #No random mutation in eval mode
                proba_unchanged = 0.1 if not eval_mode else 0.0, #No unchanged token in eval mode (all masked)
                min_num_labels_masked = 1 if not eval_mode else None #Masking at least one label during training to have well-defined loss
            )
        else:
            masked_targets, target_labels = mask_targets(
                inputs = batch[target_name],
                input_target_type = args.target_config[target_name]["type"], 
                target_processing = target_processing[target_name], 
                proba_target_mask = 0.0,
                proba_random_mutation = 0.0,
                proba_unchanged = 1.0
            )
        batch_masked_targets[target_name] = masked_targets # masked_targets is of shape (number_of_mutated_seqs_to_score, 2) for continuous targets
        batch_target_labels[target_name] = target_labels # batch_target_labels is of shape (number_of_mutated_seqs_to_score,) for continuous targets

    if args.augmentation=="zero_shot_fitness_predictions_covariate": batch_target_labels['zero_shot_fitness_predictions'] = batch['zero_shot_fitness_predictions'].to(device) # process fitness pred as a target to make things easier
    if (training_sequences is not None):
        num_sequences_training_data = len(training_sequences['mutant_mutated_seq_pairs'])
        if model.training_sample_sequences_indices is None:
            selected_indices_dict = {}
            num_ensemble_seeds = model.PNPT_ensemble_test_num_seeds if model.PNPT_ensemble_test_num_seeds > 0 else 1
            for ensemble_seed in range(num_ensemble_seeds):
                selected_indices_dict[ensemble_seed] = PNPT_sample_training_points_inference(
                    training_sequences=training_sequences, 
                    sequences_sampling_method=args.eval_training_sequences_sampling_method, 
                    num_sampled_points=args.eval_num_training_sequences_per_batch_per_gpu
                )
            model.training_sample_sequences_indices = selected_indices_dict
        selected_indices = model.training_sample_sequences_indices[selected_indices_seed]
        num_selected_training_sequences = len(np.array(training_sequences['mutant_mutated_seq_pairs'])[selected_indices])
        batch['mutant_mutated_seq_pairs'] += list(np.array(training_sequences['mutant_mutated_seq_pairs'])[selected_indices])
        if args.augmentation=="zero_shot_fitness_predictions_covariate":
            batch_target_labels['zero_shot_fitness_predictions'] = list(batch_target_labels['zero_shot_fitness_predictions'].cpu().numpy())
            batch_target_labels['zero_shot_fitness_predictions'] += list(np.array(training_sequences['zero_shot_fitness_predictions'])[selected_indices])
            batch_target_labels['zero_shot_fitness_predictions'] = torch.tensor(batch_target_labels['zero_shot_fitness_predictions']).float().to(device)
        for target_name in target_names:
            # training_sequences[target_name] expected of size (len_training_seq,2). No entry is actually masked here since we want to use all available information to predict as accurately as possible
            masked_training_targets, training_target_labels = mask_targets(
                inputs = torch.tensor(np.array(training_sequences[target_name])[selected_indices]),
                input_target_type = args.target_config[target_name]["type"], 
                target_processing = target_processing[target_name], 
                proba_target_mask = 0.0,
                proba_random_mutation = 0.0,
                proba_unchanged = 1.0
            )
            batch_masked_targets[target_name] = torch.cat( [batch_masked_targets[target_name], masked_training_targets], dim=0).float().to(device)
            num_all_mutated_sequences_input = number_of_mutated_seqs_to_score + num_selected_training_sequences
            assert batch_masked_targets[target_name].shape[0] == num_all_mutated_sequences_input, "Error adding training data to seqs to score: {} Vs {}".format(batch_masked_targets[target_name].shape[0], num_all_mutated_sequences_input)
            batch_target_labels[target_name] = torch.cat( [batch_target_labels[target_name], training_target_labels]).float().to(device) 
            assert batch_masked_targets[target_name].shape[0] == batch_target_labels[target_name].shape[0], "Lengths of masked targets and target labels do not match: {} Vs {}".format(batch_masked_targets[target_name].shape[0], batch_target_labels[target_name].shape[0])
    else:
        for target_name in target_names:
            batch_masked_targets[target_name] = batch_masked_targets[target_name].to(device)
            batch_target_labels[target_name] = batch_target_labels[target_name].to(device)
        num_all_mutated_sequences_input = number_of_mutated_seqs_to_score
        num_selected_training_sequences = 0

    # Embedding loading needs to happen here to ensure we also load training sequences at eval time
    if args.sequence_embeddings_location is not None:
        assert os.path.exists(args.sequence_embeddings_location), f"Sequence embeddings location doesn't exist: {args.sequence_embeddings_location}"
        try:
            indices_retrieved_embeddings = get_indices_retrieved_embeddings(batch,args.sequence_embeddings_location)
            assert len(indices_retrieved_embeddings)==len(batch['mutant_mutated_seq_pairs']) , "At least one embedding was missing"
            with h5py.File(args.sequence_embeddings_location, 'r') as h5f:
                sequence_embeddings = torch.tensor(np.array([h5f['embeddings'][i] for i in indices_retrieved_embeddings])).float()
        except Exception as e:
            print("Error loading main sequence embeddings:", e)
            sys.exit(0)
    else:
        sequence_embeddings = None

    # If MSAT and MSA does not cover full sequence length, we chop off all sequences to be scored as needed so that everything lines up properly.
    if (args.aa_embeddings == "MSA_Transformer") and (MSA_start_position is not None) and (MSA_end_position is not None) and ((MSA_start_position > 1) or (MSA_end_position < raw_sequence_length)): # and args.sequence_embeddings_location is None:
        MSA_start_index = MSA_start_position - 1
        MSA_end_index = MSA_end_position
        batch['mutant_mutated_seq_pairs'] = [ (mutant,seq[MSA_start_index:MSA_end_index]) for (mutant,seq) in batch['mutant_mutated_seq_pairs']]
        # Recompute sequence length (has potentially been chopped off above)
        raw_sequence_length = len(batch['mutant_mutated_seq_pairs'][0][1])

    # Sample MSA sequences - append to end of batch tokens and targets
    # We use these only if aa_embedding is MSA_Transformer. Otherwise we dont use these seqs.
    if args.aa_embeddings == "MSA_Transformer" and args.sequence_embeddings_location is None and args.num_MSA_sequences_per_training_instance > 0:
        assert MSA_weights is not None, "Trying to add MSA_sequences to scoring batch but no weights are provided"
        if model.MSA_sample_sequences is None:
            model.MSA_sample_sequences = weighted_sample_MSA(
                MSA_all_sequences=MSA_sequences, 
                MSA_non_ref_sequences_weights=MSA_weights, 
                number_sampled_MSA_sequences=args.num_MSA_sequences_per_training_instance
            )
        # Concatenate MSA sequences with labelled assay sequences
        batch['mutant_mutated_seq_pairs'] += model.MSA_sample_sequences
    
    # Slice sequences around mutation if sequence longer than context length
    if args.max_positions is not None and raw_sequence_length + 1 > args.max_positions: # Adding one for the BOS token
        if args.long_sequences_slicing_method=="center" and args.aa_embeddings=="MSA_Transformer":
            print("Center slicing method not adapted to MSA Transformer embedding as sequences would not be aligned in the same system anymore. Defaulting to 'left' mode.")
            args.long_sequences_slicing_method="left"
        batch['mutant_mutated_seq_pairs'], batch_target_labels, batch_masked_targets, batch_scoring_optimal_window = slice_sequences(
            list_mutant_mutated_seq_pairs = batch['mutant_mutated_seq_pairs'], 
            max_positions=args.max_positions,
            method=args.long_sequences_slicing_method,
            rolling_overlap=args.max_positions//4,
            eval_mode=eval_mode,
            batch_target_labels=batch_target_labels,
            batch_masked_targets=batch_masked_targets,
            start_idx=start_idx,
            target_names=target_names,
            num_extra_tokens=2 if args.aa_embeddings=="Tranception" else 1
        )
    else:
        batch_scoring_optimal_window = None

    # Tokenize protein sequences
    if args.aa_embeddings == "MSA_Transformer" and num_all_mutated_sequences_input > 1 and args.sequence_embeddings_location is None: 
        #Re-organize list of sequences to have training_num_assay_sequences_per_batch_per_gpu MSA batches, where in each the sequence to score is the first and the rest are the sampled MSA sequences.
        num_sequences = num_all_mutated_sequences_input + args.num_MSA_sequences_per_training_instance
        assert len(batch['mutant_mutated_seq_pairs']) == num_sequences, "Unexpected number of sequences"
        all_mutated_sequences_input = batch['mutant_mutated_seq_pairs'][:num_all_mutated_sequences_input]
        MSA_sequences = batch['mutant_mutated_seq_pairs'][num_all_mutated_sequences_input:]
        batch['mutant_mutated_seq_pairs'] = [ [sequence] + MSA_sequences for sequence in all_mutated_sequences_input]

    if args.aa_embeddings in ["MSA_Transformer", "Linear_embedding"] or args.aa_embeddings.startswith("ESM"):
        token_batch_converter = alphabet.get_batch_converter()
        batch_sequence_names, batch_AA_sequences, batch_token_sequences = token_batch_converter(batch['mutant_mutated_seq_pairs'])
        if args.aa_embeddings=="MSA_Transformer" and args.sequence_embeddings_location is not None: #If loading MSAT embeddings from disk, we drop the MSA dimension (done already if not MSAT via the different tokenizer)
            num_MSAs_in_batch, num_sequences_in_alignments, seqlen = batch_token_sequences.size()
            batch_token_sequences = batch_token_sequences.view(num_sequences_in_alignments, seqlen) #drop the dummy batch dimension from the tokenizer when using ESM1v / LinearEmbedding
    elif args.aa_embeddings == "Tranception":
        _, sequence = zip(*batch['mutant_mutated_seq_pairs'])
        batch_token_sequences = torch.tensor(model.alphabet(sequence, add_special_tokens=True, truncation=True, padding=True, max_length=model.aa_embedding.config.n_ctx)['input_ids'])
        
    # Mask protein sequences
    batch_masked_tokens, batch_token_labels, masked_indices = mask_protein_sequences(
        inputs = batch_token_sequences, 
        alphabet = alphabet, 
        proba_aa_mask = proba_aa_mask if not eval_mode else 0.0, #In eval mode we do not mask any token
        proba_random_mutation = 0.1, 
        proba_unchanged = 0.1
    )
    if args.sequence_embeddings_location is not None:
        if sequence_embeddings.shape[1] > masked_indices.shape[1]: # When dealing with sequences of different sizes, and sequences in batch happen to be all smaller than longest sequence in assay for which we computed embeddings
            extra_padding_in_embeddings = (sequence_embeddings.shape[1] - masked_indices.shape[1])
            sequence_embeddings = sequence_embeddings[:,:-extra_padding_in_embeddings]
        sequence_embeddings[masked_indices] = 0.0

    batch_masked_tokens = batch_masked_tokens.to(device)
    batch_token_labels = batch_token_labels.to(device)
    processed_batch = {
        'masked_tokens': batch_masked_tokens,
        'token_labels': batch_token_labels,
        'masked_targets': batch_masked_targets,
        'target_labels': batch_target_labels,
        'mutant_mutated_seq_pairs': batch['mutant_mutated_seq_pairs'],
        'num_all_mutated_sequences_input': num_all_mutated_sequences_input,
        'num_of_mutated_seqs_to_score': number_of_mutated_seqs_to_score,
        'num_selected_training_sequences': num_selected_training_sequences,
        'sequence_embeddings': sequence_embeddings
    }
    return processed_batch