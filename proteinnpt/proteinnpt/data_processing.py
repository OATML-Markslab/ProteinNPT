import os
import sys
import numpy as np
import torch
import h5py
from ..utils.data_utils import mask_targets, mask_protein_sequences, get_indices_retrieved_embeddings
from ..utils.embedding_utils import process_MSA_Transformer_batch, process_ESM_batch, process_Tranception_batch

def PNPT_sample_training_points_inference(training_sequences, sequences_sampling_method, num_sampled_points):
    """Samples training points for inference using various sampling strategies.

    Args:
        training_sequences (dict): Dictionary containing training data including:
            - mutant_mutated_seq_pairs: List of (mutant, sequence) pairs
            - fitness: List of fitness values (required for distributed_target sampling)
        sequences_sampling_method (str): Sampling strategy to use:
            - 'random': Uniform random sampling
            - 'distributed_positions': Sample based on mutation position distribution
            - 'distributed_target': Sample based on target value distribution
        num_sampled_points (int): Number of points to sample

    Returns:
        numpy.ndarray: Indices of selected training points
    """
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

def process_batch(batch, model, alphabet, args, device, MSA_sequences=None, MSA_weights=None, MSA_start_position=None, MSA_end_position=None, target_processing=None, training_sequences=None, proba_target_mask=0.15, proba_aa_mask=0.15, eval_mode=True, start_idx=1, selected_indices_seed=0, indel_mode=False):
    """Processes a batch of protein sequences and their associated targets for model input.

    This function handles multiple aspects of batch processing including:
    - Target masking and processing
    - MSA sequence sampling and integration
    - Sequence embedding loading and processing
    - Sequence length adjustment and slicing
    - Tokenization and masking of protein sequences

    Args:
        batch (dict): Input batch containing:
            - mutant_mutated_seq_pairs: List of (mutant, sequence) pairs
            - Additional target values as specified in args.target_config
        model: The Protein NPT model instance
        alphabet: Tokenizer alphabet for protein sequences
        args: Configuration object containing model parameters
        device: PyTorch device (CPU/GPU)
        MSA_sequences (list, optional): List of MSA sequences
        MSA_weights (numpy.ndarray, optional): Weights for MSA sequence sampling
        MSA_start_position (int, optional): Start position for MSA sequence slice
        MSA_end_position (int, optional): End position for MSA sequence slice
        target_processing (dict, optional): Target processing configurations
        training_sequences (dict, optional): Training sequences to include as context
        proba_target_mask (float, optional): Probability of masking target values. Defaults to 0.15
        proba_aa_mask (float, optional): Probability of masking amino acids. Defaults to 0.15
        eval_mode (bool, optional): Whether to run in evaluation mode. Defaults to True
        start_idx (int, optional): Starting index for sequence processing. Defaults to 1
        selected_indices_seed (int, optional): Seed for sampling training sequences. Defaults to 0
        indel_mode (bool, optional): Whether to handle insertions/deletions. Defaults to False

    Returns:
        dict: Processed batch containing:
            - masked_tokens: Masked protein sequences
            - token_labels: Original tokens for masked positions
            - masked_targets: Masked target values
            - target_labels: Original target values
            - mutant_mutated_seq_pairs: Original sequence pairs
            - num_all_mutated_sequences_input: Total number of sequences
            - num_of_mutated_seqs_to_score: Number of sequences to score
            - num_selected_training_sequences: Number of training sequences
            - sequence_embeddings: Pre-computed sequence embeddings if available
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
        try:
            sequence_embeddings_list = []
            for embedding_location in args.sequence_embeddings_location:
                indices_retrieved_embeddings = get_indices_retrieved_embeddings(batch,embedding_location)
                assert len(indices_retrieved_embeddings)==len(batch['mutant_mutated_seq_pairs']), "At least one embedding was missing"
                with h5py.File(embedding_location, 'r') as h5f:
                    sequence_embeddings = torch.tensor(np.array([h5f['embeddings'][i] for i in indices_retrieved_embeddings])).float()
                sequence_embeddings_list.append(sequence_embeddings)
            if len(sequence_embeddings_list)==1:
                sequence_embeddings = sequence_embeddings_list[0]
            else:
                sequence_embeddings = torch.cat(sequence_embeddings_list,dim=-1) #concatenate embeddings over the last dimension --> B, L, D1 + D2
        except:
            print("Error loading main sequence embeddings")
            sys.exit(0)
    else:
        sequence_embeddings = None
    
    if args.aa_embeddings=="MSA_Transformer" and args.sequence_embeddings_folder is None:
        processed_batch = process_MSA_Transformer_batch(
            batch = batch, 
            model = model, 
            alphabet = alphabet, 
            max_positions = args.max_positions, 
            MSA_start_position = MSA_start_position, 
            MSA_end_position = MSA_end_position, 
            MSA_weights = MSA_weights, 
            MSA_sequences = MSA_sequences,
            num_MSA_sequences = args.num_MSA_sequences_per_training_instance,
            eval_mode = eval_mode, 
            start_idx = start_idx, 
            long_sequences_slicing_method = args.long_sequences_slicing_method, 
            indel_mode = indel_mode, 
            fast_MSA_mode = False, 
            clustalomega_path = args.clustalomega_path, 
            batch_target_labels = batch_target_labels,
            batch_masked_targets = batch_masked_targets,
            target_names = target_names,
            num_extra_tokens=1
        )
        if sequence_embeddings is not None:
            # Drop unnecessary dimension if embedding model is MSAT and embeddings are retrieved from disk
            num_MSAs_in_batch, num_sequences_in_alignments, seqlen = processed_batch['input_tokens'].size()
            processed_batch['input_tokens'] = processed_batch['input_tokens'].view(num_sequences_in_alignments, seqlen)
    elif args.aa_embeddings == "Tranception":
        processed_batch = process_Tranception_batch(
                    batch = batch, 
                    model = model
                    )
        mutant, sequence = zip(*batch['mutant_mutated_seq_pairs'])
        tokenized_batch = model.aa_embedding.config.tokenizer(sequence, add_special_tokens=True, truncation=True, padding=True, max_length=model.aa_embedding.config.n_ctx)
        for k, v in tokenized_batch.items():
            processed_batch[k] = torch.tensor(v).to(model.device)
        processed_batch['input_tokens']=processed_batch['input_ids']
    else: #Default processing
        processed_batch = process_ESM_batch(
            batch = batch, 
            model = model, 
            alphabet = alphabet, 
            max_positions = args.max_positions, 
            long_sequences_slicing_method = args.long_sequences_slicing_method, 
            eval_mode = eval_mode, 
            start_idx = start_idx,
            num_extra_tokens = 2
        )
        if args.aa_embeddings=="MSA_Transformer" and args.sequence_embeddings_folder is not None and (MSA_start_position is not None) and (MSA_end_position is not None) and ((MSA_start_position > 1) or (MSA_end_position < raw_sequence_length)):
            MSA_start_index = MSA_start_position - 1
            MSA_end_index = MSA_end_position
            processed_batch['mutant_mutated_seq_pairs'] = [ (mutant,seq[MSA_start_index:MSA_end_index]) for (mutant,seq) in processed_batch['mutant_mutated_seq_pairs']]
            # Recompute sequence length (has potentially been chopped off above)
            raw_sequence_length = len(processed_batch['mutant_mutated_seq_pairs'][0][1])
            processed_batch['input_tokens'] = processed_batch['input_tokens'][:,1+MSA_start_index:1+MSA_end_index] #Adding 1 for the BOS token in input_tokens (from ESM embeddings)
    
    # Mask protein sequences
    batch_masked_tokens, batch_token_labels, masked_indices = mask_protein_sequences(
        inputs = processed_batch['input_tokens'].cpu(), 
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
        'mutant_mutated_seq_pairs': batch['mutant_mutated_seq_pairs'],
        'num_all_mutated_sequences_input': num_all_mutated_sequences_input,
        'num_of_mutated_seqs_to_score': number_of_mutated_seqs_to_score,
        'num_selected_training_sequences': num_selected_training_sequences,
        'sequence_embeddings': sequence_embeddings
    }
    if 'target_labels' not in processed_batch.keys(): processed_batch['target_labels'] = batch_target_labels
    if 'masked_targets' not in processed_batch.keys(): processed_batch['masked_targets'] = batch_masked_targets
    return processed_batch